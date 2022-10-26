import torch
import datetime
import logging
import os
import torch.backends.cudnn as cudnn

import os.path as osp
import torch.nn as nn
import neptune.new as neptune 
from tqdm import tqdm
import operator 
import math
import torch.optim as optim
from utils.optimize import *
from easydict import EasyDict as edict
from utils import *
from utils.memory import * 
from utils.flatwhite import * 
from dataset import * 
from sklearn import metrics
import sklearn
from sklearn.cluster import KMeans
from utils_1 import print_info, torchutils

from torch.utils.tensorboard import SummaryWriter

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.manifold import TSNE


class BaseTrainer(object):
    def __init__(self, config,  writer):
        torchutils.set_seed(self.config.seed)
        self.model = None
        self.optim = None

        self.logger = logging.getLogger("Agent")
        self._choose_device()
        self._create_model()
        print('模型载入完成')
        #self._create_optimizer()
        print('优化器载入完成')
        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0
        self.val_acc = []
        self.train_loss = []
        self.lr_scheduler_list = []

        # we need these to decide best loss
        self.current_loss = 0.0
        self.current_val_metric = 0.0
        self.best_val_metric = 0.0
        self.best_val_epoch = 0
        self.iter_with_no_improv = 0        
        
                        
        self.config = config
        self.writer = writer
        if self.config.neptune:
            name = self.config['note'] + '_' + self.config.source + '_' + self.config.target
            self.run = neptune.init(project='solacex/UniDA-Extension', name=name,  source_files=[], capture_hardware_metrics=False)#, mode="offline")
            self.run['config'] = self.config
            self.run['name'] = self.config['note'] + '_' + self.config.source + '_' + self.config.target
        if self.config.tensorboard:
            self.writer  = SummaryWriter(osp.join(self.config.snapshot, 'log'))
        self.best = 0.0
        self.acc_best_h = 0.0
        self.h_best_acc = 0.0 
        self.k_best = 0.0
        self.h_best = 0.0
        self.label_mask = None
        self.k_converge=False
        self.score_vec = None 
        self.test_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=100, test=True, validate=True)
        if self.config.task=='imagenet-caltech':
            self.src_loader = get_cls_sep_dataset(self.config, self.config.source, self.config.source_classes, batch_size=100, test=True)
        else:
            self.src_loader = get_dataset(self.config, self.config.source, self.config.source_classes, batch_size=100, test=True)
        self.tgt_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=100, test=True)
        self.best_prec = 0.0
        self.best_recall = 0.0 

    def forward(self):
        pass
    def backward(self):
        pass

    def iter(self):
        pass
    def train(self):
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter % self.config.print_freq ==0:
                self.print_loss(i_iter)
            if i_iter % self.config.save_freq ==0 and i_iter != 0:
                self.save_model(i_iter)
            if self.config.val and i_iter % self.config.val_freq ==0 and i_iter!=0:
                self.validate()

    def save_model(self, iter, name='last'):
        tmp_name = name + self.config.source[:2] + '_' + self.config.target[:2]+ '.pth'
        state = {'net':self.model.module.state_dict(), 'memory':self.memory.memory,
                 'cluster_mapping':self.cluster_mapping, 'global_label_set':self.global_label_set  }

        torch.save(state, osp.join(self.config.snapshot, tmp_name))
        
    def save_txt(self):
        with open(osp.join(self.config.snapshot, 'result.txt'), 'a') as f:
            f.write(self.config.source[:2] + '->' + self.config.target[:2] +'[best]: ' + str(self.best) + ' '+ str(self.k_best) + ' [H-Score]: '+ str(self.h_best) + ' ' +  str(self.acc_best_h) + ' '+ str(self.h_best_acc) + ' ' + str(self.best_prec) + ' ' + str(self.best_recall) + '\n')
            f.write(self.config.source[:2] + '->' + self.config.target[:2] +'[last]: ' + str(self.last) + ' '+ str(self.k_last) + ' [H-Score]: '+ str(self.h_last) + ' ' + str(self.last_prec) + ' ' + str(self.last_recall) + '\n')
        f.close()
    def neptune_metric(self, name, value, display=True):
        if self.config.neptune:
            self.run[name].log(value)
        if self.config.tensorboard:
            self.writer.add_scalar(name, value) 
        if display:
            print('{} is {:.2f}'.format(name, value))
            
    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor +'  '+ loss_infor)
        if self.config.neptune:
            for key in self.losses.keys():
                self.neptune_metric('train/'+key, self.losses[key].item(), False)
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/'+key, self.losses[key], iter)
    def print_acc(self, acc_dict):
        str_dict = [str(k) + ': {:.2f}'.format(v) for k, v in acc_dict.items() ]
        output = ' '.join(str_dict)
        print(output)
    def cos_simi(self, x1, x2):
        simi = torch.matmul(x1, x2.transpose(0, 1))
        return simi    

    def gather_feats(self,return_from_memory=False):
        # self.model.set_bn_domain(1)
        data_feat, data_gt, data_paths, data_probs  = [], [], [], []
        gts = []
        gt = {}
        preds = []
        names = []
        for _, batch in tqdm(enumerate(self.tgt_loader)):
            img, label, name, _ = batch
            names.extend(name)
            with torch.no_grad():
                _, output, _, prob = self.model(img.cuda())
            feature = output.squeeze()#view(1,-1)
            N, C = feature.shape
            data_feat.extend(torch.chunk(feature, N, dim=0))
            gts.extend(torch.chunk(label, N, dim=0))

        if return_from_memory:
            memory_bank_wrapper = self.get_attr("target", "memory_bank_wrapper")
            feats = memory_bank_wrapper.as_tensor()
            feats = F.normalize(feats, p=2, dim=-1)            
            gts = self.get_attr("target", "train_labels")
            
        for k,v in zip(names, gts):
            gt[k]=v.cuda()
        #feats =  torch.cat(data_feat, dim=0)
        #feats = F.normalize(feats, p=2, dim=-1)
        #_,t_gts，_
        return feats, gt, preds

    def validate(self, i_iter, class_set):
        print(self.config.source, self.config.target, self.config.note)
        print(self.global_label_set)
        if not self.config.prior:
            if self.config.num_centers == len(self.cluster_mapping):
                result = self.close_validate(i_iter)
            else:
                result = self.open_validate(i_iter)
        elif self.config.setting in ['uda', 'osda']:
            result = self.open_validate(i_iter)
        else:
            result = self.close_validate(i_iter)
        over_all, k, h_score, recall, precision = result 
        #over_all为acc k为known acc 
        if over_all > self.best:
            self.best = over_all
            self.k_best = k
            self.acc_best_h = h_score
        if h_score > self.h_best:
            self.h_best = h_score
            self.h_best_acc = over_all
            self.best_recall = recall
            self.best_prec = precision
        if i_iter+1 == self.config.stop_steps:
            self.last = over_all
            self.k_last = k
            self.h_last = h_score
            self.last_recall = recall
            self.last_prec = precision

        return result 

    def close_validate(self, i_iter):
        self.model.train(False)
        knows = 0.0
        unknows = 0.0
        k_co = 0.0
        uk_co = 0.0
        accs = GroupAverageMeter()
        test_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=100, test=True)
        common_index = torch.Tensor(self.global_label_set).cuda().long()

        for _, batch in tqdm(enumerate(test_loader)):
            acc_dict = {}
            img, label, name, _ = batch
            label = label.cuda()
            with torch.no_grad():
                _, neck, pred, pred2 = self.model(img.cuda())
 #           pred2 = pred2[:, common_index]
            pred_label =  pred2.argmax(dim=-1)
#            pred_label = common_index[pred_label]

            label = torch.where(label>=self.config.num_classes, torch.Tensor([self.config.num_classes]).cuda(),label.float())
            for i in label.unique().tolist():
                mask = label==i
                count = mask.sum().float()
                correct = (pred_label==label) * mask
                correct = correct.sum().float()

                acc_dict[i] = ((correct/count).item(), count.item())
            accs.update(acc_dict)
        acc = np.mean(list(accs.avg.values()))
        self.print_acc(accs.avg)
        if acc > self.best:
            self.best = acc
        self.model.train(True)
        self.neptune_metric('val/Test Accuracy', acc)
        return acc, 0.0, 0.0, 0.0, 0.0

    def open_validate(self, i_iter, training=True):
        self.model.train(False)
        knows = 0.0
        unknows = 0.0
        accs = GroupAverageMeter()
        #state = {'net':self.model.module.state_dict(), 'memory':self.memory,
        #         'cluster_mapping':self.cluster_mapping, 'global_label_set':self.global_label_set  }
        if not training:
            params = torch.load(self.config.init_weight)
            t_centers = params['memory']
            self.cluster_mapping = params['cluster_mapping']
            self.global_label_set = params['global_label_set']
        
        if training:
            t_centers = self.memory.memory
        length = len(self.test_loader.sampler)
        cls_pred_all = torch.zeros(length).cuda()
        memo_pred_all = torch.zeros(length).cuda()
        gt_all = torch.zeros(length).cuda()
        uk_index = self.config.num_classes

        cnt = 0
        for _, batch in tqdm(enumerate(self.test_loader)):
            acc_dict = {}
            img, label, name, _ = batch
            label = label.cuda()
            img = img.cuda()
            with torch.no_grad():
                _, neck, pred, pred2 = self.model(img)
            N = neck.shape[0]
            simi2cluster = self.cos_simi(F.normalize(neck, p=2, dim=-1), t_centers)
            clus_index = simi2cluster.argmax(dim=-1)
            cls_pred = pred2.argmax(-1)
            cls_pred_all[cnt:cnt+N] = cls_pred.squeeze()
            memo_pred_all[cnt:cnt+N] = clus_index.squeeze()
            gt_all[cnt:cnt+N] = label.squeeze()
            cnt+=N

        #clus_mapping = self.cluster_mapping # mapping between source label and target cluster index 

        uk_null = torch.ones_like(memo_pred_all).float().cuda() * uk_index
        map_mask =  torch.zeros_like(memo_pred_all).float().cuda() 

        for k,v in self.cluster_mapping.items():
            if v in self.global_label_set:
                map_mask += torch.where(memo_pred_all==k, torch.Tensor([1.0]).cuda().float(), map_mask.float()) 


        pred_label = torch.where(map_mask>0, cls_pred_all, uk_null)

        gt_all = torch.where(gt_all>=self.config.num_classes, torch.Tensor([uk_index]).cuda(), gt_all.float())
        mask = pred_label!=uk_index
        pred_binary = (pred_label==uk_index).squeeze().tolist()
        gt_binary = (gt_all==uk_index).squeeze().tolist()

        for i in gt_all.unique().tolist():
            mask = gt_all==i
            count = mask.sum().float()
            correct = (pred_label==gt_all) * mask
            correct = correct.sum().float()
            acc_dict[i] = ((correct/count).item(), count.item())
        accs.update(acc_dict)
        
        acc = np.mean(list(accs.avg.values()))
        self.print_acc(accs.avg)
        if uk_index not in accs.avg:
            self.model.train(True)
            self.neptune_metric('memo-val/Test Accuracy[center]', acc)    
            return acc, acc, 0.0, 0.0, 0.0
        bi_rec = metrics.recall_score(gt_binary, pred_binary, zero_division=0)
        bi_prec = metrics.precision_score(gt_binary, pred_binary, zero_division=0)
        #target私有类的召回率与准确率
        self.neptune_metric('val/bi recall[center]', bi_rec)
        self.neptune_metric('val/bi prec[center]', bi_prec)

        k_acc = (acc * len(accs.avg) - accs.avg[uk_index])/(len(accs.avg)-1)
        uk_acc = accs.avg[uk_index]
        common_sum = 0.0
        common_cnt = 0.0
        for k, v in accs.sum.items():
            if k != uk_index:
                common_sum += v
                common_cnt += accs.count[k]
        common_acc = common_sum / common_cnt
        h_score = 2 * (common_acc * uk_acc) / (common_acc + uk_acc)
        self.neptune_metric('memo-val/H-score', h_score)
        self.model.train(True)
        self.neptune_metric('memo-val/Test Accuracy[center]', acc)
        self.neptune_metric('memo-val/UK classification accuracy[center]', accs.avg[uk_index])
        self.neptune_metric('memo-val/Known category accuracy[center]', k_acc)
        if not training:
            with open(osp.join(self.config.snapshot, 'result.txt'), 'a') as f:
                f.write(self.config.source[:2] + '->' + self.config.target[:2] + '\n')
                f.write('HOS :'+ str(h_score) +'\n')                
                f.write('OS* :'+ str(k_acc)  +'\n')
                f.write('UNK :' + str(accs.avg[uk_index]) +'\n')
            f.close()
        return acc, k_acc, h_score, bi_rec, bi_prec

    def get_src_centers(self,return_from_memory=False):
          
        self.model.eval()
        num_cls = self.config.cls_share + self.config.cls_src
    
        if self.config.model!='res50':
            s_center = torch.zeros((num_cls, 256)).float().cuda()
        else:
            s_center = torch.zeros((num_cls, 256)).float().cuda()

        if self.config.model == 'swin-T':
            s_center = torch.zeros((num_cls, 256)).float().cuda()

        if not self.config.bottleneck:
            s_center = torch.zeros((num_cls, 2048)).float().cuda()
        #print('s_center',s_center.shape)
        counter = torch.zeros((num_cls,1)).float().cuda()
        s_feats = []
        s_labels = []
        for _, batch in tqdm(enumerate(self.src_loader)):
            acc_dict = {}
            img, label, _, _ = batch
            label = label.cuda().squeeze()
            with torch.no_grad():
                #print(img.shape)
                _, neck, _, _  = self.model(img.cuda())
            #neck为获得的倒数第二层的特征
            neck = F.normalize(neck, p=2, dim=-1)
            N, C = neck.shape
            s_labels.extend(label.tolist())
            s_feats.extend(torch.chunk(neck, N, dim=0))
        s_feats = torch.stack(s_feats).squeeze()
        s_labels = torch.from_numpy(np.array(s_labels)).cuda()
        if return_from_memory:
            memory_bank_wrapper = self.get_attr("source", "memory_bank_wrapper")
            s_feats = memory_bank_wrapper.as_tensor()
            #s_labels = self.get_attr("source", "train_labels")
          
        for i in s_labels.unique():
            i_msk = s_labels==i
            index = i_msk.squeeze().nonzero(as_tuple=False)
            i_feat = s_feats[index, :].mean(0)
            i_feat = F.normalize(i_feat, p=2, dim=1)
            s_center[i, :] = i_feat
            #print('i_feat shape',i_feat.shape)
        
        return s_center, s_feats, s_labels
    def sklearn_kmeans(self, feat, num_centers, init=None):
        if self.config.task in ['domainnet', 'visda']:
            return self.faiss_kmeans(feat, num_centers, init=init)
        if init is not None:
            kmeans = KMeans(n_clusters=num_centers, init=init, random_state=0).fit(feat.cpu().numpy())
        else:
            kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(feat.cpu().numpy())
        center, t_codes = kmeans.cluster_centers_, kmeans.labels_
        score = sklearn.metrics.silhouette_score(feat.cpu().numpy(), t_codes)
        return torch.from_numpy(center).cuda(), torch.from_numpy(t_codes).cuda(), score

    def faiss_kmeans(self, feat, K, init=None, niter=500):
        import faiss
        feat = feat.cpu().numpy()
        d = feat.shape[1]
        kmeans = faiss.Kmeans(d, K, niter=niter, verbose=False,  spherical=True)
        kmeans.train(feat)
        center = kmeans.centroids
        D, I = kmeans.index.search(feat, 1)
        center = torch.from_numpy(center).cuda()
        I= torch.from_numpy(I).cuda()
        D= torch.from_numpy(D).cuda()
        center = F.normalize(center, p=2 , dim=-1)
        return center, I.squeeze(), D
    def save_npy(self, name, tensor, iter=None):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        if name[-4:]!='.npy':
            name = name+ '.npy'
        if self.config.transfer_all:
            name = '{}_{}_'.format(self.config.source, self.config.target)+name
        if iter is not None:
            name = '{}_'+name
            np.save(osp.join(self.config.snapshot, name.format(iter)), tensor)
        else:
            np.save(osp.join(self.config.snapshot, name), tensor)
    def save_pickle(self, name, dic, iter=None):
        if name[-4:]!='.pkl':
            name = name+'.pkl'
        if self.config.transfer_all:
            name = '{}_{}_'.format(self.config.source, self.config.target)+name
        if iter is None:
            path = osp.join(self.config.snapshot, name)
        else:
            name = '{}_'+name
            path = osp.join(self.config.snapshot, name.format(iter))
        print(path)
        files = open(path, 'wb')
        pickle.dump(dic, files)
        files.close()

    def _choose_device(self):
        # check if use gpu
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA"
            )
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            cudnn.benchmark = True

            if self.config.gpu_device is None:
                self.config.gpu_device = list(range(torch.cuda.device_count()))
            elif not isinstance(self.config.gpu_device, list):
                self.config.gpu_device = [self.config.gpu_device]
            self.gpu_devices = self.config.gpu_device

            # set device when only one gpu
            num_gpus = len(self.gpu_devices)
            self.multigpu = num_gpus > 1 and torch.cuda.device_count() > 1
            if not self.multigpu:
                torch.cuda.set_device(self.gpu_devices[0])

            gpu_devices = ",".join([str(_gpu_id) for _gpu_id in self.gpu_devices])
            self.logger.info(f"User specified {num_gpus} GPUs: {gpu_devices}")
            self.parallel_helper_idxs = torch.arange(len(self.gpu_devices)).to(
                self.device
            )
            print('paaaaaaaaa',self.parallel_helper_idxs)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            torchutils.print_cuda_statistics(output=self.logger.info, nvidia_smi=False)
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError            
    
    def get_attr(self, domain, name):
        return getattr(self, f"{name}_{domain}")

    def set_attr(self, domain, name, value):
        setattr(self, f"{name}_{domain}", value)
        return self.get_attr(domain, name)    
    