from copy import deepcopy
from distutils.command.config import config
from stringprep import in_table_a1
from tkinter.tix import Tree
from tokenize import Number

from numpy import source
#from pyrsistent import T
from sklearn import cluster
from .base_trainer import *
from model import *
from dataset import *
import sklearn
from utils.joint_memory import Memory
from model.basenet import ResClassifier_MME
from utils.ovaloss import ova_loss,open_entropy
import math

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sklearn import metrics
from tqdm import tqdm

from . import base_trainer

def ExpWeight(step, gamma=3, max_iter=5000, reverse=False):
    step = max_iter-step
    ans = 1.0 * (np.exp(- gamma * step * 1.0 / max_iter))
    return float(ans)

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10,
                     power=0.75, init_lr=0.001,weight_decay=0.0005,
                     max_iter=10000):
    #10000
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    #max_iter = 10000
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i] * 10
        i+=1
    return lr

def distill_loss(teacher_output, student_out):
    teacher_out = F.softmax(teacher_output, dim=-1)    
    loss = torch.sum( -teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
    #print(teacher_out)
    #print(-teacher_out * F.log_softmax(student_out, dim=-1))
    return loss.mean()


class Trainer(BaseTrainer):
    def __init__(self, config,  writer):
        self.config = config

        super(Trainer, self).__init__(config,writer)



    def optimize(self):
        for i_iter in tqdm(range(self.config.stop_steps)):
            self.model = self.model.train()
            self.losses = edict({})
            losses = self.iter(i_iter)
            #if i_iter % self.config.print_freq ==0:
                #self.print_loss(i_iter)
            if not self.warmup or  i_iter+1>=self.config.warmup_steps:
                if (i_iter+1) % self.config.stage_size ==0:
                    self.class_set = self.re_clustering(i_iter)
                if (i_iter+1) % self.config.val_freq ==0:
                    self.validate(i_iter, self.class_set)

            
    def _create_model(self):

        self.model = init_model(self.config)
        out_dim = self.config.model_params.out_dim

        # classification head
        if self.config.multi_gpu:    
            self.cls_head = self.model.module.classifer
        else:
            self.cls_head = self.model.classifer
                
    def _create_optimizer(self):
        lr = self.config.optim_params.learning_rate
        momentum = self.config.optim_params.momentum
        weight_decay = self.config.optim_params.weight_decay
        conv_lr_ratio = self.config.optim_params.conv_lr_ratio

        parameters = []
        # batch_norm layer: no weight_decay
        params_bn, _ = torchutils.split_params_by_name(self.model, "bn")
        parameters.append({"params": params_bn, "weight_decay": 0.0})
        # conv layer: small lr
        _, params_conv = torchutils.split_params_by_name(self.model, ["fc", "bn"])
        if conv_lr_ratio:
            parameters[0]["lr"] = lr * conv_lr_ratio
            parameters.append({"params": params_conv, "lr": lr * conv_lr_ratio})
        else:
            parameters.append({"params": params_conv})
        # fc layer
        params_fc, _ = torchutils.split_params_by_name(self.model, "fc")
        #if self.cls and self.config.optim_params.cls_update:
            #params_fc.extend(list(self.cls_head.parameters()))
        parameters.append({"params": params_fc})

        if self.config.model == "vit":
            if self.config.multi_gpu:                
                parameters = self.model.module.optim_parameters(lr,conv_ratio=conv_lr_ratio)
            else:
                parameters = self.model.optim_parameters(lr,conv_ratio=conv_lr_ratio)                

        self.optim = torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=self.config.optim_params.nesterov,
        )

        # lr schedular
        if self.config.optim_params.lr_decay_schedule:
            optim_stepLR = torch.optim.lr_scheduler.MultiStepLR(
                self.optim,
                milestones=self.config.optim_params.lr_decay_schedule,
                gamma=self.config.optim_params.lr_decay_rate,
            )
            self.lr_scheduler_list.append(optim_stepLR)

        if self.config.optim_params.decay:
            self.optim_iterdecayLR = torchutils.lr_scheduler_invLR(self.optim)                
            


   
# compute train features

    @torch.no_grad()
    def compute_train_features(self):
        if self.is_features_computed:
            return
        else:
            self.is_features_computed = True
        self.model.eval()

        for domain in ("source", "target"):
            train_loader = self.get_attr(domain, "train_init_loader")
            features, y, idx = [], [], []
            tqdm_batch = tqdm(
                total=len(train_loader), desc=f"[Compute train features of {domain}]"
            )
            for batch_i, (indices, images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                feat = self.model(images,return_feat_for_pcs=True)
                feat = F.normalize(feat, dim=1)

                features.append(feat)
                y.append(labels)
                idx.append(indices)

                tqdm_batch.update()
            tqdm_batch.close()

            features = torch.cat(features)
            y = torch.cat(y)
            idx = torch.cat(idx).to(self.device)

            self.set_attr(domain, "train_features", features)
            self.set_attr(domain, "train_labels", y)
            self.set_attr(domain, "train_indices", idx)

    def clear_train_features(self):
        self.is_features_computed = False

    # Memory bank

    @torch.no_grad()
    def _init_memory_bank(self):
        out_dim = self.config.model_params.out_dim
        for domain_name in ("source", "target"):
            data_len = self.get_attr(domain_name, "train_len")
            memory_bank = MemoryBank(data_len, out_dim)
            if self.config.model_params.load_memory_bank:
                self.compute_train_features()
                idx = self.get_attr(domain_name, "train_indices")
                feat = self.get_attr(domain_name, "train_features")
                memory_bank.update(idx, feat)
                # self.logger.info(
                #     f"Initialize memorybank-{domain_name} with pretrained output features"
                # )
                # save space
                if self.config.data_params.name in ["visda17", "domainnet"]:
                    delattr(self, f"train_indices_{domain_name}")
                    delattr(self, f"train_features_{domain_name}")

            self.set_attr(domain_name, "memory_bank_wrapper", memory_bank)

            self.loss_fn.module.set_attr(domain_name, "data_len", data_len)
            self.loss_fn.module.set_broadcast(
                domain_name, "memory_bank", memory_bank.as_tensor()
            )

    @torch.no_grad()
    def _update_memory_bank(self, domain_name, indices, new_data_memory):
        memory_bank_wrapper = self.get_attr(domain_name, "memory_bank_wrapper")
        memory_bank_wrapper.update(indices, new_data_memory)
        updated_bank = memory_bank_wrapper.as_tensor()
        self.loss_fn.module.set_broadcast(domain_name, "memory_bank", updated_bank)

    def _load_memory_bank(self, memory_bank_dict):
        """load memory bank from checkpoint

        Args:
            memory_bank_dict (dict): memory_bank dict of source and target domain
        """
        for domain_name in ("source", "target"):
            memory_bank = memory_bank_dict[domain_name]._bank.cuda()
            self.get_attr(domain_name, "memory_bank_wrapper")._bank = memory_bank
            self.loss_fn.module.set_broadcast(domain_name, "memory_bank", memory_bank)

    # Cluster

    @torch.no_grad()
    def _update_cluster_labels(self):
        k_list = (self.config.k_list).copy()
        for clus_type in self.config.loss_params.clus.type:
            cluster_labels_domain = {}
            cluster_centroids_domain = {}
            cluster_phi_domain = {}

            # clustering for each domain
            if clus_type == "each":
                for domain_name in ("target", "source"):

                    memory_bank_tensor = self.get_attr(
                        domain_name, "memory_bank_wrapper"
                    ).as_tensor()
                    if domain_name == "source":
                        k_list = [self.num_class for i in range(len(k_list))]
                    # clustering
                    cluster_labels, cluster_centroids, cluster_phi = torch_kmeans(
                        k_list,
                        memory_bank_tensor,
                        seed=self.current_epoch + self.current_iteration,
                    )

                    cluster_labels_domain[domain_name] = cluster_labels
                    cluster_centroids_domain[domain_name] = cluster_centroids
                    cluster_phi_domain[domain_name] = cluster_phi

                self.cluster_each_centroids_domain = cluster_centroids_domain
                self.cluster_each_labels_domain = cluster_labels_domain
                self.cluster_each_phi_domain = cluster_phi_domain
            else:
                print(clus_type)
                raise NotImplementedError

            # update cluster to losss_fn
            for domain_name in ("source", "target"):
                self.loss_fn.module.set_broadcast(
                    domain_name,
                    f"cluster_labels_{clus_type}",
                    cluster_labels_domain[domain_name],
                )
                self.loss_fn.module.set_broadcast(
                    domain_name,
                    f"cluster_centroids_{clus_type}",
                    cluster_centroids_domain[domain_name],
                )
                if cluster_phi_domain:
                    self.loss_fn.module.set_broadcast(
                        domain_name,
                        f"cluster_phi_{clus_type}",
                        cluster_phi_domain[domain_name],
                    )
