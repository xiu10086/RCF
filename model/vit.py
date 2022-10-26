from copy import deepcopy
from torchvision import models
import torch.nn as nn 
import torch.nn.functional as F
import torch 
from cd_trans_model.make_model import make_model
from .res50 import CosineClassifier
from pcs.utils import (AverageMeter, datautils, is_div, per, reverse_domain,
                       torchutils, utils)
    
class Vit(nn.Module):
    def __init__(self, num_classes, bottleneck=True, pretrained=True, extra=False,cfg=None,inc=256,temp=0.1):
        super(Vit, self).__init__()
        self.name = "vit"
        self.bottleneck = bottleneck
        self.features = make_model(cfg,inc)
        self.in_planes = 384 if 'small' in cfg.MODEL.Transformer_TYPE else 768        
        #self.fc = nn.Linear(self.in_planes, inc)   
        self.fc = nn.Sequential(nn.Linear(self.in_planes,1000), nn.BatchNorm1d(1000),nn.Linear(1000,inc))    
        if bottleneck:
            self.classifer = CosineClassifier(num_class=num_classes,inc=inc,temp=temp)
            torchutils.weights_init(self.classifer)
            
        else:
            ori_fc  = self.features.fc
            self.classifer = nn.Linear(768, num_classes)
        print('pretrained:',pretrained)

        self.num_classes = num_classes 
    def forward(self, x,  return_feat_for_pcs=False,images_for_cdd=False,x2=None):
        if len(x.shape)>4:
            x = x.squeeze()
        assert len(x.shape)==4
        if images_for_cdd:
            #x_len = int(x.shape[0]/2)
            #x1 = x[:x_len]
            #x2 = x[x_len:]
            feat1, feat2, feat_mix = self.features(x,x2,images_for_cdd=True)
            feat1 = self.fc(feat1)
            feat2 = self.fc(feat2)
            feat_mix = self.fc(feat_mix)
            return feat1,feat2,feat_mix
            
        else:
            feat = self.features(x,x)
            feat = feat.squeeze()
        #print(feat.shape)
        if return_feat_for_pcs:
            feat = self.fc(feat)            
            return feat
        if self.bottleneck:
            #_, bottleneck, prob, af_softmax = self.classifer(feat)
            bottleneck = self.fc(feat)
            prob = self.classifer(bottleneck)
        else:
            prob = self.classifer(feat)        
            bottleneck = feat
        return feat, bottleneck, prob, F.softmax(prob, dim=-1)


    def optim_parameters(self, lr, conv_ratio=0.1):
        d = [{'params': self.features.parameters(), 'lr': lr * conv_ratio},
                {'params': self.fc.parameters(), 'lr': lr },             
                {'params': self.classifer.parameters(), 'lr':  lr}]
        return d









