from torchvision import models
import torch.nn as nn 
import torch.nn.functional as F
import torch 
from torch.autograd import Function


class CLS(nn.Module):
    """
    From: https://github.com/thuml/Universal-Domain-Adaptation
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.bn = nn.BatchNorm1d(bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
#        for module in self.main.children():
 #           x = module(x)
  #          out.append(x)
        x = self.bn(self.bottleneck(x))
        out.append(x)
        x =self.fc(x)
        out.append(x)
        out.append(F.softmax(x, dim=-1))
        return out
    
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class CosineClassifier(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        self.normalize_fc()

        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x)
        x_out = x_out / self.temp

        return x_out

    def normalize_fc(self):
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, eps=1e-12, dim=1)

    @torch.no_grad()
    def compute_discrepancy(self):
        self.normalize_fc()
        W = self.fc.weight.data
        D = torch.mm(W, W.transpose(0, 1))
        D_mask = 1 - torch.eye(self.num_class).cuda()
        return torch.sum(D * D_mask).item()

class Res50(nn.Module):
    def __init__(self, num_classes, bottleneck=True, pretrained=True, extra=False,inc=256,temp=0.1):
        super(Res50, self).__init__()
        self.name = "resnet"
        self.bottleneck = bottleneck
        features = models.resnet50(pretrained=pretrained)
        self.features =  nn.Sequential(*list(features.children())[:-1])
        self.fc = nn.Linear(2048, inc)
        #torchutils.weights_init(self.fc)

        if bottleneck:
            #self.classifer = CLS(2048, num_classes)
            self.classifer = CosineClassifier(num_class=num_classes,inc=inc,temp=temp)
            #torchutils.weights_init(self.classifer)

        else:
            ori_fc  = features.fc
            self.classifer = nn.Linear(2048, num_classes)
        print('pretrained:',pretrained)

        self.num_classes = num_classes 
        
    def forward(self, x, return_feat_for_pcs=False, tsne=False):
        if len(x.shape)>4:
            x = x.squeeze()
        assert len(x.shape)==4
        feat = self.features(x)
        feat = feat.squeeze()
        #print(feat.shape)
        if tsne:
            return feat, feat, feat, feat
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


    def optim_parameters(self, lr):
        d = [{'params': self.features.parameters(), 'lr': lr},
             {'params': self.fc.parameters(), 'lr':  lr*10},
                {'params': self.classifer.parameters(), 'lr':  lr*10}]
        return d
