from torchvision import models
import torch.nn as nn 
import torch.nn.functional as F
import torch 
from .swin_transformer import *

def load_pretrained(model):
    #logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load('/home/hy/zx/DA code/Domain-Consensus-Clustering/model/swin_tiny_patch4_window7_224.pth', map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            #logger.warning(f"Error in loading {k}, passing......")
            print('Error in loading {k}, passing...... ')
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            #logger.warning(f"Error in loading {k}, passing......")
            print('Error in loading {k}, passing......')
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            #logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            #logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    model.load_state_dict(state_dict, strict=False)
    #logger.warning(msg)

    #logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


class CLS(nn.Module):
    """
    From: https://github.com/thuml/Universal-Domain-Adaptation
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.bn = nn.BatchNorm1d(bottle_neck_dim)
        self.gn = nn.GroupNorm(32,bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
#        for module in self.main.children():
 #           x = module(x)
  #          out.append(x)
        #batchnorm
        #x = self.bn(self.bottleneck(x))  
        #groupnorm
        x = self.gn(self.bottleneck(x))
        out.append(x)
        x =self.fc(x)
        out.append(x)
        out.append(F.softmax(x, dim=-1))
        return out


class Swin_T(nn.Module):
    def __init__(self, num_classes, bottleneck=True, pretrained=True, extra=False):
        super(Swin_T, self).__init__()

        self.bottleneck = bottleneck
        self.features = SwinTransformer(num_classes=num_classes)
        if pretrained:
            load_pretrained(self.features)
        if bottleneck:
            self.classifer = CLS(768, num_classes)
        else:
            ori_fc  = self.features.fc
            self.classifer = nn.Linear(768, num_classes)
        print('pretrained:',pretrained)

        self.num_classes = num_classes 
    def forward(self, x):
        if len(x.shape)>4:
            x = x.squeeze()
        assert len(x.shape)==4
        out,feat = self.features(x)
        feat = feat.squeeze()
        if self.bottleneck:
            _, bottleneck, prob, af_softmax = self.classifer(feat)
        else:
            prob = self.classifer(feat)        
            bottleneck = feat
        return feat, bottleneck, prob, F.softmax(prob, dim=-1)


    def optim_parameters(self, lr):
        d = [{'params': self.features.parameters(), 'lr': lr},
                {'params': self.classifer.parameters(), 'lr':  lr*10}]
        return d









