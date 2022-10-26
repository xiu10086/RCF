from collections import OrderedDict
from operator import mod
from .res50 import *
from.vgg19 import * 
from model.basenet import ResClassifier_MME
import torch.optim as optim

import torch 

def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = False
def release_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = True

def init_model(cfg):
    num_classes =cfg.num_classes
    if cfg.extra:
        num_classes +=1
    if cfg.model=='res50':
        print('选用res50模型')
        model = Res50(num_classes, bottleneck=cfg.bottleneck,pretrained=cfg.pretrained, extra=cfg.extra).cuda()
    elif cfg.model =='vgg19':
        model =VGG19(num_classes, bottleneck=cfg.bottleneck, extra=cfg.extra,pretrained=cfg.pretrainded).cuda()
    elif cfg.model == 'swin-T' :
        model = Swin_T(num_classes, bottleneck=cfg.bottleneck,pretrained=cfg.pretrained, extra=cfg.extra).cuda()
    elif cfg.model == 'vit':
        model = Vit(num_classes, bottleneck=cfg.bottleneck,pretrained=cfg.pretrained, extra=cfg.extra,cfg=cfg.vit_set).cuda()
        print('选用vit模型')
    if cfg.fix_bn:
        freeze_bn(model)
    else:
        release_bn(model)


    if cfg.init_weight != 'None':
        #print('?????????',cfg.init_weight)
        params = torch.load(cfg.init_weight)
        print('Model restored with weights from : {}'.format(cfg.init_weight))
        try:
            model.load_state_dict(params['net'], strict=True)
            print('成功')
        except Exception as e:
            temp = OrderedDict()
            for k,v in params['net'].items():
                name = k[7:]
                temp[name] = v
            model.load_state_dict(temp)

    if cfg.multi_gpu:
        model = nn.DataParallel(model)
    if cfg.train:
        model = model.train().cuda()
        print('Mode --> Train')
    else:
        model = model.eval().cuda()
        print('Mode --> Eval')
    return model

def init_C2(cfg):
    C2 = ResClassifier_MME(num_classes=2 * cfg.num_classes,
                           norm=False, input_size=256)
    device = torch.device("cuda")
    C2.to(device)
    opt_c = optim.SGD(list(C2.parameters()), lr=1.0,
                       momentum=cfg.momentum, weight_decay=0.0005,
                       nesterov=True)
    if cfg.multi_gpu:
      C2  = nn.DataParallel(C2)
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    return C2,opt_c,param_lr_c, cfg.lr,cfg.stop_steps

def init_Rot(cfg):
    rot_head = torch.nn.Linear(256, 4)
    device = torch.device("cuda")
    rot_head = rot_head.to(device)
    if cfg.multi_gpu:
          rot_head  = nn.DataParallel(rot_head)
    opt_c = optim.SGD(list(rot_head.parameters()), lr=1.0,
                       momentum=cfg.momentum, weight_decay=0.0005,
                       nesterov=True)
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    return rot_head,opt_c,param_lr_c