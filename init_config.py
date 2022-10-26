from easydict import EasyDict as edict
from yaml import load, dump
import yaml
from utils.flatwhite import *
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random
import platform
from utils.adjust_config import adjust_config,process_config
import json
def easy_dic(dic):
    dic = edict(dic)
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = edict(value)
    return dic
def show_config(config, sub=False):
    msg = ''
    for key, value in config.items():
        if isinstance(value, dict):
            msg += show_config(value, sub=True)
        else :
            msg += '{:>25} : {:<15}\n'.format(key, value)
    return msg

def type_align(source, target):
    if isinstance(source, int):
        return int(target)
    elif isinstance(source, float):
        return float(target)
    elif isinstance(source, str):
        return target
    elif isinstance(source, bool):
        return bool(source)
    else:
        print("Unsupported type: {}".format(type(source)))

def config_parser(config, args):
    print(args)
    for arg in args:
        if '=' not in arg:
            continue
        else:
            key, value = arg.split('=')
        print(key)
        print(config[key],type(config[key]))
        print(value,type(value))
        value = type_align(config[key], value) 
        config[key] = value
    return config

def load_json(f_path):
    with open(f_path, "r") as f:
        return json.load(f)

def init_config(config_path, argvs):
   # with open(config_path, 'r') as f:
   #     config = yaml.load(f, Loader=yaml.FullLoader)
   # f.close()
    
    #config = easy_dic(config)
    #config = config_parser(config, argvs)
    config_json = load_json(config_path)  
    config = process_config(config_json)
    config = config_parser(config, argvs)    
    config = adjust_config(config)    
    config.snapshot = osp.join(config.snapshot, config.note)
    mkdir(config.snapshot)
    print('Snapshot stored in: {}'.format(config.snapshot))
    if config.tensorboard:
        config.tb = osp.join(config.log, config.note)
        mkdir(config.tb)
        writer = SummaryWriter(config.tb)
    else:
        writer = None
    if config.fix_seed:
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
    #message = show_config(config)
    config.data_params["source"] = config.source
    config.data_params["target"] = config.target 
    config.optim_params["learning_rate"] = config.lr
    print(type(config.loss_weight))   
    print(config.loss_weight)
    config.loss_params["weight"] = eval(config.loss_weight)
    print(config)
    return config, writer

