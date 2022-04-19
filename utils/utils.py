import torch
import numpy as np
import random
import os
import argparse
import pprint


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def str2bool(v):
    if v.lower() in ['True', 'true', 'T', 't', 1]:
        return True
    elif v.lower() in ['False', 'false', 'F', 'f', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)
