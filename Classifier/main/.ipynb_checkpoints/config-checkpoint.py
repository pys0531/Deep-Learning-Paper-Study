import os
import os.path as osp
import sys

class Config():
    
    ## Networks Info
    network_list = {0: 'CNN', 1: 'Resnet', 2: 'MobileNetV2'}
    network = network_list[1]
    network_type = '50' #'18' #'50'   ## <= Can be ignored

    ## Training Info
    num_epochs = 25#120
    batch_size = 128#256
    num_workers = 16
    lr = 1e-3 #0.1*batch_size/256 #1e-3
    
    ## LR Schedule Info => CosineAnnealingWarmupRestarts
    cycle = 1#3
    min_lr = 0.00001
    max_lr = lr
    gamma = 0.5

    ## Optim Info
    optim_list = {0: 'sgd', 1: 'adam'}
    optim_type = optim_list[1]
    momentum = 0.9
    weight_decay = 5e-4
    
    ## Dateset Info
    dataset_list = {0: 'cifar10', 1: 'cifar100', 2: 'stl10', 3: 'imagenet'}
    dataset =  dataset_list[2]

    input_shape = {
        'cifar10': (32, 32),
        'cifar100': (32, 32),
        'stl10': (96, 96),
        'imagenet': (224, 224)
    }[dataset]
    
    class_num = {
        'cifar10': 10,
        'cifar100': 100,
        'stl10': 10,
        'imagenet': 1000
    }[dataset]
    
    mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'stl10': (0.4914, 0.4822, 0.4465),
        'imagenet': (0.485, 0.456, 0.406)
    }[dataset]

    std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'stl10': (0.2471, 0.2435, 0.2616),
        'imagenet': (0.229, 0.224, 0.225)
    }[dataset]
    
    # classes = {
    #     'cifar10': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # }[dataset]
    
    
    ## Directory Info
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, "..")
    model_dir = osp.join(root_dir, "model_dump")
    

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import make_folder
make_folder(cfg.model_dir)