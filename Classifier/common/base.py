import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import cfg
from model import get_network
from utils.torch_utils import CosineAnnealingWarmupRestarts

import abc
import os.path as osp
import math


class Base():
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, ):
        self.cur_epoch = 0
    
    @abc.abstractmethod
    def _make_batch_generator(self):
        return
    
    @abc.abstractmethod
    def _make_model(self):
        return
    
    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pth.tar')
        torch.save(state, file_path)
        

class Trainer(Base):
    def __init__(self,):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_optimizer(self, model):
        if cfg.optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg.min_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        elif cfg.optim_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg.min_lr, weight_decay = cfg.weight_decay)
        return optimizer
    
    def set_lr(self,):
        self.scheduler.step()

    def get_lr(self,):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(cfg.input_shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ])
        
        if cfg.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=f'{cfg.root_dir}/data/{cfg.dataset}', train=True, download=True, transform=transform_train)
            validset = torchvision.datasets.CIFAR10(root=f'{cfg.root_dir}/data/{cfg.dataset}', train=False, download=True, transform=transform_val)
        elif cfg.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root=f'{cfg.root_dir}/data/{cfg.dataset}', train=True, download=True, transform=transform_train)
            validset = torchvision.datasets.CIFAR100(root=f'{cfg.root_dir}/data/{cfg.dataset}', train=False, download=True, transform=transform_val)
        elif cfg.dataset == 'stl10':
            trainset = torchvision.datasets.STL10(root=f'{cfg.root_dir}/data/{cfg.dataset}', split='train', download=True, transform=transform_train)
            validset = torchvision.datasets.STL10(root=f'{cfg.root_dir}/data/{cfg.dataset}', split='test', download=True, transform=transform_val)
        elif cfg.dataset == 'imagenet':
            trainset = torchvision.datasets.ImageNet(root=f'{cfg.root_dir}/data/{cfg.dataset}', split='train', download=True, transform=transform_train)
        print(f"Load {cfg.dataset} Dataset")
        self.train_batch_generator = DataLoader(trainset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_workers)
        self.val_batch_generator = DataLoader(validset, batch_size = cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
        
    def _make_lrschedule(self):
        first_cycle_steps = len(self.train_batch_generator) * cfg.num_epochs // cfg.cycle
        scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer, 
                first_cycle_steps=first_cycle_steps, 
                cycle_mult=1.0,
                max_lr=cfg.max_lr, 
                min_lr=cfg.min_lr, 
                warmup_steps=int(first_cycle_steps * 0.2), 
                gamma=cfg.gamma
            )
        return scheduler
        
    def _make_model(self):
        model = get_network(pretrained = True)
        model.to(self.device)
        self.start_epoch = 0
        self.model = model
        self.optimizer = self.get_optimizer(model)
        #model.train()
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = self._make_lrschedule()


        
        
class Tester(Base):
    def __init__(self, test_epoch):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_epoch = int(test_epoch)
        
    def _make_batch_generator(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ])
        
        if cfg.dataset == 'cifar10':
            testset = torchvision.datasets.CIFAR10(root=f'{cfg.root_dir}/data', train=False, download=True, transform=transform_test)
        elif cfg.dataset == 'cifar100':
            testset = torchvision.datasets.CIFAR100(root=f'{cfg.root_dir}/data', train=False, download=True, transform=transform_test)
        elif cfg.dataset == 'stl10':
            testset = torchvision.datasets.STL10(root=f'{cfg.root_dir}/data/{cfg.dataset}', split='test', download=True, transform=transform_test)
        
        self.test_batch_generator = DataLoader(testset, batch_size = cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
        
        
    def _make_model(self):
        model_path = osp.join(cfg.model_dir, f"snapshot_{self.test_epoch}.pth.tar")
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        
        model = get_network(pretrained = False)
        model.to(self.device)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        #model.eval()
        criterion = nn.CrossEntropyLoss()
        
        self.model = model
        self.criterion = criterion