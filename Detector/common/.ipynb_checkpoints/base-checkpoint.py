import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from config import cfg
from model import get_network
from utils.torch_utils import CosineAnnealingWarmupRestarts
from losses import MultiBoxLoss

import abc
import os.path as osp
import math
import glob

exec(f"from {cfg.dataset} import {cfg.dataset}")

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
        
    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Load weight snapshot_' + str(cur_epoch) + '.pth.tar')

        return start_epoch, model, optimizer
        

class Trainer(Base):
    def __init__(self,):
        super(Trainer, self).__init__()

    def get_optimizer(self, model):
        if cfg.optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        elif cfg.optim_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay = cfg.weight_decay)
        return optimizer
    
    def set_lr(self, epoch):
        if cfg.lr_scheduler == 'cos':
            self.scheduler.step()
        elif cfg.lr_scheduler == 'multi':
            self.scheduler.step(epoch)

    def get_lr(self,):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):        
        trainset = eval(cfg.dataset)("trainval", cfg.transform_train)
        validset = eval(cfg.dataset)("test", cfg.transform_test)
        print(f"Complete Load {cfg.dataset} Dataset")
        self.train_batch_generator = DataLoader(trainset, batch_size = cfg.batch_size, shuffle = True, collate_fn = trainset.collate_fn, num_workers = cfg.num_workers, pin_memory=True)
        self.val_batch_generator = DataLoader(validset, batch_size = cfg.batch_size, shuffle = False, collate_fn = trainset.collate_fn, num_workers = cfg.num_workers, pin_memory=True)
        
    def _make_lrschedule(self):
        if cfg.lr_scheduler == 'cos':
            first_cycle_steps = len(self.train_batch_generator) * cfg.num_epochs // cfg.cycle
            scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer, 
                first_cycle_steps=first_cycle_steps, 
                cycle_mult=1.0,
                max_lr=cfg.max_lr, 
                min_lr=cfg.min_lr, 
                warmup_steps=int(first_cycle_steps * 0.2), 
                gamma=cfg.gamma,
                last_epoch = self.start_epoch * len(self.train_batch_generator) if cfg.continue_train else -1
            )
        elif cfg.lr_scheduler == 'multi':
            scheduler = MultiStepLR(self.optimizer, cfg.milestones, gamma=cfg.gamma)
        return scheduler
        
    def _make_model(self):
        model = get_network(pretrained = True)
        model.to(cfg.device)
        optimizer = self.get_optimizer(model)
        
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
            
        #model.train()
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.criterion = MultiBoxLoss(model.priors).to(cfg.device) # nn.CrossEntropyLoss()
        self.scheduler = self._make_lrschedule()


        
        
class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        
    def _make_batch_generator(self):
        testset = eval(cfg.dataset)("test", cfg.transform_test)
        print(f"Complete Load {cfg.dataset} Dataset")
        
        self.test_batch_generator = DataLoader(testset, batch_size = cfg.batch_size, shuffle = False, collate_fn = testset.collate_fn, num_workers = cfg.num_workers, pin_memory=True)
        
    def _make_model(self):
        model_path = osp.join(cfg.model_dir, f"snapshot_{self.test_epoch}.pth.tar")
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        
        model = get_network(pretrained = False)
        model.to(cfg.device)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        self.model = model
        self.criterion = criterion