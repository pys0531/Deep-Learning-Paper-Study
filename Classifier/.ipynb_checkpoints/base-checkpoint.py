import torch
import torch.optim as optim
import timm

from config import cfg

class train():
    def __init__(self):
        pass
    
    def _make_model(self):
        self.model = 
        
    def _make_optimize(self, model):
        optimizer = optim.adam(model.parameters(), lr = cfg.lr)
        return optimizer
    
    def set_lr(self, optimizer):
        for m in optimzer:
            m["lr"] = 
            
    
    

    
