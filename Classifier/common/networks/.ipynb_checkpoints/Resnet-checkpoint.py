import torch
import torch.nn as nn
from torchvision import models

from networks.modules.conv import ConvBlock, BasicBlock, Bottleneck
from config import cfg

## Accuracy with CosineAnnealingWarmupRestarts ##
# Model | Type   | Dataset | Acc
# Resnet cifar10 'cifar10': 91.33 (Adam / lr: 1-e3)
# Resnet cifar10 'cifar10': 92.64 (Sgd / lr: 0.1)
# Resnet 50      'stl10' : 70.36 (Adam / lr: 1-e3)
# Resnet 50      'stl10' : 70.13 (Sgd / lr: 0.1)

## Pretrained ##
# Resnet 50      'cifar10' : 89.49 (Adam / lr: 1-e3) epoch 25
# Resnet 50      'stl10'   : 88.43 (Adam / lr: 1-e3) epoch 25

##

class Resnet(nn.Module):
    def __init__(self, pretrain = True):
        super(Resnet, self).__init__()
        class_num = cfg.class_num
        
        resnet_type = {
            'cifar10': [BasicBlock, [5, 5, 5], [16, 32, 64], 16], # written as Resnet110 in the paper
            '18' : [BasicBlock, [2, 2,  2, 2], [64, 128, 256, 512], 64],
            '34' : [BasicBlock, [3, 4,  6, 3], [64, 128, 256, 512], 64],
            '50' : [Bottleneck, [3, 4,  6, 3], [64, 128, 256, 512], 64],
            '101': [Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512], 64],
            '152': [Bottleneck, [3, 8, 36, 3], [64, 128, 256, 512], 64]
        }
        
        block, layer, channel, inplanes = resnet_type[cfg.network_type]
        self.inplanes = inplanes
        
        
        ## Stem Layer ##
        stem = []
        if cfg.network_type == 'cifar10':
            stem.append(ConvBlock(3, self.inplanes, 3, 1, 1, bias = False))
        else:
            stem.append(ConvBlock(3, self.inplanes, 7, 2, 3, bias = False))
            stem.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.stem = nn.Sequential(*stem)
        
        
        ## Residual Block Layer ##
        layers = []
        for i in range(len(channel)):
            stride = 2
            if i == 0:
                stride = 1
            layers.append(self._make_layer(block, channel[i], layer[i], stride = stride))
        self.layers = nn.Sequential(*layers)
        
        
        ## FC Layer ##
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel[-1] * block.expansion, class_num)
        
        
        ## Init Weight ##        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        
    def _make_layer(self, block, planes, layer, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(self.inplanes, planes * block.expansion, 1, stride, bias = False, activation = None)
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample = downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, layer):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
            
    def forward(self, x):
        x = self.stem(x)
        
        x = self.layers(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    
    def init_weights(self):
        if cfg.network_type == 'cifar10':
            print(f"resnet {cfg.network_type} does not initialize to a pre-trained model.")
            return
        
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained
        pretrained_state_dict = eval(f"models.resnet{cfg.network_type}(pretrained=True)").state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-2]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

        print(f"Initialize resnet{cfg.network_type} from pretrained model")
    
if __name__ == "__main__":
    resnet = Resnet()
    x = torch.randn(4, 3, 224, 224)
    out = resnet(x)
    print(out.shape)