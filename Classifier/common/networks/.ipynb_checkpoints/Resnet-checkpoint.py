import torch
import torch.nn as nn
from networks.modules import ConvBlock, BasicBlock, Bottleneck
from config import cfg

class Resnet(nn.Module):
    def __init__(self):
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
            downsample = ConvBlock(self.inplanes, planes * block.expansion, 1, stride, bias = False, is_activation = False)
        
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
    
if __name__ == "__main__":
    resnet = Resnet()
    x = torch.randn(4, 3, 224, 224)
    out = resnet(x)
    print(out.shape)