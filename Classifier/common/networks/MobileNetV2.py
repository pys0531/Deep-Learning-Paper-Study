import torch.nn as nn
from networks.modules import ConvBlock
from torchvision import models

from networks.modules import InvertedResidual
from config import cfg

## Accuracy with CosineAnnealingWarmupRestarts ##

# cifar10: 84.23 (Adam / lr: 1-e3)
# cifar10: 87.30 (Sgd / lr: 0.1)
# stl10: 63.27 (Adam / lr: 1-e3)
# stl10: 71.08 (Sgd / lr: 0.1)

## Pretrained ##
# 'cifar10' : 87.46 (Adam / lr: 1-e3) epoch 25
# 'stl10'   : 89.25 (Adam / lr: 1-e3) epoch 25

##

class MobileNetV2(nn.Module):
    def __init__(self, ):
        super(MobileNetV2, self).__init__()
        params = [
                # t, c, n, s
                [1,  16, 1, 1],
                [6,  24, 2, 2],
                [6,  32, 3, 2],
                [6,  64, 4, 2],
                [6,  96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        # t = [ 1,  6,  6,  6,  6,   6,   6]
        # c = [16, 24, 32, 64, 96, 160, 320]
        # n = [ 1,  2,  3,  4,  3,   3,   1]
        # s = [ 1,  2,  2,  2,  1,   2,   1]
        
        inplanes = 32
        last_channel = 1280
        class_num = cfg.class_num
        
        ## Stem Layer ##
        self.stem = ConvBlock(3, inplanes, 3, stride = 2, activation = nn.ReLU6)
        
        ## InvertedResidual Layer ##
        blocks = []
        for t, c, n, s in params:
            planes = c
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(inplanes, planes, stride, t))
                inplanes = planes
        blocks.append(ConvBlock(inplanes, last_channel, 1, activation = nn.ReLU6))
        self.blocks = nn.Sequential(*blocks)
        
        ## FC Layer ##
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, class_num),
        )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avg_pool(x).flatten(1)
        x = self.fc(x)
        return x
    
    
    def init_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained
        pretrained_state_dict = models.mobilenet_v2(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-2]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

        print("Initialize mobilenet_v2 from pretrained model")