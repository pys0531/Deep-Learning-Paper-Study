import torch.nn as nn
from torchvision import models

from networks.modules.conv import ConvBlock
from config import cfg

## Accuracy with CosineAnnealingWarmupRestarts ##


## Pretrained ##
# VGG 19      'cifar10' : 92.61 (Adam / lr: 1-e3) epoch 25
# VGG 19      'stl10' : 89.93 (Adam / lr: 1-e3) epoch 25

##


class VGG(nn.Module):
    def __init__(self, ):
        super(VGG, self).__init__()
        
        vgg_type = {
            '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        layers_dim = vgg_type[cfg.network_type]

        inplanes = 3
        class_num = cfg.class_num
        
        ## VGG Layer ##
        layer = []
        for dim in layers_dim:
            if dim == 'M':
                layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer.append(ConvBlock(inplanes, dim, 3, bias = True))
                inplanes = dim
        self.layer = nn.Sequential(*layer)
        
        ## FC Layer ##        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(inplanes, class_num),
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
        x = self.layer(x)
        x = self.avg_pool(x).flatten(1)
        x = self.fc(x)
        return x
    
    
    def init_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained
        pretrained_state_dict = eval(f"models.vgg{cfg.network_type}_bn(pretrained=True)").state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-2]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

        print(f"Initialize vgg{cfg.network_type} from pretrained model")