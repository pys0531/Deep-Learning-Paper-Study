import torch.nn as nn
from networks.modules import ConvBlock
from torchvision import models

from networks.modules import InvertedResidual, SeparableConv
from config import cfg

"""
## Accuracy with MultiStepLR ##
## Pretrained ##

###########################################################################################################
# Dataset : PASCALVOC 2012+2007
# train_mode : trainval
# val_mode : test
# VGG 16      'PASCALVOC' : 0.663 mAP (SGD / lr: 1-e3) epoch 231 => milestones: [154, 193] gamma: 0.1
{'aeroplane': 0.6940515637397766,
 'bicycle': 0.7686474919319153,
 'bird': 0.6269093751907349,
 'boat': 0.5383041501045227,
 'bottle': 0.24640177190303802,
 'bus': 0.7829529643058777,
 'car': 0.7755828499794006,
 'cat': 0.8300252556800842,
 'chair': 0.43113476037979126,
 'cow': 0.6896128058433533,
 'diningtable': 0.6740036606788635,
 'dog': 0.7712397575378418,
 'horse': 0.8389371633529663,
 'motorbike': 0.7912953495979309,
 'person': 0.6684796214103699,
 'pottedplant': 0.39493805170059204,
 'sheep': 0.6155164837837219,
 'sofa': 0.7010417580604553,
 'train': 0.7967758178710938,
 'tvmonitor': 0.6314273476600647}

Mean Average Precision (mAP): 0.663
###########################################################################################################
"""

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
        num_class = cfg.num_class
        
        self.predict_block = SeparableConv
        
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
        self.blocks = nn.Sequential(*blocks[:-5]) # C 16 -> 96
        self.mid_block = nn.Sequential(*blocks[-5:-1]) # C 96 - 320
        self.last_block = blocks[-1] # 320 -> 1280
        
        
        ## Remove FC Layer
        #### Extra Layer
        self.conv8 = InvertedResidual(1280, 512, stride = 2, expand_ratio=0.2)
        
        self.conv9 = InvertedResidual(512, 256, stride = 2, expand_ratio=0.25)
        
        self.conv10 = InvertedResidual(256, 256, stride = 2, expand_ratio=0.5)
        
        self.conv11 = InvertedResidual(256, 64, stride = 2, expand_ratio=0.25)

        
        ## FC Layer ##
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(last_channel, num_class),
#         )
        
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
        x = self.blocks(x) # [1, 96, 19, 19]
        conv5 = x
        x = self.mid_block(x) # [1, 320, 10, 10]
        x = self.last_block(x) # [1, 1280, 10, 10]
        conv7 = x
        
        ## Extra Layer
        x = self.conv8(x)
        conv8 = x
        x = self.conv9(x)
        conv9 = x
        x = self.conv10(x)
        conv10 = x
        x = self.conv11(x)
        conv11 = x
        return [conv5, conv7, conv8, conv9, conv10, conv11]
    
    
    def init_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained
        pretrained_state_dict = models.mobilenet_v2(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-2]):
            if "conv8" in param:
                break
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

        print("Initialize mobilenet_v2 from pretrained model")