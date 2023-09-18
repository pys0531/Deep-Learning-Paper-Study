import torch
import torch.nn as nn
from networks.modules.conv import ConvBlock
from config import cfg

## Accuracy ##

# cifar10: 86.34600067138672

##

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        class_num = cfg.class_num

        self.conv1 = ConvBlock(3, 32, 3, 2, padding = 1)
        self.conv2 = ConvBlock(32, 64, 3, 2, padding = 1)
        self.conv3 = ConvBlock(64, 128, 3, 2, padding = 1)

        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x