import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
        self, 
        input: int,
        output: int,
        kernel_size: int,
        stride: int = 1, 
        padding: int = 0, 
        bias: bool = True, 
        activation = None, 
        is_activation: bool = True, 
        norm_layer = None, 
        is_norm_layer: bool = True
    ):
        super(ConvBlock, self).__init__()
            
        modules = []
        modules.append(nn.Conv2d(input, output, kernel_size, stride, padding = padding, bias = bias))
        if is_norm_layer: 
            modules.append(nn.BatchNorm2d(output)) if norm_layer is None else modules.append(norm_layer(output))
        if is_activation: 
            modules.append(nn.ReLU(inplace=True)) if activation is None else modules.append(activation(inplace=True))
            
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)

    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = ConvBlock(inplanes, planes, 3, stride, padding = 1, bias = False)
        self.conv2 = ConvBlock(planes, planes, 3, padding = 1, bias = False, is_activation = False)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity 
        out = self.relu(out)
        return out
        
        
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = ConvBlock(inplanes, planes, 1, bias = False)
        self.conv2 = ConvBlock(planes, planes, 3, stride, padding = 1, bias = False)
        self.conv3 = ConvBlock(planes, planes * self.expansion, 1, bias = False, is_activation = False)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        
    def forward(self, x):        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out
