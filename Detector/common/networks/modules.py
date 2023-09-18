import torch
import torch.nn as nn
from typing import Callable, List, Optional, Sequence, Tuple, Union

class ConvBlock(nn.Module):
    def __init__(
        self, 
        input: int,
        output: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1, 
        padding: Optional[Union[int, Tuple[int, ...], str]] = None, 
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        activation: Optional[Callable[..., torch.nn.Module]] = nn.ReLU, 
        norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.BatchNorm2d, 
        inplace: Optional[bool] = True
    ):
        super(ConvBlock, self).__init__()
            
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
            
        if bias is None:
            bias = norm_layer is None
            
        modules = []
        modules.append(nn.Conv2d(input, output, kernel_size, stride, padding = padding, groups = groups, bias = bias))
        if norm_layer is not None: 
            modules.append(norm_layer(output))
        if activation is not None: 
            params = {} if inplace is None else {"inplace": inplace}
            modules.append(activation(**params))
            
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)

    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = ConvBlock(inplanes, planes, 3, stride, padding = 1, bias = False)
        self.conv2 = ConvBlock(planes, planes, 3, padding = 1, bias = False, activation = None)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out
        
        
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = ConvBlock(inplanes, planes, 1, bias = False)
        self.conv2 = ConvBlock(planes, planes, 3, stride, padding = 1, bias = False)
        self.conv3 = ConvBlock(planes, planes * self.expansion, 1, bias = False, activation = None)
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

    
class InvertedResidual(nn.Module):
    def __init__(self, inplanes, planes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        
        self.use_res_connect = (stride == 1) and (inplanes == planes)
        expand_dim = int(round(inplanes * expand_ratio))
        
        layer = []
        if expand_ratio != 1:
            # expand
            layer.append(ConvBlock(inplanes, expand_dim, kernel_size = 1, activation = nn.ReLU6))
        # depthwise
        layer.append(ConvBlock(expand_dim, expand_dim, kernel_size = 3, stride = stride, groups = expand_dim, activation = nn.ReLU6))
        # linear
        layer.append(ConvBlock(expand_dim, planes, kernel_size = 1, activation = None))
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        out = self.layer(x)
        if self.use_res_connect:
            out += x
        return out
        
        
class SeparableConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, padding=0):
        super(SeparableConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(inplanes, inplanes, kernel_size = kernel_size, stride = stride, padding = padding, groups= inplanes, activation = nn.ReLU6),
            nn.Conv2d(inplanes, planes, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)