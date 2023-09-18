import torch
import torch.nn as nn
from networks.modules import ConvBlock
from torchvision import models

from networks.modules import InvertedResidual
from config import cfg


"""
## Accuracy with MultiStepLR ##
## Pretrained ##

###########################################################################################################
# Dataset : PASCALVOC 2012
# train_mode : train
# val_mode : val
# VGG 16      'PASCALVOC' : 0.705 mAP (SGD / lr: 1-e3) epoch 231 => milestones: [154, 193] gamma: 0.1
 'aeroplane': 0.7765766978263855,
 'bicycle': 0.7790604829788208,
 'bird': 0.6802600622177124,
 'boat': 0.6209621429443359,
 'bottle': 0.37180715799331665,
 'bus': 0.8133156895637512,
 'car': 0.8169997334480286,
 'cat': 0.8449958562850952,
 'chair': 0.4917326867580414,
 'cow': 0.739271342754364,
 'diningtable': 0.7017614245414734,
 'dog': 0.7976252436637878,
 'horse': 0.8037703633308411,
 'motorbike': 0.7848232388496399,
 'person': 0.7504602074623108,
 'pottedplant': 0.3863990306854248,
 'sheep': 0.6979055404663086,
 'sofa': 0.7503731846809387,
 'train': 0.792302668094635,
 'tvmonitor': 0.6956378221511841

Mean Average Precision (mAP): 0.705

###########################################################################################################
# Dataset : PASCALVOC 2012
# train_mode : trainval
# val_mode : test
# VGG 16      'PASCALVOC' : 0.753 mAP (SGD / lr: 1-e3) epoch 231 => milestones: [154, 193] gamma: 0.1
{'aeroplane': 0.7982099652290344,
 'bicycle': 0.8258264064788818,
 'bird': 0.7239344716072083,
 'boat': 0.6640470623970032,
 'bottle': 0.4588026702404022,
 'bus': 0.8568034768104553,
 'car': 0.8526769280433655,
 'cat': 0.8828631639480591,
 'chair': 0.5765849351882935,
 'cow': 0.8088170289993286,
 'diningtable': 0.7420951128005981,
 'dog': 0.8339937925338745,
 'horse': 0.8422805070877075,
 'motorbike': 0.8273301720619202,
 'person': 0.7812727093696594,
 'pottedplant': 0.4968195855617523,
 'sheep': 0.7632260322570801,
 'sofa': 0.7612094283103943,
 'train': 0.83784419298172,
 'tvmonitor': 0.7251632809638977}

Mean Average Precision (mAP): 0.753

###########################################################################################################
# Dataset : PASCALVOC 2012+2007
# train_mode : trainval
# val_mode : test
# VGG 16      'PASCALVOC' : 0.753 mAP (SGD / lr: 1-e3) epoch 231 => milestones: [154, 193] gamma: 0.1
{'aeroplane': 0.815758466720581,
 'bicycle': 0.8544331789016724,
 'bird': 0.7703700065612793,
 'boat': 0.6882923245429993,
 'bottle': 0.5274109840393066,
 'bus': 0.8657081127166748,
 'car': 0.8734980821609497,
 'cat': 0.8736726641654968,
 'chair': 0.6088107228279114,
 'cow': 0.8158715963363647,
 'diningtable': 0.771665096282959,
 'dog': 0.8532626032829285,
 'horse': 0.8772947192192078,
 'motorbike': 0.8622145652770996,
 'person': 0.80378657579422,
 'pottedplant': 0.5344825387001038,
 'sheep': 0.7935630679130554,
 'sofa': 0.8052281737327576,
 'train': 0.8507954478263855,
 'tvmonitor': 0.7709472179412842}

Mean Average Precision (mAP): 0.781
"""


class VGG(nn.Module):
    def __init__(self, ):
        super(VGG, self).__init__()
        
        vgg_type = {
            '11': {"conv4_3": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512], "conv4_pool": ['M'], "conv5": [512, 512, 'LM']},
            '13': {"conv4_3": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512], "conv4_pool": ['M'], "conv5": [512, 512, 'LM']},
            '16': {"conv4_3": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512], "conv4_pool": ['M'], "conv5": [512, 512, 512, 'LM']},
            '19': {"conv4_3": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512], "conv4_pool": ['M'], "conv5": [512, 512, 512, 512, 'LM']},
        }
        layers_dim = vgg_type[cfg.backbone_type]

        self.predict_block = nn.Conv2d
        inplanes = 3
        
        ## VGG Layer ##
        for name in layers_dim:
            exec(f"{name} = []")
            for dim in layers_dim[name]:
                if dim == 'M':
                    eval(f"{name}.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))")
                elif dim == 'LM':
                    eval(f"{name}.append(nn.MaxPool2d(kernel_size=3, stride=1, padding = 1, ceil_mode=True))")
                else:
                    eval(f"{name}.append(ConvBlock(inplanes, dim, 3, bias = True))")
                    inplanes = dim
                 
        for name in layers_dim:
            exec(f"self.{name} = nn.Sequential(*{name})")
        
        ## Remove FC Layer
        ## Additional Layer
        #### FC -> Conv
        self.conv6 = ConvBlock(512, 1024, 3)#, padding = 6, dilation = 6, bias = True)
        self.conv7 = ConvBlock(1024, 1024, 1)
        
        #### Extra Layer
        self.conv8_1 = ConvBlock(1024, 256, 1)
        self.conv8_2 = ConvBlock(256, 512, 3, stride = 2)
        
        self.conv9_1 = ConvBlock(512, 128, 1)
        self.conv9_2 = ConvBlock(128, 256, 3, stride = 2)
        
        self.conv10_1 = ConvBlock(256, 128, 1)
        self.conv10_2 = ConvBlock(128, 256, 3, padding = 0)
        
        self.conv11_1 = ConvBlock(256, 128, 1)
        self.conv11_2 = ConvBlock(128, 256, 3, padding = 0) # , norm_layer = None
        
        
        ## Conv4_3 L2 Normalize
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)
        
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        
    def forward(self, x):
        x = self.conv4_3(x)
        conv4_3_feat = x
        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feat.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feat = conv4_3_feat / norm  # (N, 512, 38, 38)
        conv4_3_feat = conv4_3_feat * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        x = self.conv4_pool(x)
        x = self.conv5(x)
        
        ## Additional Layer in SSD use instead and
        ## Remove FC Layer
        x = self.conv6(x)
        x = self.conv7(x)
        conv7_feat = x
        
        ## Exra Feature Layer
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        conv8_2_feat = x
        
        x = self.conv9_1(x)
        x = self.conv9_2(x)
        conv9_2_feat = x
        
        x = self.conv10_1(x)
        x = self.conv10_2(x)
        conv10_2_feat = x
        
        x = self.conv11_1(x)
        x = self.conv11_2(x)
        conv11_2_feat = x
        
        return [conv4_3_feat, conv7_feat, conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat]
    
    
    def init_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())[1:] # remove rescale_factors
        
        # Pretrained
        pretrained_state_dict = eval(f"models.vgg{cfg.backbone_type}_bn(pretrained=True)").state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names):
            if "conv6" in param:
                break
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

        print(f"Initialize vgg{cfg.backbone_type} from pretrained model")