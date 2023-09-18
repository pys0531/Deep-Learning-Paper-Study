import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config():
    network = "SSD"
    backbone_list = {0: 'VGG', 1: 'MobileNetV2'}
    backbone_network = backbone_list[1]
    exec(f'from configs.backbone_configs.{backbone_network} import Backbon_Config') 
    backbone_type = Backbon_Config.backbone_type
    
    ## Detection Info
    num_box_points = 4
    num_anchors = Backbon_Config.num_anchors
    s_min = 0.2
    s_max = 0.9
    ## Origin Scale factor in SSD Paper
    # scale_factor = [s_min + (s_max-s_min)/(len(num_anchors)-1)*(i-1) for i in range(1, len(num_anchors)+1)] # [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]
    ## Repaired Scale factor
    scale_factors = [0.1]
    for i in range(1,len(num_anchors)):
        scale_factors += [s_min + (s_max-s_min)/(len(num_anchors)-2)*(i-1)] # [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
    aspect_ratios = [1, 2, 1/2, 3, 1/3] # [[1., 2., 0.5], [1., 2., 3., 0.5, .333], [1., 2., 3., 0.5, .333], [1., 2., 3., 0.5, .333], [1., 2., 0.5], [1., 2., 0.5]]
    feature_resolutions = Backbon_Config.feature_resolutions
    feature_dimensions = Backbon_Config.feature_dimensions
    
    ## Training Info
    num_epochs = 232
    batch_size = 8#32
    num_workers = 4
    continue_train = False
    lr = 1e-3 #0.1*batch_size/256 #1e-3
    lr_scheduler_list = ['cos', 'multi']
    lr_scheduler = lr_scheduler_list[1]

    ## Multi-LR Schedule Info
    if lr_scheduler == 'multi':
        milestones = [154, 193]
        gamma = 0.1
    
    ## Cos-LR Schedule Info => CosineAnnealingWarmupRestarts
    if lr_scheduler == 'cos':
        cycle = 1#3
        min_lr = 0
        max_lr = lr
        gamma = 0.5

    ## Optim Info
    optim_list = {0: 'sgd', 1: 'adam'}
    optim_type = optim_list[0]
    momentum = 0.9
    weight_decay = 0.0005
    
    ## Augmentation Info
    resize_shape = (300, 300)

    ## Dateset Info
    dataset_list = {0: 'PASCALVOC'}
    dataset =  dataset_list[0]

    input_shape = {
        'PASCALVOC': (300, 300)
    }[dataset]
    
    num_class = {
        'PASCALVOC': 21
    }[dataset]
    
    mean = {
        'PASCALVOC': [0.485, 0.456, 0.406] # (0.5, 0.5, 0.5)
    }[dataset]

    std = {
        'PASCALVOC': [0.229, 0.224, 0.225] # (0.5, 0.5, 0.5)
    }[dataset]
    
    classes = {'PASCALVOC': 
               {
                   'background' :   0,
                   'aeroplane' :    1,
                   'bicycle' :      2,
                   'bird' :         3,
                   'boat' :         4,
                   'bottle' :       5,
                   'bus' :          6,
                   'car' :          7,
                   'cat' :          8,
                   'chair' :        9,
                   'cow' :          10,
                   'diningtable' :  11,
                   'dog' :          12,
                   'horse' :        13,
                   'motorbike' :    14,
                   'person' :       15,
                   'pottedplant' :  16,
                   'sheep' :        17,
                   'sofa' :         18,
                   'train' :        19,
                   'tvmonitor' :    20
               }
              }[dataset]
    
    
    ## Augmentation Info
    transform_train = A.Compose([
        A.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-18 / 255., 18 / 255.), p = 0.5),
        A.HorizontalFlip(p = 0.5),
        A.Resize(input_shape[1], input_shape[0]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields = [])) # format = albumentations / pascal_voc
    
    transform_test = A.Compose([
        A.Resize(input_shape[1], input_shape[0]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields = []))
    
    
    transform_infer = A.Compose([
        A.Resize(input_shape[1], input_shape[0]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])