import torchvision.transforms as transforms

class Config():
    
    ## Networks Info
    network = 'ViT'
    type_ = {0: 'base', 1: 'large', 2: 'huge'}
    network_type = type_[0]

    ## Training Info
    accm_batch_size = 512
    batch_size = 16
    gradient_accumulation_steps = int(accm_batch_size / batch_size)
    num_steps = 1000
    save_steps = 100
    num_workers = 16
    continue_train = True
    lr = 3e-2
    lr_scheduler_list = ['cos', 'multi']
    lr_scheduler = lr_scheduler_list[0]

    ## Multi-LR Schedule Info
    if lr_scheduler == 'multi':
        milestones = [32000, 48000]
        gamma = 0.1
    
    ## Cos-LR Schedule Info => CosineAnnealingWarmupRestarts
    if lr_scheduler == 'cos':
        cycle = 1
        min_lr = 0
        max_lr = lr
        gamma = 0.5
        warmup_steps = 100
    
    ## Optim Info
    optim_list = {0: 'sgd', 1: 'adam'}
    optim_type = optim_list[0]
    momentum = 0.9
    weight_decay = 0
    
    ## Dateset Info
    dataset_list = {0: 'CIFAR10', 1: 'CIFAR10', 2: 'STL10', 3: 'ImageNet'}
    dataset =  dataset_list[0]
    data_split_ratio = {"train": 0.98, "val": 0.02, "test": 1}

    input_shape = (224, 224)
    resize_shape = (256, 256)
    
    class_num = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'STL10': 10,
        'ImageNet': 1000
    }[dataset]
    
    mean = {
        'CIFAR10': (0.5, 0.5, 0.5),
        'CIFAR100': (0.5, 0.5, 0.5),
        'STL10': (0.5, 0.5, 0.5),
        'ImageNet': (0.5, 0.5, 0.5)
    }[dataset]

    std = {
        'CIFAR10': (0.5, 0.5, 0.5),
        'CIFAR100': (0.5, 0.5, 0.5),
        'STL10': (0.5, 0.5, 0.5),
        'ImageNet': (0.5, 0.5, 0.5)
    }[dataset]
    
    
    ## Transform Info
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(input_shape, scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])