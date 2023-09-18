import torchvision.transforms as transforms

class Config():
    
    ## Networks Info
    network = 'MobileNetV2'
    network_type = 'None' ## <= Can be ignored

    ## Training Info
    accm_batch_size = 512
    batch_size = 16
    gradient_accumulation_steps = int(accm_batch_size / batch_size)
    num_steps = 64000
    save_steps = 1000
    num_workers = 16
    continue_train = False
    lr = 1e-3
    lr_scheduler_list = ['cos', 'multi']
    lr_scheduler = lr_scheduler_list[1]

    ## Multi-LR Schedule Info => MultiStepLR
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
    weight_decay = 5e-4
    
    ## Dateset Info
    dataset_list = {0: 'CIFAR10', 1: 'CIFAR100', 2: 'STL10', 3: 'ImageNet'}
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
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4867, 0.4408),
        'STL10': (0.4467, 0.4398, 0.4067),
        'ImageNet': (0.485, 0.456, 0.406)
    }[dataset]

    std = {
        'CIFAR10': (0.2471, 0.2435, 0.2616), # 0.2023, 0.1994, 0.2010
        'CIFAR100': (0.2675, 0.2565, 0.2761),
        'STL10': (0.2603, 0.2566, 0.2713),
        'ImageNet': (0.229, 0.224, 0.225)
    }[dataset]
    
    
    ## Transform Info
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_shape),
        transforms.RandomCrop(input_shape),
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