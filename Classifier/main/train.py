import torch
import torch.backends.cudnn as cudnn

from config import cfg
from base import Trainer
from function import train

import random
import numpy as np

def main():
    ## seed 고정 => 고정해도 값이 바뀌면 lr scheduling/dropout 때문에 그럴수도 있음
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ## 학습모델 설정
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
        
    # 학습시작
    train(trainer)
        
        
            
if __name__ == "__main__":
    main()
    
    
    

