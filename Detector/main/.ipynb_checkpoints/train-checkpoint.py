import torch.backends.cudnn as cudnn

from config import cfg
from base import Trainer
from function import train

def main():
    cudnn.fastest = True
    cudnn.benchmark = True
    
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
        
    train(trainer)
            
            
if __name__ == "__main__":
    main()