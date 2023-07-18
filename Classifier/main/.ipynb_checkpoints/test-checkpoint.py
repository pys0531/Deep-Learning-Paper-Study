import torch
import torch.backends.cudnn as cudnn

import argparse
from config import cfg
from base import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cudnn.fastest = True
    cudnn.benchmark = True
    
    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    tester.model.eval()
    
    total = 0
    correct = 0
    for itr, (img, label) in enumerate(tester.test_batch_generator):
        img = img.to(tester.device)
        label = label.to(tester.device)
        
        output = tester.model(img)
        loss = tester.criterion(output, label)
        
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += torch.sum(predicted.eq(label))
        
    acc = correct / total * 100
    print(f"test_loss: {loss}, test_acc: {acc}")
    
if __name__ == "__main__":
    main()