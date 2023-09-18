import torch
import torchvision.transforms as transforms

from networks.ViT import ViT 

import os.path as osp
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


## start script is vis.py in main folder
def attention_score(cfg):
    assert cfg.network == "ViT", "The network must be ViT"

    ## Prepare Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT().to(device)
    model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
    cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
    ckpt = torch.load(osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')) 
    model.load_state_dict(ckpt['network'])
    print('Load weight snapshot_' + str(cur_epoch) + '.pth.tar')
    model.eval()


    ## Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    im = Image.open("Cat.jpg")
    x = transform(im).to(device)
    x.size()


    ## Inference
    logits, att_mat = model(x.unsqueeze(0), attn_vis = True) # [12,1,12,197,197]
    print(logits)
    print(torch.argmax(logits))
    att_mat = torch.stack(att_mat).squeeze(1) # [12,12,197,197]

    ## Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1) # [12,197,197]

    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)


    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    for i, v in enumerate(joint_attentions):
        mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        result = (mask * im).astype("uint8")

        plt.imshow(result)
        plt.savefig(f'attn_score{i}.png')