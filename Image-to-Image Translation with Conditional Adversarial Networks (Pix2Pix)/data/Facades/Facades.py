import os
import os.path as osp

from torch.utils.data import Dataset

import copy
import glob
import numpy as np
from PIL import Image


class Facades(Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        self.train_dir = osp.join("data", "Facades", "train")
        self.val_dir = osp.join("data", "Facades", "val")
        self.test_dir = osp.join("data", "Facades", "test")

        self.datalist = self.load_data()


    def load_data(self, ):
        if self.data_split == "train":
            img_path = sorted(glob.glob(os.path.join(self.train_dir) + "/*.jpg"))
            ## train 이미지가 적기 때문에 test 이미지를 추가사용
            img_path.extend(sorted(glob.glob(os.path.join(self.test_dir) + "/*.jpg")))

        elif self.data_split == "val":
            img_path = sorted(glob.glob(os.path.join(self.val_dir) + "/*.jpg"))

        return img_path


    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        img = Image.open(data)

        w, h = img.size
        gt_img = img.crop((0, 0, w / 2, h))
        label_img = img.crop((w / 2, 0, w, h))

        ## gt_img와 label_img가 동시에 flip되어야 하기 때문에 transforms.RandomHorizontalFlip를 사용하지 않고 따로 처리
        if np.random.random() < 0.5: 
            gt_img = Image.fromarray(np.array(gt_img)[:, ::-1, :], "RGB")
            label_img = Image.fromarray(np.array(label_img)[:, ::-1, :], "RGB")

        gt_img = self.transform(gt_img)
        label_img = self.transform(label_img)

        return {"gt": gt_img, "label": label_img}


    def __len__(self, ):
        return len(self.datalist)
