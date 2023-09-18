import torch
from torch.utils.data import Dataset

import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
import copy
import cv2

from config import cfg
from utils.preprocessing import load_img, augmentation

class PASCALVOC(Dataset):
    def __init__(self, mode, transform):
        self.data_dir = osp.join(cfg.root_dir, "data", cfg.dataset) # cifar-10-batches-py
        self.mode = mode
        self.transform = transform
        self.db = self.load_data()
        print(f"Load Dataset PASCALVOC {mode}set: {len(self.db)}")
        
    def __len__(self,):
        return len(self.db)
        
    def load_data(self,):
        if self.mode == "trainval" or self.mode == "train" or self.mode == "val":
            folders = ["VOC2012", "VOC2007"]
        elif self.mode == "test":
            folders = ["VOC2007"]
        else:
            raise Exception(f"mode variable value '{self.mode}' is not defined")
            
        db_list = []
        for folder in folders:
            train_path = osp.join(self.data_dir, folder)
            img_name_path = osp.join(train_path, "ImageSets", "Main", f"{self.mode}.txt")
            with open(img_name_path, 'r') as f:
                img_names = f.readlines()
                img_names = list(map(lambda s: s.strip('\n'), img_names))

            for img_name in img_names:
                db = defaultdict(list)
                anno_path = osp.join(train_path, "Annotations", f"{img_name}.xml")
                with open(anno_path) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()

                    ## get img name
                    full_image_name = osp.join(train_path, "JPEGImages", root.find('filename').text)

                    ## get img size
                    size = root.find("size")
                    width = int(size.find("width").text)
                    height = int(size.find("height").text)
                    channel = int(size.find("depth").text)


                    db['img_path'] = full_image_name
                    db['img_shape'] = np.array([height, width, channel])

                    ## get objects in img
                    objects = root.findall("object")
                    for _object in objects:
                        name = _object.find("name").text
                        bbox = _object.find("bndbox")
                        xmin = int(bbox.find("xmin").text)-1
                        ymin = int(bbox.find("ymin").text)-1
                        xmax = int(bbox.find("xmax").text)-1
                        ymax = int(bbox.find("ymax").text)-1
                        difficult = int(_object.find('difficult').text)

                        db['classes'].append(cfg.classes[name])
                        db['bboxes'].append([xmin, ymin, xmax, ymax])
                        db['difficults'].append(difficult)
                    db['classes'] = np.array(db['classes'])
                    db['bboxes'] = np.array(db['bboxes'])
                    db['difficults'] = np.array(db['difficults'])
                    db_list.append(db)
                    
        return db_list     
                    
    
    def __getitem__(self, idx):
        db = copy.deepcopy(self.db[idx])
        
        img = load_img(db['img_path'])
        
        labels = db['classes']
        bboxes = db['bboxes']
        difficults = db['difficults']
        
        # self.show_test(img, bboxes)
        img, bboxes, labels, difficults = augmentation(img, bboxes, labels, difficults, self.transform, self.mode)
        # self.show_test(img.detach().numpy().transpose(1,2,0), bboxes)
        
        bboxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)
        difficults = torch.LongTensor(difficults)
        
        return img, labels, bboxes, difficults
    
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        
        images, boxes, classes, difficulties = zip(*batch)
        images = torch.stack(images, axis=0)
        return images, boxes, classes, difficulties

    def show_test(self, ori_img, bboxes, name = "img"):
        img = ori_img.copy()
        cv2.imshow(f"{name}", img)
        for n, bbox in enumerate(bboxes):
            img = cv2.rectangle(img, (int(bbox[0]),int(bbox[1])) , (int(bbox[2]),int(bbox[3])), (0, 255, 0), 3)
            cv2.imshow(f"{name}_{n}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        