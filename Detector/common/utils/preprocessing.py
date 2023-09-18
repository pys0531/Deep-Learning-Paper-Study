import torch
import cv2
import numpy as np
import random

from config import cfg

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def xy_to_cxcy(xy):
    if isinstance(xy, np.ndarray):
        return np.concatenate([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)
    elif isinstance(xy, torch.Tensor):
        return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)
    else:
        raise f"variable must be np.ndarray or torch.Tensor"

def cxcy_to_xy(cxcy):
    if isinstance(cxcy, np.ndarray):
        return np.concatenate([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)
    elif isinstance(cxcy, torch.Tensor):
        return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)
    else:
        raise f"variable must be np.ndarray or torch.Tensor"

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def IoU(box1, box2, mode = "corner"):
    if mode == "center":
        box1 = cxcy_to_xy(box1)
        box2 = cxcy_to_xy(box2)
    elif mode == "corner":
        pass
    else:
        raise f"{mode} variable of IoU is not defined. select between center or corner"
    
    if isinstance(box1, np.ndarray) and isinstance(box2, np.ndarray):
        ## Intersection area
        inter_top = np.maximum(box1[:, :2][:, None], box2[:, :2][None, :])
        inter_bottom = np.minimum(box1[:, 2:][:, None], box2[:, 2:][None, :])
        inter_wh = (inter_bottom - inter_top).clip(0)
        inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]

        ## Union area
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = box1_area[:, None] + box2_area[None, :] - inter
    elif isinstance(box1, torch.Tensor) and isinstance(box2, torch.Tensor):
        ## Intersection area
        inter_top = torch.max(box1[:, :2].unsqueeze(1), box2[:, :2].unsqueeze(0))
        inter_bottom = torch.min(box1[:, 2:].unsqueeze(1), box2[:, 2:].unsqueeze(0))
        inter_wh = (inter_bottom - inter_top).clamp(0)
        inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]

        ## Union area
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter
    else:
        raise f"box1 and box2 must be np.ndarray or torch.Tensor"
    
    return inter / union


def augmentation(img, bboxes, labels, difficults, transform, mode):
    if mode == 'val' or mode == 'test':
        ## bboxes normalize
        h, w, c = img.shape
        bboxes = bboxes / [w, h, w, h]
        
        ## test transform
        transformed = transform(image = img, bboxes = bboxes)
        img = transformed['image']
        bboxes = transformed['bboxes']
        
        return img, bboxes, labels, difficults
            
    ## random_expand
    img, bboxes = random_expand(img, bboxes)
    img, bboxes, labels, difficults = random_crop(img, bboxes, labels, difficults)
    
    ## bboxes normalize
    h, w, c = img.shape
    bboxes = bboxes / [w, h, w, h]
    
    ## random photometric_distortions / flip / resize / totensor
    transformed  = transform(image = img, bboxes = bboxes)
    img = transformed['image']
    bboxes = transformed['bboxes']
    
    return img, bboxes, labels, difficults
    
    
def random_expand(img, bboxes, p = 0.5):
    if random.random() >= p:
        return img, bboxes

    h, w, c = img.shape
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * h)
    new_w = int(scale * w)

    # Create such an image with the filler
    new_img = np.ones((new_h, new_w, 3), dtype=np.float32) * np.expand_dims(cfg.mean, (0,1))
    new_img = new_img.astype(np.uint8)
    #new_img = (np.random.rand(new_h, new_w, 3) * 255).astype(np.uint8)

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - w)
    right = left + w
    top = random.randint(0, new_h - h)
    bottom = top + h
    new_img[top:bottom, left:right, :] = img

    # Adjust bounding boxes' coordinates accordingly
    bboxes = bboxes + [left, top, left, top]# (n_objects, 4), n_objects is the no. of objects in this image

    return new_img, bboxes
    
    
    
def random_crop(img, bboxes, labels, difficults):
    h, w, c = img.shape
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return img, bboxes, labels, difficults

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * h)
            new_w = int(scale_w * w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, w - new_w)
            right = left + new_w
            top = random.randint(0, h - new_h)
            bottom = top + new_h
            crop = np.array([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = IoU(crop[None,:], bboxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_img = img[top:bottom, left:right, :]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_bboxes = bboxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficults = difficults[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_bboxes[:, :2] = np.maximum(new_bboxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_bboxes[:, :2] -= crop[:2]
            new_bboxes[:, 2:] = np.minimum(new_bboxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_bboxes[:, 2:] -= crop[:2]

            return new_img, new_bboxes, new_labels, new_difficults
    

    
def random_flip(img, bboxes, p = 0.5):
    if random.random() <= p:
        return img, bboxes
    
    ## random flip
    img = img[:, ::-1, :]

    # Flip boxes
    bboxes[:, 0] = img.shape[1] - bboxes[:, 0] - 1
    bboxes[:, 2] = img.shape[1] - bboxes[:, 2] - 1
    bboxes = bboxes[:, [2, 1, 0, 3]]
    return img, bboxes
    
    
def resize(img, bboxes):
    h, w, c = img.shape
    img = cv2.resize(img, cfg.input_shape)
    
    # Resize bounding boxes
    bboxes = bboxes / [h, w, h, w]# * [cfg.input_shape[0], cfg.input_shape[1], cfg.input_shape[0], cfg.input_shape[1]]  # percent coordinates

    return img, bboxes