import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.modules import ConvBlock
from config import cfg
from utils.preprocessing import cxcy_to_xy, gcxgcy_to_cxcy, IoU

from math import sqrt

exec(f"from networks.backbone.{cfg.backbone_network} import {cfg.backbone_network}")


class PredictBlock(nn.Module):
    def __init__(self, predict_block):
        super(PredictBlock, self).__init__()
        
        num_cls_loc = cfg.num_box_points + cfg.num_class
        self.num_cls_loc = num_cls_loc
        self.conv4_3_cl = predict_block(cfg.feature_dimensions[0], num_cls_loc*cfg.num_anchors[0], 3, padding = 1)
        self.conv7_cl = predict_block(cfg.feature_dimensions[1], num_cls_loc*cfg.num_anchors[1], 3, padding = 1)
        self.conv8_2_cl = predict_block(cfg.feature_dimensions[2], num_cls_loc*cfg.num_anchors[2], 3, padding = 1)
        self.conv9_2_cl = predict_block(cfg.feature_dimensions[3], num_cls_loc*cfg.num_anchors[3], 3, padding = 1)
        self.conv10_2_cl = predict_block(cfg.feature_dimensions[4], num_cls_loc*cfg.num_anchors[4], 3, padding = 1)
        self.conv11_2_cl = predict_block(cfg.feature_dimensions[5], num_cls_loc*cfg.num_anchors[5], 3, padding = 1)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        
    def forward(self, backbone_aux_feats):
        conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2 = backbone_aux_feats
        batch_size = conv4_3.size(0)
        
        conv4_3 = self.conv4_3_cl(conv4_3) # [N, 14*4, 38, 38] => [N, 38 * 38 * 4, 4] , [N, 38 * 38 * 4, 10]
        conv7 = self.conv7_cl(conv7)
        conv8_2 = self.conv8_2_cl(conv8_2)
        conv9_2 = self.conv9_2_cl(conv9_2)
        conv10_2 = self.conv10_2_cl(conv10_2)
        conv11_2 = self.conv11_2_cl(conv11_2)
        
        conv4_3_l, conv4_3_c = conv4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_cls_loc).split([cfg.num_box_points, cfg.num_class], 2)
        conv7_l, conv7_c = conv7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_cls_loc).split([cfg.num_box_points, cfg.num_class], 2)
        conv8_2_l, conv8_2_c = conv8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_cls_loc).split([cfg.num_box_points, cfg.num_class], 2)
        conv9_2_l, conv9_2_c = conv9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_cls_loc).split([cfg.num_box_points, cfg.num_class], 2)
        conv10_2_l, conv10_2_c = conv10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_cls_loc).split([cfg.num_box_points, cfg.num_class], 2)
        conv11_2_l, conv11_2_c = conv11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_cls_loc).split([cfg.num_box_points, cfg.num_class], 2)
        
        locs = torch.cat([conv4_3_l, conv7_l, conv8_2_l, conv9_2_l, conv10_2_l, conv11_2_l], dim=1)
        clss = torch.cat([conv4_3_c, conv7_c, conv8_2_c, conv9_2_c, conv10_2_c, conv11_2_c], dim=1)

        return locs, clss

    
class SSD(nn.Module):
    def __init__(self):
        super(SSD, self).__init__()
        
        self.backbone = eval(cfg.backbone_network)()
        self.predict_block = PredictBlock(self.backbone.predict_block)
        self.priors = self.create_prior_box()
        
    def forward(self, x):
        # VGG: [1, 512, 38, 38], [1, 1024, 19, 19], [1, 512, 10, 10], [1, 256, 5, 5], [1, 256, 3, 3], [1, 256, 1, 1])
        # MobileNetV2: [1, 96, 19, 19], [1, 1280, 10, 10], [1, 512, 5, 5], [1, 256, 3, 3], [1, 256, 2, 2], [1, 64, 1, 1]
        backbone_aux_feats = self.backbone(x)
        
        # [1, 8732, 4], [1, 8732, 20]
        locs, clss = self.predict_block(backbone_aux_feats)
        
        return locs, clss
    
    def create_prior_box(self, ):
        prior_boxes = []

        for n, res in enumerate(cfg.feature_resolutions):
            for i in range(res):
                for j in range(res):
                    cx = (j + 0.5) / res
                    cy = (i + 0.5) / res

                    for ratio in cfg.aspect_ratios[:cfg.num_anchors[n] - 1]: ## Duplicate ratio 1 => [1, 1, 2, 1/2, 3, 1/3]
                        prior_boxes.append([cx, cy, cfg.scale_factors[n] * sqrt(ratio), cfg.scale_factors[n] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(cfg.scale_factors[n] * cfg.scale_factors[n+1])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(cfg.device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes
    
    
    def init_weights(self):
        self.backbone.init_weights()
        
        
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, cfg.num_class):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = IoU(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(cfg.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(cfg.device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(cfg.device))
                image_labels.append(torch.LongTensor([0]).to(cfg.device))
                image_scores.append(torch.FloatTensor([0.]).to(cfg.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

        

        