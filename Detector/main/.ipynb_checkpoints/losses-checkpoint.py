import torch
import torch.nn as nn

from utils.preprocessing import cxcy_to_xy, xy_to_cxcy, cxcy_to_gcxgcy, IoU
from config import cfg

class MultiBoxLoss(nn.Module):
    def __init__(self, priors, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors = priors
        self.priors_xy = cxcy_to_xy(priors)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        
    def forward(self, p_locs, p_clss, bboxes, labels):
        ## check prior num
        batch_size = p_locs.size(0)
        n_priors = self.priors.size(0)
        assert n_priors == p_locs.size(1) == p_clss.size(1)
        
        # 각 prior에 가장 가까운 정답 bbox의 위치 정보 저장 
        true_locs = torch.zeros((batch_size, n_priors, cfg.num_box_points), dtype=torch.float).to(cfg.device)  # true_locs: (N, 8732, 4) <= bboxes: (N, n_objects, 4)
        # 각 prior에 가장 가까운 정답 bbox의 라벨을 저장 
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(cfg.device)  # true_classes: (N, 8732) <= labels: (N, n_objects)
        
        # For each image
        for i in range(batch_size):
            n_objects = bboxes[i].size(0)

            # bboxes : (N, n_objects, 4) / self.priors_xy : (N, 8732, 4)
            iou_score = IoU(bboxes[i], self.priors_xy)  # (n_objects, 8732) <- IoU([n_objects, 4], [8732, 4])
            prior_best_iou_score, prior_best_bbox_idx = iou_score.max(0) # (8732) => each prior과 max로 겹치는 bbox의 iou_score와 bbox_idx
            bbox_best_iou_score, bbox_best_prior_idx = iou_score.max(1) # (n_objects) => each bbox와 max로 겹치는 prior의 iou_score와 prior_idx
            
            ## 1. prior best가 bbox의 obj_idx에 완벽히 매칭이 안될수 있어서 수정 
            prior_best_bbox_idx[bbox_best_prior_idx] = torch.LongTensor(range(n_objects)).to(cfg.device)
            
            ## 2. max overlap이 무조건 계산되도록 score가 threshold 0.5보다 크게 수정
            prior_best_iou_score[bbox_best_prior_idx] = 1.
            
            ## 1,2를 가지고 ground truth 생성           
            label_for_prior_best_bbox = labels[i][prior_best_bbox_idx] # (8732) => labels값에서 prior_best_bbox_idx 해당하는 index를 대입 => 8732개 각각의 prior이 갖는 object당 bbox label값 대입
            label_for_prior_best_bbox[prior_best_iou_score < self.threshold] = 0 # IoU값이 threshold보다 큰 것만 Loss 계산
            
            true_classes[i] = label_for_prior_best_bbox # => 각 batch size에서의 계산할 label을 넣어줌
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(bboxes[i][prior_best_bbox_idx]), self.priors)  # (8732, 4) => bboxes에서의 해당하는 index값을 넣어줌

            
        ## Loss Calculation
        positive_priors = true_classes != 0  # (N, 8732) => [[True, False, ..., False, False], ... , [True, False, ..., False, False]]
        
        # Localization Loss
        loc_loss = self.smooth_l1(p_locs[positive_priors], true_locs[positive_priors])
        
        # Confidence Loss
        n_positives = positive_priors.sum(dim=1) # Number of positive prior and hard-negative priors per image
        n_hard_negatives = self.neg_pos_ratio * n_positives
        
        conf_loss_all = self.cross_entropy(p_clss.view(-1, cfg.num_class), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)
        conf_loss_pos = conf_loss_all[positive_priors] # (sum(n_positives))
        
        conf_loss_neg = conf_loss_all.clone() # (N, 8732)
        conf_loss_neg[positive_priors] = 0. # (N, 8732)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True) # (N, 8732)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(cfg.device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732) => n_hard_negatives 개수만큼 가져오기
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))
        
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
        
        return conf_loss + self.alpha * loc_loss