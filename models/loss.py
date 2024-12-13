import torch
import torch.nn as nn

from abc import ABC
from torch.functional import F

### https://github.com/tfzhou/ContrastiveSeg/blob/main/lib/loss/loss_contrast.py#L15
class PixelContrastLoss(nn.Module, ABC):
    
    def __init__(self, temperature=0.07, base_temperature=0.07, max_samples=256, max_views=256):
        super(PixelContrastLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.max_samples = max_samples
        self.max_views = max_views
        self.ignore_label = 255


    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        
        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.float().clone().unsqueeze(1)
        
        labels = F.interpolate(
            labels,
            (feats.shape[2], feats.shape[3]),
            mode='nearest'
        )
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


class ContrastiveLoss(nn.Module):

    def __init__(self, seg_criterion: nn.Module, ctl_criterion: nn.Module, lmbd: float=1):
        super(ContrastiveLoss, self).__init__()
        self.seg_criterion = seg_criterion
        self.ctl_criterion = ctl_criterion
        self.lmbd = lmbd

    def forward(self, embed: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
        embed: torch.Tensor # (B, C_embed, H_embed, W_embed)
        pred: torch.Tensor # (B, N_classes, H_seg, W_seg) or (B, H_seg, W_seg) if N_classes == 1
        target: torch.Tensor = target.squeeze(1) # (B, H_target, W_target)

        pred = F.interpolate(
            pred,
            (target.size(1), target.size(2)),
            mode='bilinear',
            align_corners=True
        )
        if isinstance(self.seg_criterion, nn.BCEWithLogitsLoss):
            pred = pred.squeeze(1)
            target = target.float()
        
        seg_loss = self.seg_criterion(pred, target)

        if isinstance(self.seg_criterion, nn.BCEWithLogitsLoss):
            pred = pred.unsqueeze(1)
            target = target.long()

        _, predict = torch.max(pred, dim=1)
        ctl_loss = self.ctl_criterion(
            feats=embed,
            predict=predict,
            labels=target
        )
        
        loss = seg_loss + self.lmbd * ctl_loss

        return {
            'loss': loss,
            'seg_loss': seg_loss,
            'ctl_loss': ctl_loss
        }
