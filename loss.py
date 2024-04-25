import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


def normalize(x, axis=-1):
    """
    Normalizing to unit length along the specified dimension.
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.
    Reference: Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
    - margin (float): margin for triplet.
    - inputs: feature matrix with shape (batch_size, feat_dim).
    - targets: ground truth labels with shape (num_classes).
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()  # torch.eq: greater than or equal to >=

        return loss, correct


# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class MarginMMD_Loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, P=4, K=4, margin=None):
        super(MarginMMD_Loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.P = P
        self.K = K
        self.margin = margin
        if self.margin:
            print(f'Using Margin : {self.margin}')
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) + 1e-9 for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        if torch.sum(torch.isnan(sum(kernel_val))):
            ## We encountered a Nan in Kernel
            print(f'Bandwidth List : {bandwidth_list}')
            print(f'L2 Distance : {L2_distance}')
            ## Check for Nan in L2 distance
            print(f'L2 Nan : {torch.sum(torch.isnan(L2_distance))}')
            for bandwidth_temp in bandwidth_list:
                print(f'Temp: {bandwidth_temp}')
                print(f'BW Nan : {torch.sum(torch.isnan(L2_distance / bandwidth_temp))}')
        return sum(kernel_val), L2_distance

    def forward(self, source, target, labels1=None, labels2=None):
        ## Source  - [P*K, 2048], Target - [P*K, 2048]
        ## Devide them in "P" groups of "K" images
        rgb_features_list, ir_features_list = list(torch.split(source, [self.K] * self.P, dim=0)), list(torch.split(target, [self.K] * self.P, dim=0))
        total_loss = torch.tensor([0.], requires_grad=True).to(torch.device('cuda'))
        if labels1 is not None and labels2 is not None:
            rgb_labels, ir_labels = torch.split(labels1, [self.K] * self.P, dim=0), torch.split(labels2,
                                                                                                [self.K] * self.P,
                                                                                                dim=0)
            print(f'RGB Labels : {rgb_labels}')
            print(f'IR Labels : {ir_labels}')

        xx_batch, yy_batch, xy_batch, yx_batch = 0, 0, 0, 0

        for rgb_feat, ir_feat in zip(rgb_features_list, ir_features_list):
            source, target = rgb_feat, ir_feat  ## 4, 2048 ; 4*2048 -> 4*2048
            ## (rgb, ir, mid) -> rgb - mid + ir- mid ->
            batch_size = int(source.size()[0])
            kernels, l2dist = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul,
                                                   kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]

            xx_batch += torch.mean(XX)
            yy_batch += torch.mean(YY)
            xy_batch += torch.mean(XY)
            yx_batch += torch.mean(YX)

            if self.margin:
                loss = torch.mean(XX + YY - XY - YX)
                if loss - self.margin > 0:
                    total_loss += loss
                else:
                    total_loss += torch.clamp(loss - self.margin, min=0)

            else:
                total_loss += torch.mean(XX + YY - XY - YX)

        total_loss /= self.P
        return total_loss, torch.max(l2dist), [xx_batch / self.P, yy_batch / self.P, xy_batch / self.P,
                                               yx_batch / self.P]


class SupConLoss(nn.Module):
    def __init__(self, temperature = 0.1, scale_by_temperature = True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels = None, mask = None):
        n = features.size(0)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p = 2, dim=1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        #Compute similarity
        dist = torch.matmul(features, features.T) #inner product
        # dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n) #euclidean distance
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, features, features.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        #compute logits
        anchor_dot_contrast = torch.div(
            dist, self.temperature
        )

        # for numerical stability
        logits_max, index = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # tile mask
        logits_mask = torch.ones_like(mask).int() - torch.eye(batch_size).to(device)
        positive_mask = mask * logits_mask
        negative_mask = 1. - mask

        num_positive_per_row = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(exp_logits * negative_mask, dim=1, keepdim=True) + torch.sum(exp_logits * positive_mask, dim=1, keepdim=True)
        log_probs = logits - torch.log(denominator)

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positive_mask, dim=1)[num_positive_per_row > 0] / num_positive_per_row[num_positive_per_row > 0]

        #loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()

        return loss


def KL_divergence(logits_p, logits_q):
    p = F.softmax(logits_p, dim=1)
    q = F.softmax(logits_q, dim=1)

    shape = list(p.size())
    _shape = list(q.size())
    assert shape == _shape
    num_classes = shape[1]
    epsilon = 1e-8
    _p = (p + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    _q = (q + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    return torch.mean(torch.sum(_p * torch.log(_p / _q), 1))







