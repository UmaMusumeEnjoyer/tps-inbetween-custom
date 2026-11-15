# [CELL 6]: Bộ Hàm Mất Mát CompositeLossEngine_v30 (2-Stage Curriculum, Gold 1.10x & Hair-Tip Loss)
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img, iters=3):
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    for _ in range(iters - 1):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img))
    return torch.clamp(skel, 0.0, 1.0)

class SoftClDiceLoss(nn.Module):
    def __init__(self, iters=3, smooth=1.0):
        super(SoftClDiceLoss, self).__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred, target):
        p_skel = soft_skeletonize(pred, iters=self.iters)
        t_skel = soft_skeletonize(target, iters=self.iters)
        tprec = (torch.sum(p_skel * target) + self.smooth) / (torch.sum(p_skel) + self.smooth)
        tsens = (torch.sum(t_skel * pred) + self.smooth) / (torch.sum(t_skel) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
        return 1.0 - cldice

class StrokeDensityRegularizationLoss_v3(nn.Module):
    def __init__(self, target_min=1.04, target_max=1.14, target_center=1.10, ramp_weight=0.25):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max
        self.target_center = target_center
        self.ramp_weight = ramp_weight

    def forward(self, pred_ink, gt_ink):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        m_soft_p = torch.clamp((p_ink - 0.06) / 0.18, 0.0, 1.0) + self.ramp_weight * F.relu(p_ink - 0.24)
        m_soft_g = torch.clamp((g_ink - 0.06) / 0.18, 0.0, 1.0) + self.ramp_weight * F.relu(g_ink - 0.24)
        m_hard_p = (p_ink > 0.15).float()
        m_hard_g = (g_ink > 0.15).float()
        m_ste_p = m_hard_p + (m_soft_p - m_soft_p.detach())
        m_ste_g = m_hard_g + (m_soft_g - m_soft_g.detach())
        sum_p = torch.sum(m_ste_p, dim=[-3, -2, -1])
        sum_g = torch.sum(m_ste_g, dim=[-3, -2, -1])
        ratio = sum_p / torch.clamp(sum_g, min=10.0)
        dev_low = F.relu(self.target_min - ratio)
        dev_high = F.relu(ratio - self.target_max)
        loss_low = dev_low ** 2
        loss_high = 2.5 * dev_high + 1.5 * (dev_high ** 2)
        center_pull = 0.05 * ((ratio - self.target_center) ** 2)
        return torch.mean(loss_low + loss_high + center_pull)

class HairTipEndpointContinuityLoss(nn.Module):
    def __init__(self, endpoint_focal=4.0):
        super().__init__()
        self.endpoint_focal = endpoint_focal

    def forward(self, pred_ink, gt_ink):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        skel_gt = soft_skeletonize(g_ink, iters=3)
        nbrs = F.avg_pool2d(skel_gt, kernel_size=3, stride=1, padding=1) * 9.0 - skel_gt
        endpoints = (nbrs <= 1.8) & (skel_gt > 0.20)
        hair_tip_roi = soft_dilate(endpoints.float())
        tip_under = F.relu(g_ink - p_ink) * hair_tip_roi
        return self.endpoint_focal * torch.mean(tip_under)

class OcclusionAwareGhostLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, v0, v1):
        pred_f = torch.clamp(pred, 0.0, 1.0)
        gt_f = torch.clamp(gt, 0.0, 1.0)
        outside_gt = (gt_f < 0.05).float()
        far_ghost = F.relu(pred_f - 0.03) * outside_gt
        dark_ghost = F.relu(pred_f - 0.30) * outside_gt
        cleanliness_penalty = 0.35 * F.relu(pred_f - 0.008) * outside_gt
        gate_penalty = F.relu(v0 + v1 - 1.05)
        return 1.2 * torch.mean(far_ghost) + 3.0 * torch.mean(dark_ghost) + torch.mean(cleanliness_penalty) + 0.10 * torch.mean(gate_penalty)

class CompositeLossEngine_v30(nn.Module):
    def __init__(self):
        super(CompositeLossEngine_v30, self).__init__()
        self.cldice = SoftClDiceLoss(iters=3)
        self.density_loss = StrokeDensityRegularizationLoss_v3(target_min=1.04, target_max=1.14, target_center=1.10)
        self.hair_tip_loss = HairTipEndpointContinuityLoss()
        self.ghost_loss = OcclusionAwareGhostLoss()

    def forward(self, pred, target, v0=None, v1=None, epoch=1):
        loss_l1 = F.l1_loss(pred, target)
        loss_dens = self.density_loss(pred, target)
        loss_hair = self.hair_tip_loss(pred, target)

        if epoch <= 2:
            # Giai đoạn 1 Warmup: Tập trung căn chỉnh L1, Density và HairTip
            total = 1.0 * loss_l1 + 0.55 * loss_dens + 0.30 * loss_hair
            loss_dice = torch.tensor(0.0, device=pred.device)
        else:
            # Giai đoạn 2: Bật toàn bộ clDice và Ghost Suppression
            loss_dice = self.cldice(pred, target)
            loss_ghost = self.ghost_loss(pred, target, v0, v1) if (v0 is not None and v1 is not None) else torch.tensor(0.0, device=pred.device)
            total = 1.0 * loss_l1 + 0.15 * loss_dice + 0.55 * loss_dens + 0.30 * loss_hair + 0.20 * loss_ghost

        return total, loss_l1, loss_dice, loss_dens, loss_hair

criterion = CompositeLossEngine_v30().to(device)
print("[✓] CompositeLossEngine_v30 sẵn sàng (2-Stage Curriculum + Gold 1.10x + Hair-Tip Loss + Cleanliness Penalty).")
sys.stdout.flush()