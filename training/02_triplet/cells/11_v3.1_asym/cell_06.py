# [CELL 6]: Bộ Hàm Mất Mát CompositeLossEngine_v31 (Fast Laplacian, ClDice-2, Gold 1.10x & Ghost-11)
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

def soft_skeletonize(img, iters=2):
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    for _ in range(iters - 1):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img))
    return torch.clamp(skel, 0.0, 1.0)

# =============================================================================
# [LOSS 1 - V3.1]: Fast Multi-Scale Laplacian Edge Loss (Thay thế VGG16 nặng nề)
# =============================================================================
class FastLaplacianEdgeLoss(nn.Module):
    """
    Toán tử Laplace vi phân bậc 2 tĩnh: Phát hiện cạnh biên và độ sắc nét siêu tốc (< 0.5ms).
    Nhạy cảm tuyệt đối với cấu trúc nét vẽ anime 1-2 pixel mà không tốn tài nguyên GPU.
    """
    def __init__(self):
        super(FastLaplacianEdgeLoss, self).__init__()
        lap_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

    def forward(self, pred, target):
        p_f = pred.float()
        g_f = target.float()
        # Thang đo gốc 1x
        lap_p1 = F.conv2d(p_f, self.lap_kernel, padding=1)
        lap_g1 = F.conv2d(g_f, self.lap_kernel, padding=1)
        loss1 = F.l1_loss(lap_p1, lap_g1)

        # Thang đo nửa 0.5x
        p_sub = F.avg_pool2d(p_f, kernel_size=2, stride=2)
        g_sub = F.avg_pool2d(g_f, kernel_size=2, stride=2)
        lap_p2 = F.conv2d(p_sub, self.lap_kernel, padding=1)
        lap_g2 = F.conv2d(g_sub, self.lap_kernel, padding=1)
        loss2 = F.l1_loss(lap_p2, lap_g2)

        return loss1 + 0.5 * loss2

# =============================================================================
# [LOSS 2 - V3.1]: Tinh Gọn Soft-clDice Topology Loss (iters=2)
# =============================================================================
class SoftClDiceLoss_v31(nn.Module):
    def __init__(self, iters=2, smooth=1.0):
        super(SoftClDiceLoss_v31, self).__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred, target, t_skel=None):
        p_skel = soft_skeletonize(pred, iters=self.iters)
        if t_skel is None:
            t_skel = soft_skeletonize(target, iters=self.iters)
        tprec = (torch.sum(p_skel * target) + self.smooth) / (torch.sum(p_skel) + self.smooth)
        tsens = (torch.sum(t_skel * pred) + self.smooth) / (torch.sum(t_skel) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
        return 1.0 - cldice

# =============================================================================
# [LOSS 3 - V3.1]: Khóa Chuẩn Mật Độ Nét Vàng 1.10x (Khớp Chuẩn V2.6)
# =============================================================================
class StrokeDensityRegularizationLoss_v31(nn.Module):
    def __init__(self, target_min=1.06, target_max=1.14, target_center=1.10, ramp_weight=0.25):
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
        dev_low = torch.clamp(self.target_min - ratio, min=0.0, max=3.0)
        dev_high = torch.clamp(ratio - self.target_max, min=0.0, max=3.0)
        loss_low = 3.0 * (dev_low ** 2)
        loss_high = 2.0 * dev_high + 1.5 * (dev_high ** 2)
        center_pull = 0.08 * ((ratio - self.target_center) ** 2)
        return torch.mean(loss_low + loss_high + center_pull)

# =============================================================================
# [LOSS 4 - V3.1]: Bảo Toàn Đỉnh Ngọn Tóc & Chi Tiết Mảnh
# =============================================================================
class HairTipEndpointContinuityLoss_v31(nn.Module):
    def __init__(self, endpoint_focal=1.5):
        super().__init__()
        self.endpoint_focal = endpoint_focal

    def forward(self, pred_ink, gt_ink, skel_gt=None):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        if skel_gt is None:
            skel_gt = soft_skeletonize(g_ink, iters=2)
        nbrs = F.avg_pool2d(skel_gt, kernel_size=3, stride=1, padding=1) * 9.0 - skel_gt
        endpoints = (nbrs <= 1.8) & (skel_gt > 0.20)
        hair_tip_roi = soft_dilate(endpoints.float())
        tip_under = F.relu(g_ink - p_ink) * hair_tip_roi
        return self.endpoint_focal * torch.mean(tip_under)

# =============================================================================
# [LOSS 5 - V3.1]: Occlusion-Aware Anti-Ghosting Với Vùng Đệm An Toàn Mở Rộng
# =============================================================================
class OcclusionAwareGhostLoss_v31(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, v0, v1):
        pred_f = torch.clamp(pred, 0.0, 1.0)
        gt_f = torch.clamp(gt, 0.0, 1.0)
        # Nới rộng bán kính an toàn bằng 2 lần soft_dilate (~9x9 pixels)
        gt_safe_mask = soft_dilate(soft_dilate((gt_f > 0.08).float()))
        outside_safe = 1.0 - gt_safe_mask

        far_ghost = F.relu(pred_f - 0.04) * outside_safe
        dark_ghost = F.relu(pred_f - 0.30) * outside_safe
        cleanliness_penalty = 0.25 * F.relu(pred_f - 0.01) * outside_safe
        gate_penalty = F.relu(v0 + v1 - 1.08)
        return 1.0 * torch.mean(far_ghost) + 2.5 * torch.mean(dark_ghost) + torch.mean(cleanliness_penalty) + 0.08 * torch.mean(gate_penalty)

# =============================================================================
# [BỘ LOSS TỔNG HỢP]: CompositeLossEngine_v31
# =============================================================================
class CompositeLossEngine_v31(nn.Module):
    def __init__(self):
        super(CompositeLossEngine_v31, self).__init__()
        self.laplacian_loss = FastLaplacianEdgeLoss()
        self.cldice = SoftClDiceLoss_v31(iters=2)
        self.density_loss = StrokeDensityRegularizationLoss_v31(target_min=1.06, target_max=1.14, target_center=1.10)
        self.hair_tip_loss = HairTipEndpointContinuityLoss_v31()
        self.ghost_loss = OcclusionAwareGhostLoss_v31()

    def forward(self, pred, target, v0=None, v1=None, epoch=1):
        loss_l1 = F.l1_loss(pred, target)
        loss_lap = self.laplacian_loss(pred, target)
        
        # Tối ưu: Precompute GT skeleton một lần duy nhất (không cần graph đạo hàm trên GT)
        with torch.no_grad():
            skel_gt = soft_skeletonize(target, iters=2)

        if epoch <= 2:
            # Epoch 1-2: Hội tụ thô ổn định trên L1 + Laplacian + clDice nhẹ
            # Theo Mục 2.5 & Mục 4 của Optimization Guide: không kích hoạt density/hair/ghost sớm
            loss_dice = self.cldice(pred, target, t_skel=skel_gt)
            total = 1.0 * loss_l1 + 0.25 * loss_lap + 0.05 * loss_dice
            loss_dens = torch.tensor(0.0, device=pred.device)
            loss_hair = torch.tensor(0.0, device=pred.device)
        else:
            # Epoch 3+: Bổ sung điều tiết mật độ nét chuẩn vàng 1.10x, bảo toàn ngọn tóc & anti-ghosting
            loss_dice = self.cldice(pred, target, t_skel=skel_gt)
            loss_dens = self.density_loss(pred, target)
            loss_hair = self.hair_tip_loss(pred, target, skel_gt=skel_gt)
            loss_ghost = self.ghost_loss(pred, target, v0, v1) if (v0 is not None and v1 is not None) else torch.tensor(0.0, device=pred.device)
            total = 1.0 * loss_l1 + 0.25 * loss_lap + 0.10 * loss_dice + 0.15 * loss_dens + 0.10 * loss_hair + 0.05 * loss_ghost

        return total, loss_l1, loss_dice, loss_dens, loss_hair

criterion = CompositeLossEngine_v31().to(device)
print("[✓] CompositeLossEngine_v31 sẵn sàng (Fast Laplacian + clDice-2 + Gold 1.10x + Safe Ghost).")
sys.stdout.flush()
