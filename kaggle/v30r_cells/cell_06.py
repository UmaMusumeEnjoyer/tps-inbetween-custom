# [CELL 6]: Bộ Hàm Mất Mát Liên Tục CompositeLossEngine_v30R (DT Loss + Tversky-clDice + FineTip + Gold 1.10x)
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

class FastLaplacianEdgeLoss(nn.Module):
    """Toán tử Laplace vi phân bậc 2 tĩnh phát hiện cạnh biên và độ sắc nét siêu tốc (< 0.5ms)."""
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
        lap_p1 = F.conv2d(p_f, self.lap_kernel, padding=1)
        lap_g1 = F.conv2d(g_f, self.lap_kernel, padding=1)
        loss1 = F.l1_loss(lap_p1, lap_g1)

        p_sub = F.avg_pool2d(p_f, kernel_size=2, stride=2)
        g_sub = F.avg_pool2d(g_f, kernel_size=2, stride=2)
        lap_p2 = F.conv2d(p_sub, self.lap_kernel, padding=1)
        lap_g2 = F.conv2d(g_sub, self.lap_kernel, padding=1)
        loss2 = F.l1_loss(lap_p2, lap_g2)

        return loss1 + 0.5 * loss2

class TruncatedDifferentiableDTLoss(nn.Module):
    """
    Hàm mất mát biến đổi khoảng cách khả vi trên GPU (Truncated DT Loss, d_max=16px):
    Tạo mặt dốc liên tục dẫn đường cho nét dự đoán hội tụ về đúng vị trí hình học của Ground Truth.
    Theo đúng Mục 3.3 Công thức (10) của bài báo TPS gốc (Zhu et al., HIT 2024).
    """
    def __init__(self, steps=4):
        super(TruncatedDifferentiableDTLoss, self).__init__()
        self.steps = steps

    def _approx_dist_map(self, x):
        d_acc = 1.0 - x
        curr = x
        for s in range(1, self.steps + 1):
            k = 2 * s + 1
            dil = F.max_pool2d(curr, kernel_size=k, stride=1, padding=s)
            weight = 1.0 / (s + 1.0)
            d_acc = d_acc - weight * (dil - curr)
            curr = dil
        return torch.clamp(d_acc, 0.0, 1.0)

    def forward(self, pred, target):
        p_d = self._approx_dist_map(torch.clamp(pred, 0.0, 1.0))
        g_d = self._approx_dist_map(torch.clamp(target, 0.0, 1.0))
        return F.l1_loss(p_d, g_d)

class SoftTverskyClDiceLoss_v30R(nn.Module):
    """
    Tversky-clDice Loss ưu tiên Sensitivity (beta=0.65) hơn Precision (alpha=0.35):
    Buộc mạng nơ-ron không được làm đứt khung xương của các nét vẽ mảnh và ngọn tóc.
    """
    def __init__(self, iters=2, alpha=0.35, beta=0.65, smooth=1.0):
        super(SoftTverskyClDiceLoss_v30R, self).__init__()
        self.iters = iters
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target, t_skel=None):
        p_skel = soft_skeletonize(pred, iters=self.iters)
        if t_skel is None:
            t_skel = soft_skeletonize(target, iters=self.iters)
        tprec = (torch.sum(p_skel * target) + self.smooth) / (torch.sum(p_skel) + self.smooth)
        tsens = (torch.sum(t_skel * pred) + self.smooth) / (torch.sum(t_skel) + self.smooth)
        tversky = (tsens * tprec + self.smooth) / (self.beta * tsens + self.alpha * tprec + self.smooth)
        return 1.0 - tversky

class StrokeDensityRegularizationLoss_v30R(nn.Module):
    """Phạt mật độ nét vẽ ổn định qua Straight-Through Estimator (STE), neo chuẩn 1.10x."""
    def __init__(self, target_min=1.06, target_max=1.20, target_center=1.10, ramp_weight=0.25):
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
        ratio = torch.clamp(sum_p / torch.clamp(sum_g, min=20.0), min=0.0, max=5.0)

        dev_low = torch.clamp(self.target_min - ratio, min=0.0, max=3.0)
        dev_high = torch.clamp(ratio - self.target_max, min=0.0, max=3.0)
        loss_low = 3.0 * (dev_low ** 2)
        loss_high = 1.2 * dev_high + 0.8 * (dev_high ** 2)
        center_pull = 0.04 * ((ratio - self.target_center) ** 2)
        return torch.mean(loss_low + loss_high + center_pull)

class FineStrokeAndTipLoss(nn.Module):
    """
    Bảo vệ nét vẽ mảnh 1-2px (sợi tóc, mí mắt) và ngọn tóc vuốt nhọn (hair tips).
    Phạt bất đối xứng khi mạng dự đoán thiếu nét (under-prediction).
    """
    def __init__(self, focal=3.0):
        super().__init__()
        self.focal = focal

    def forward(self, pred_ink, gt_ink, skel_gt=None):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        opened_g = soft_open(g_ink)
        thin_strokes = torch.clamp(g_ink - opened_g, 0.0, 1.0)

        if skel_gt is None:
            skel_gt = soft_skeletonize(g_ink, iters=2)

        nbrs = F.avg_pool2d(skel_gt, kernel_size=3, stride=1, padding=1) * 9.0 - skel_gt
        endpoints = (nbrs <= 1.8) & (skel_gt > 0.20)
        endpoints_roi = soft_dilate(endpoints.float())

        w_fine = 1.0 + 3.0 * thin_strokes + 2.5 * endpoints_roi
        under_pred = F.relu(g_ink - p_ink) * w_fine
        return self.focal * torch.mean(under_pred)

class FBFCOcclusionAwareGhostLoss(nn.Module):
    """
    Ghost Loss nhận thức FBFC:
    Chỉ phạt bóng ma ngoài vùng hành lang 5x5 của GT tại nơi chuyển động nhất quán (conf_fbfc cao).
    Tuyệt đối không áp dụng cleanliness_penalty trần trụi làm bào mòn ngọn tóc.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, conf_fbfc=None):
        pred_f = torch.clamp(pred, 0.0, 1.0)
        gt_f = torch.clamp(gt, 0.0, 1.0)
        gt_safe_mask = soft_dilate(soft_dilate((gt_f > 0.08).float()))
        outside_safe = 1.0 - gt_safe_mask

        if conf_fbfc is not None:
            if conf_fbfc.shape[-2:] != pred_f.shape[-2:]:
                conf = conf_fbfc[:, :, :pred_f.shape[-2], :pred_f.shape[-1]].detach()
            else:
                conf = conf_fbfc.detach()
        else:
            conf = torch.ones_like(pred_f)
        far_ghost = F.relu(pred_f - 0.08) * outside_safe * conf
        dark_ghost = F.relu(pred_f - 0.35) * outside_safe * conf
        return 1.0 * torch.mean(far_ghost) + 2.5 * torch.mean(dark_ghost)

class CompositeLossEngine_v30R(nn.Module):
    """
    Bộ hàm mất mát CompositeLossEngine_v30R R2:
    - LIÊN TỤC VÀ KHẢ VI TỪ EPOCH 1 (Không còn nhảy bậc Loss đột ngột ở Epoch 3).
    - Kết hợp Truncated DT Loss + Tversky-clDice + FineStroke/HairTip + FBFC Ghost.
    """
    def __init__(self):
        super(CompositeLossEngine_v30R, self).__init__()
        self.dt_loss = TruncatedDifferentiableDTLoss(steps=4)
        self.laplacian_loss = FastLaplacianEdgeLoss()
        self.cldice = SoftTverskyClDiceLoss_v30R(iters=2, alpha=0.35, beta=0.65)
        self.density_loss = StrokeDensityRegularizationLoss_v30R(target_min=1.06, target_max=1.20, target_center=1.10)
        self.fine_tip_loss = FineStrokeAndTipLoss(focal=3.0)
        self.ghost_loss = FBFCOcclusionAwareGhostLoss()

    def forward(self, pred, target, conf_fbfc=None, epoch=1):
        loss_charb = torch.mean(torch.sqrt((pred - target) ** 2 + 1e-6))
        loss_dt = self.dt_loss(pred, target)
        loss_lap = self.laplacian_loss(pred, target)

        with torch.no_grad():
            skel_gt = soft_skeletonize(target, iters=2)

        loss_dice = self.cldice(pred, target, t_skel=skel_gt)
        loss_dens = self.density_loss(pred, target)
        loss_hair = self.fine_tip_loss(pred, target, skel_gt=skel_gt)
        loss_ghost = self.ghost_loss(pred, target, conf_fbfc=conf_fbfc)

        total = (1.0 * loss_charb + 
                 0.35 * loss_dt + 
                 0.20 * loss_lap + 
                 0.30 * loss_dice + 
                 0.25 * loss_hair + 
                 0.08 * loss_dens + 
                 0.05 * loss_ghost)

        return total, loss_charb, loss_dt, loss_dice, loss_dens, loss_hair, loss_ghost

device = globals().get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
criterion = CompositeLossEngine_v30R().to(device)
print("[✓] CompositeLossEngine_v30R R2 sẵn sàng (Continuous Loss + Truncated DT + Tversky-clDice + Gold 1.10x).")
sys.stdout.flush()
