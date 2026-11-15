# -*- coding: utf-8 -*-
"""
src/tps_v22/losses.py
Module các hàm mất mát (Loss Functions) cho Pipeline Finetune V2.2
Bao gồm:
- 3 hàm loss mới khả vi (Differentiable):
  1. StrokeDensityRegularizationLoss (lambda = 0.30)
  2. DirectionalTangentCoherenceLoss (lambda = 0.20)
  3. SoftClDiceLoss (lambda = 0.20)
- Các thành phần kế thừa hiệu quả từ V2.0.2:
  - L1 Loss (lambda = 1.00)
  - VGGPerceptualLoss (lambda = 0.15)
  - StrokeDiceLoss (lambda = 0.40)
  - GPUMorphologicalDistanceLoss (lambda = 0.20)
  - GhostStrokePenaltyLoss (lambda = 0.10)
- Module gộp: CompositeLossEngine_v22
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==============================================================================
# 1. CÁC HÀM MẤT MÁT KẾ THỪA TỪ V2.0.2
# ==============================================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-16 Perceptual Loss trích xuất đặc trưng biên nét ở các tầng nông."""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        try:
            vgg_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        except Exception:
            vgg_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*[vgg_features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[vgg_features[x] for x in range(4, 9)])
        del vgg_features
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, pred, target):
        # Chuyển tensor 1 kênh sang 3 kênh cho VGG
        pred_3c = pred.repeat(1, 3, 1, 1)
        target_3c = target.repeat(1, 3, 1, 1)
        h_pred1 = self.slice1(pred_3c)
        h_target1 = self.slice1(target_3c)
        loss = F.l1_loss(h_pred1, h_target1)
        h_pred2 = self.slice2(h_pred1)
        h_target2 = self.slice2(h_target1)
        loss += F.l1_loss(h_pred2, h_target2)
        return loss


class StrokeDiceLoss(nn.Module):
    """Stroke Dice Loss: Tối ưu hóa độ chồng khít của nét vẽ."""
    def __init__(self, eps=1e-5):
        super(StrokeDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        intersection = 2.0 * torch.sum(pred * target, dim=[-3, -2, -1]) + self.eps
        union = torch.sum(pred ** 2, dim=[-3, -2, -1]) + torch.sum(target ** 2, dim=[-3, -2, -1]) + self.eps
        dice = intersection / union
        return torch.mean(1.0 - dice)


class GPUMorphologicalDistanceLoss(nn.Module):
    """
    GPUMorphologicalDistanceLoss: Phạt khoảng cách hình thái học đa thang đo
    giữa nét dự đoán và nét Ground Truth trên GPU.
    """
    def __init__(self, scales=(3, 5, 9, 15)):
        super(GPUMorphologicalDistanceLoss, self).__init__()
        self.scales = scales

    def forward(self, pred, target):
        p_bin = (pred > 0.15).float()
        t_bin = (target > 0.15).float()

        loss_accum = 0.0
        for k in self.scales:
            pad = k // 2
            t_dilated = F.max_pool2d(t_bin, kernel_size=k, stride=1, padding=pad)
            p_dilated = F.max_pool2d(p_bin, kernel_size=k, stride=1, padding=pad)

            loss_p2t = torch.mean(pred * (1.0 - t_dilated))
            loss_t2p = torch.mean(target * (1.0 - p_dilated))
            loss_accum += (loss_p2t + loss_t2p)
        return loss_accum / float(len(self.scales))


class GhostStrokePenaltyLoss(nn.Module):
    """
    GhostStrokePenaltyLoss: Phạt nặng các pixel nét ma xám mờ xuất hiện
    ở vùng nền trắng ngoài vùng bảo vệ GT (bán kính radius=7).
    """
    def __init__(self, radius=7):
        super(GhostStrokePenaltyLoss, self).__init__()
        self.radius = radius

    def forward(self, pred, target):
        pad = self.radius // 2
        t_dilated = F.max_pool2d((target > 0.15).float(), kernel_size=self.radius, stride=1, padding=pad)
        outside_gt = 1.0 - t_dilated
        # Phạt năng lượng bóng ma lơ lửng pred > 0.08 ngoài vùng nét vẽ GT
        ghost_energy = F.relu(pred - 0.08) * outside_gt
        return torch.mean(ghost_energy ** 2)


# ==============================================================================
# 2. CÁC HÀM MẤT MÁT ĐỘT PHÁ MỚI CHO V2.2
# ==============================================================================

class StrokeDensityRegularizationLoss(nn.Module):
    """
    Hàm mất mát điều tiết mật độ mực nét vẽ (Stroke Density Regularization):
    - Mục tiêu: Kéo Density Ratio từ mức phình nét 1.3139 về khoảng chuẩn [1.00 - 1.05].
    - Công thức: ratio = sum(pred_ink) / (sum(gt_ink) + 1e-6)
    - Phạt Smooth L1 khi ratio lệch ra ngoài dải dung sai [target_min, target_max] = [0.98, 1.03].
      Loss = |ratio - 1.0| (khả vi 100%).
    """
    def __init__(self, target_min=0.98, target_max=1.03, eps=1e-6):
        super(StrokeDensityRegularizationLoss, self).__init__()
        self.target_min = target_min
        self.target_max = target_max
        self.eps = eps

    def forward(self, pred, target, ink_mode=True):
        # Quy ước: Nếu ink_mode=True (chuẩn training tensor), 1.0 là mực đen, 0.0 là nền trắng.
        # Nếu ink_mode=False (chuẩn ảnh pixel), 0.0 là mực đen, 1.0 là nền trắng.
        if not ink_mode:
            p_ink = 1.0 - pred
            g_ink = 1.0 - target
        else:
            p_ink = pred
            g_ink = target

        sum_p = torch.sum(p_ink, dim=[-3, -2, -1])
        sum_g = torch.sum(g_ink, dim=[-3, -2, -1])
        ratio = sum_p / (sum_g + self.eps)

        dev_low = F.relu(self.target_min - ratio)
        dev_high = F.relu(ratio - self.target_max)
        is_outside = (dev_low > 0) | (dev_high > 0)

        diff_from_one = torch.abs(ratio - 1.0)
        # Sử dụng Smooth L1 trên độ lệch khỏi 1.0 khi nằm ngoài dải [0.98, 1.03]
        loss = torch.where(
            is_outside,
            F.smooth_l1_loss(diff_from_one, torch.zeros_like(diff_from_one), reduction='none', beta=0.01),
            torch.zeros_like(diff_from_one)
        ).mean()
        return loss, ratio.mean().item()


class DirectionalTangentCoherenceLoss(nn.Module):
    """
    Hàm mất mát nhất quán hướng tiếp tuyến nét vẽ (Directional Tangent Coherence Loss):
    - Sử dụng bộ lọc Sobel 3x3 khả vi để tính gradient (Gx, Gy).
    - Tính Cosine Similarity của vector hướng giữa nét dự đoán và Ground Truth.
    - L_tangent = 1.0 - |cos(theta_pred, theta_gt)| trên vùng mask nét vẽ Ground Truth.
    - Triệt tiêu hiện tượng nét bị gãy khúc, bậc thang hoặc răng cưa ở các đường cong.
    """
    def __init__(self, eps=1e-8):
        super(DirectionalTangentCoherenceLoss, self).__init__()
        self.eps = eps
        # Bộ lọc Sobel 3x3 khả vi chuẩn hóa
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        dev = pred.device
        gx_p = F.conv2d(pred, self.sobel_x.to(dev), padding=1)
        gy_p = F.conv2d(pred, self.sobel_y.to(dev), padding=1)
        gx_g = F.conv2d(target, self.sobel_x.to(dev), padding=1)
        gy_g = F.conv2d(target, self.sobel_y.to(dev), padding=1)

        mag_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + self.eps)
        mag_g = torch.sqrt(gx_g ** 2 + gy_g ** 2 + self.eps)

        # Dot product giữa 2 vector gradient
        dot = gx_p * gx_g + gy_p * gy_g
        cos_sim = dot / (mag_p * mag_g)

        # |cos(theta)| tính góc định hướng đường nét (đồng phương không kể chiều)
        loss_pixel = 1.0 - torch.abs(cos_sim)

        # Mask tập trung vào vùng biên nét của Ground Truth (tránh vùng nền phẳng)
        mask = (mag_g > 0.02) | (target > 0.10)
        mask_f = mask.float()
        loss = torch.sum(loss_pixel * mask_f) / (torch.sum(mask_f) + 1e-5)
        return loss


class SoftClDiceLoss(nn.Module):
    """
    Centerline Dice Loss khả vi (Soft-clDice Loss):
    - Cài đặt soft_skeletonize(x, iter=5) bằng các phép toán khả vi F.max_pool2d và min-pool (-max_pool(-x)).
    - Trích xuất khung xương nét vẽ và tính Centerline Dice Loss.
    - Chống đứt gãy nét mảnh (tóc, mắt, ngón tay) mà KHÔNG làm phình nét.
    """
    def __init__(self, iters=5, eps=1e-5):
        super(SoftClDiceLoss, self).__init__()
        self.iters = iters
        self.eps = eps

    def soft_erode(self, img):
        # Phép co hình thái học khả vi: -max_pool(-img)
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

    def soft_open(self, img):
        # Phép mở hình thái học khả vi: dilate(erode(img))
        return F.max_pool2d(self.soft_erode(img), kernel_size=3, stride=1, padding=1)

    def soft_skeletonize(self, img):
        skel = torch.zeros_like(img)
        x = img
        for _ in range(self.iters):
            opened = self.soft_open(x)
            sub = F.relu(x - opened)
            skel = torch.max(skel, sub)
            x = self.soft_erode(x)
        return skel

    def forward(self, pred, target):
        skel_pred = self.soft_skeletonize(pred)
        skel_gt = self.soft_skeletonize(target)

        # Centerline Precision & Sensitivity
        tprec = (torch.sum(skel_pred * target, dim=[-3, -2, -1]) + self.eps) / (torch.sum(skel_pred, dim=[-3, -2, -1]) + self.eps)
        tsens = (torch.sum(skel_gt * pred, dim=[-3, -2, -1]) + self.eps) / (torch.sum(skel_gt, dim=[-3, -2, -1]) + self.eps)

        cldice = (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
        return torch.mean(1.0 - cldice)


# ==============================================================================
# 3. COMPOSITE LOSS ENGINE V2.2
# ==============================================================================

class CompositeLossEngine_v22(nn.Module):
    """
    Bộ động cơ mất mát tích hợp V2.2 (100% PyTorch GPU Native):
    - L1 Loss                 (lambda = 1.00)
    - VGG Perceptual          (lambda = 0.15)
    - Stroke Dice             (lambda = 0.40)
    - Morphological Distance  (lambda = 0.20)
    - Ghost Stroke Penalty    (lambda = 0.10)
    - Stroke Density Regular. (lambda = 0.30) [MỚI V2.2]
    - Directional Tangent     (lambda = 0.20) [MỚI V2.2]
    - Soft-clDice             (lambda = 0.20) [MỚI V2.2]
    """
    def __init__(self,
                 lambda_l1=1.00,
                 lambda_vgg=0.15,
                 lambda_dice=0.40,
                 lambda_morph=0.20,
                 lambda_ghost=0.10,
                 lambda_density=0.30,
                 lambda_tangent=0.20,
                 lambda_cldice=0.20):
        super(CompositeLossEngine_v22, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_dice = lambda_dice
        self.lambda_morph = lambda_morph
        self.lambda_ghost = lambda_ghost
        self.lambda_density = lambda_density
        self.lambda_tangent = lambda_tangent
        self.lambda_cldice = lambda_cldice

        self.vgg_loss_fn = VGGPerceptualLoss()
        self.dice_loss_fn = StrokeDiceLoss()
        self.morph_dist_fn = GPUMorphologicalDistanceLoss()
        self.ghost_loss_fn = GhostStrokePenaltyLoss(radius=7)
        self.density_loss_fn = StrokeDensityRegularizationLoss(target_min=0.98, target_max=1.03)
        self.tangent_loss_fn = DirectionalTangentCoherenceLoss()
        self.cldice_loss_fn = SoftClDiceLoss(iters=5)

    def forward(self, pred, target):
        loss_l1 = F.l1_loss(pred, target)
        loss_vgg = self.vgg_loss_fn(pred, target)
        loss_dice = self.dice_loss_fn(pred, target)
        loss_morph = self.morph_dist_fn(pred, target)
        loss_ghost = self.ghost_loss_fn(pred, target)
        loss_density, density_ratio = self.density_loss_fn(pred, target, ink_mode=True)
        loss_tangent = self.tangent_loss_fn(pred, target)
        loss_cldice = self.cldice_loss_fn(pred, target)

        total_loss = (
            self.lambda_l1 * loss_l1 +
            self.lambda_vgg * loss_vgg +
            self.lambda_dice * loss_dice +
            self.lambda_morph * loss_morph +
            self.lambda_ghost * loss_ghost +
            self.lambda_density * loss_density +
            self.lambda_tangent * loss_tangent +
            self.lambda_cldice * loss_cldice
        )

        loss_dict = {
            'total_loss': total_loss,
            'loss_l1': loss_l1,
            'loss_vgg': loss_vgg,
            'loss_dice': loss_dice,
            'loss_morph': loss_morph,
            'loss_ghost': loss_ghost,
            'loss_density': loss_density,
            'loss_tangent': loss_tangent,
            'loss_cldice': loss_cldice,
            'batch_density_ratio': density_ratio
        }
        return total_loss, loss_dict
