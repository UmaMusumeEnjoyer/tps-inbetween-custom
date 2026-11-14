# -*- coding: utf-8 -*-
"""
pipelines/tps_v22/metrics.py
Module đo lường chỉ số thẩm định chuẩn hóa (Standardized GPU & CPU Metrics):
- Chuẩn hóa 100% cách tính Ghost Pixels giữa vòng lặp train validation và fast benchmark.
- Tự động crop/slice vùng padding 32 về kích thước ảnh gốc [h, w] trước khi đo.
- Đo lường đầy đủ: PSNR, SSIM, Stroke Density Ratio, Stroke Recall, Chamfer Distance, Ghost Pixels, Blank Rate.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import cv2


# ==============================================================================
# 1. SSIM TRÊN GPU
# ==============================================================================

_GAUSSIAN_KERNEL = None

def _create_gaussian_kernel2d(kernel_size=11, sigma=1.5, device='cuda'):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return kernel_2d

def calculate_ssim_gpu(pred_1c, gt_1c):
    """Tính SSIM trực tiếp trên GPU bằng conv2d 11x11, không cấp phát heap CPU."""
    global _GAUSSIAN_KERNEL
    dev = pred_1c.device
    if _GAUSSIAN_KERNEL is None or _GAUSSIAN_KERNEL.device != dev:
        _GAUSSIAN_KERNEL = _create_gaussian_kernel2d(kernel_size=11, sigma=1.5, device=dev)

    pad = 5
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Chuyển về thang pixel [0, 1] (1 = trắng, 0 = đen)
    x = 1.0 - pred_1c
    y = 1.0 - gt_1c

    mu1 = F.conv2d(x, _GAUSSIAN_KERNEL, padding=pad)
    mu2 = F.conv2d(y, _GAUSSIAN_KERNEL, padding=pad)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sig1_sq = F.conv2d(x * x, _GAUSSIAN_KERNEL, padding=pad) - mu1_sq
    sig2_sq = F.conv2d(y * y, _GAUSSIAN_KERNEL, padding=pad) - mu2_sq
    sig12 = F.conv2d(x * y, _GAUSSIAN_KERNEL, padding=pad) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sig12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2) + 1e-8)
    return float(torch.clamp(ssim_map.mean(), 0.0, 1.0).item())


# ==============================================================================
# 2. CHAMFER DISTANCE (CPU SCIPY EXACT)
# ==============================================================================

def compute_chamfer_distance_np(img_pred_u8, img_gt_u8, thresh=200):
    """
    Tính Chamfer Distance hai chiều chuẩn hóa theo đường chéo khung hình.
    img_pred_u8, img_gt_u8: uint8 [0, 255] trong đó 0 là mực đen, 255 là nền trắng.
    """
    p_mask = (img_pred_u8 < thresh)
    g_mask = (img_gt_u8 < thresh)
    if not np.any(p_mask) and not np.any(g_mask):
        return 0.0
    if not np.any(p_mask) or not np.any(g_mask):
        return 1.0
    h, w = img_pred_u8.shape
    diag = np.sqrt(h**2 + w**2)
    edt_g = distance_transform_edt(~g_mask)
    edt_p = distance_transform_edt(~p_mask)
    cd_p2g = np.mean(edt_g[p_mask]) / diag
    cd_g2p = np.mean(edt_p[g_mask]) / diag
    return float(0.5 * (cd_p2g + cd_g2p))


# ==============================================================================
# 3. CHUẨN HÓA BỘ CHỈ SỐ TOÀN DIỆN CHO VÒNG LẶP VALIDATION
# ==============================================================================

def calculate_standardized_metrics(pred_tensor, gt_tensor, orig_h=None, orig_w=None, compute_chamfer=True):
    """
    Hàm tính toán chỉ số chuẩn hóa V2.2 đồng nhất 100% với Fast Benchmark:
    - pred_tensor, gt_tensor: Tensor ink space [B=1, 1, H, W] với 1.0 = mực đen, 0.0 = nền trắng.
    - Tự động cắt bỏ phần padding về [orig_h, orig_w].
    - Chuẩn hóa ngưỡng Ghost Pixels: pred_u8 in [180, 250] và gt_u8 > 250 (nền trắng).
    """
    with torch.no_grad():
        if orig_h is not None and orig_w is not None:
            p_slice = pred_tensor[:, :, :orig_h, :orig_w]
            g_slice = gt_tensor[:, :, :orig_h, :orig_w]
        else:
            p_slice = pred_tensor
            g_slice = gt_tensor

        # 1. PSNR trên GPU
        mse = F.mse_loss(p_slice, g_slice).item()
        psnr = float(20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-10))))

        # 2. SSIM trên GPU
        ssim = calculate_ssim_gpu(p_slice, g_slice)

        # 3. Chuyển sang định dạng pixel uint8 [0, 255] (0 = mực đen, 255 = nền trắng) trên GPU
        pred_u8 = ((1.0 - p_slice) * 255.0).clamp(0, 255).to(torch.uint8)
        gt_u8 = ((1.0 - g_slice) * 255.0).clamp(0, 255).to(torch.uint8)

        # 4. Stroke Metrics (Ngưỡng ink pixel < 200, tức mực > 55/255 ~ 0.21)
        p_lines = int(torch.sum(pred_u8 < 200).item())
        g_lines = int(torch.sum(gt_u8 < 200).item())
        density_ratio = float(p_lines / max(1, g_lines))

        # Kiểm tra hiện tượng sụp đổ ảnh trắng (Blank Collapse)
        is_blank = bool(p_lines < 100 and g_lines >= 500)

        # 5. Stroke Recall trên GPU (Dilated buffer 7x7)
        if g_lines < 50:
            recall = 1.0 if p_lines < 50 else 0.0
        else:
            p_mask = (pred_u8 < 200).float()
            g_mask = (gt_u8 < 200).float()
            p_dilated = F.max_pool2d(p_mask, kernel_size=7, stride=1, padding=3)
            overlap = torch.sum((g_mask > 0) & (p_dilated > 0)).item()
            recall = float(overlap / max(1, g_lines))

        # 6. Ghost Pixels chuẩn hóa (Đồng nhất bit-exact với fast benchmark):
        # Đếm số pixel xám mờ [180, 250] nằm ngoài nét vẽ GT (> 250)
        ghost_mask = (pred_u8 >= 180) & (pred_u8 <= 250) & (gt_u8 > 250)
        ghost_pixels = int(torch.sum(ghost_mask).item()) if not is_blank else -1

        # 7. Chamfer Distance
        chamfer_dist = 0.0
        if compute_chamfer and not is_blank:
            pred_np = pred_u8[0, 0].cpu().numpy()
            gt_np = gt_u8[0, 0].cpu().numpy()
            chamfer_dist = compute_chamfer_distance_np(pred_np, gt_np, thresh=200)

    return {
        'psnr': round(psnr, 2),
        'ssim': round(ssim, 4),
        'stroke_recall': round(recall, 4),
        'stroke_density_ratio': round(density_ratio, 4),
        'ghost_pixels': ghost_pixels,
        'chamfer_distance': round(chamfer_dist, 6),
        'is_blank_collapse': is_blank,
        'p_lines': p_lines,
        'g_lines': g_lines
    }
