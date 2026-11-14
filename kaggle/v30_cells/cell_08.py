# [CELL 8]: Hàm Đo Lường Chỉ Số Thẩm Định Chống Bẫy V3.0 (100% Batched Pure GPU Metrics)
import os
import sys
import math
import torch
import torch.nn.functional as F
def _create_gaussian_1d_kernel(kernel_size=11, sigma=1.5, device='cuda'):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    g_x = g.view(1, 1, 1, kernel_size).repeat(5, 1, 1, 1)
    g_y = g.view(1, 1, kernel_size, 1).repeat(5, 1, 1, 1)
    return g_x, g_y

_GAUSS_X = None
_GAUSS_Y = None

def calculate_metrics_batch_gpu(pred_tensor, gt_tensor):
    global _GAUSS_X, _GAUSS_Y
    with torch.no_grad():
        p = torch.clamp(pred_tensor.float(), 0.0, 1.0)
        g = torch.clamp(gt_tensor.float(), 0.0, 1.0)
        b, c, h, w = p.shape
        dev = p.device

        # 1. Batched PSNR
        mse = torch.mean((p - g) ** 2, dim=[-3, -2, -1])
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

        # 2. Batched GPU SSIM
        if _GAUSS_X is None or _GAUSS_X.device != dev:
            _GAUSS_X, _GAUSS_Y = _create_gaussian_1d_kernel(11, 1.5, dev)
        
        gx = _GAUSS_X[:c] if _GAUSS_X.shape[0] >= c else _GAUSS_X[0:1].repeat(c, 1, 1, 1)
        gy = _GAUSS_Y[:c] if _GAUSS_Y.shape[0] >= c else _GAUSS_Y[0:1].repeat(c, 1, 1, 1)

        def _conv2d_separable(t):
            t_pad = F.pad(t, (5, 5, 5, 5), mode='reflect')
            return F.conv2d(F.conv2d(t_pad, gx, groups=c), gy, groups=c)

        mu_p = _conv2d_separable(p)
        mu_g = _conv2d_separable(g)
        mu_p_sq = mu_p ** 2
        mu_g_sq = mu_g ** 2
        mu_pg = mu_p * mu_g
        sigma_p_sq = _conv2d_separable(p * p) - mu_p_sq
        sigma_g_sq = _conv2d_separable(g * g) - mu_g_sq
        sigma_pg = _conv2d_separable(p * g) - mu_pg
        c1_s = 0.01 ** 2
        c2_s = 0.03 ** 2
        ssim_map = ((2.0 * mu_pg + c1_s) * (2.0 * sigma_pg + c2_s)) / ((mu_p_sq + mu_g_sq + c1_s) * (sigma_p_sq + sigma_g_sq + c2_s))
        ssim = torch.mean(ssim_map, dim=[-3, -2, -1])

        # 3. Stroke Recall & F1
        p_bin = (p > 0.15).float()
        g_bin = (g > 0.15).float()
        tp = torch.sum(p_bin * g_bin, dim=[-3, -2, -1])
        fn = torch.sum((1.0 - p_bin) * g_bin, dim=[-3, -2, -1])
        fp = torch.sum(p_bin * (1.0 - g_bin), dim=[-3, -2, -1])
        recall = tp / torch.clamp(tp + fn, min=1e-5)
        f1 = (2.0 * tp) / torch.clamp(2.0 * tp + fp + fn, min=1e-5)

        # 4. Density Ratio
        sum_p = torch.sum(p_bin, dim=[-3, -2, -1])
        sum_g = torch.sum(g_bin, dim=[-3, -2, -1])
        density = sum_p / torch.clamp(sum_g, min=1.0)

        # 5. Ghost Pixels & Dark Ghost
        g_dilated = soft_dilate(g_bin)
        outside_core = 1.0 - g_dilated
        ghost_px = torch.sum((p > 0.10).float() * outside_core, dim=[-3, -2, -1])
        dark_ghost_px = torch.sum((p > 0.30).float() * outside_core, dim=[-3, -2, -1])

        # 6. Approximated Chamfer
        p_sum = torch.clamp(sum_p, min=1.0)
        chamfer = torch.sum(torch.abs(p - g) * (1.0 - g), dim=[-3, -2, -1]) / (p_sum * max(1, h * w))

        return (
            ssim.detach().cpu().numpy(),
            psnr.detach().cpu().numpy(),
            recall.detach().cpu().numpy(),
            f1.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            ghost_px.detach().cpu().numpy(),
            dark_ghost_px.detach().cpu().numpy(),
            chamfer.detach().cpu().numpy()
        )

print("[✓] Bộ đo lường batched GPU metrics sẵn sàng (SSIM, PSNR, Recall, F1, Density, Ghost, Chamfer).")
sys.stdout.flush()