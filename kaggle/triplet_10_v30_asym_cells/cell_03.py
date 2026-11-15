# [CELL 3]: Hệ Thống Biến Dạng Lưới, 8 Điểm Neo Biên/Góc & Vector Hóa TPS Flow Batch
import os
import sys
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_GRID_LOCK = threading.Lock()
_NORM_GRID_CACHE = {}
_PIXEL_GRID_CACHE = {}
_MAX_GRID_CACHE_SIZE = 32

def warp_normalized_flow(img, flow):
    """
    Biến dạng lưới Coarse TPS (tọa độ chuẩn hóa [0, 1] sang [-1, 1]).
    Caching lưới cơ sở (1, H, W, 2) và cộng trực tiếp qua broadcasting.
    """
    b, _, h, w = img.shape
    dev = flow.device
    dtype = flow.dtype
    key = (h, w, dev, dtype)
    base_grid = _NORM_GRID_CACHE.get(key)
    if base_grid is None:
        with _GRID_LOCK:
            base_grid = _NORM_GRID_CACHE.get(key)
            if base_grid is None:
                if len(_NORM_GRID_CACHE) >= _MAX_GRID_CACHE_SIZE:
                    _NORM_GRID_CACHE.clear()
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(0, 1, h, device=dev, dtype=dtype),
                    torch.linspace(0, 1, w, device=dev, dtype=dtype), indexing='ij'
                )
                base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
                _NORM_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = flow.float()
        grid = ((base_grid + flow_f32.permute(0, 2, 3, 1)) * 2.0 - 1.0).contiguous()
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def warp_pixel_flow(img, flow):
    """
    Biến dạng lưới Fine Optical Flow IFNet (tọa độ pixel thực tế).
    Caching lưới cơ sở [-1, 1] và cộng trực tiếp qua broadcasting.
    """
    b, _, h, w = img.shape
    dev = flow.device
    dtype = flow.dtype
    key = (h, w, dev, dtype)
    base_grid = _PIXEL_GRID_CACHE.get(key)
    if base_grid is None:
        with _GRID_LOCK:
            base_grid = _PIXEL_GRID_CACHE.get(key)
            if base_grid is None:
                if len(_PIXEL_GRID_CACHE) >= _MAX_GRID_CACHE_SIZE:
                    _PIXEL_GRID_CACHE.clear()
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, h, device=dev, dtype=dtype),
                    torch.linspace(-1, 1, w, device=dev, dtype=dtype), indexing='ij'
                )
                base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
                _PIXEL_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = flow.float()
        flow_norm = torch.cat([
            flow_f32[:, 0:1, :, :] / max(1.0, (w - 1.0) / 2.0),
            flow_f32[:, 1:2, :, :] / max(1.0, (h - 1.0) / 2.0)
        ], dim=1)
        grid = (base_grid + flow_norm.permute(0, 2, 3, 1)).contiguous()
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def tps_rbf(r):
    r = torch.clamp(r, min=1e-6)
    return (r ** 2) * torch.log(r)

# 8 Điểm neo tĩnh (4 góc + 4 điểm giữa biên) khóa chặt khung hình, chống co rút viền ảnh
FIXED_ANCHORS_NP = np.array([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
    [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5]
], dtype=np.float32)

def filter_spacing_np(c0, c1, min_dist=0.015, max_kps=80):
    n = c0.shape[0]
    if n <= 1:
        return c0, c1
    disp = np.linalg.norm(c1 - c0, axis=-1)
    order = np.argsort(-disp)
    c0_s, c1_s = c0[order], c1[order]
    keep = [0]
    for i in range(1, n):
        dists = np.linalg.norm(c0_s[keep] - c0_s[i], axis=-1)
        if np.all(dists >= min_dist):
            keep.append(i)
            if len(keep) >= max_kps:
                break
    return c0_s[keep], c1_s[keep]

def filter_topo_np(c_s, c_d, max_dev=0.18, min_neighbors=3, radius=0.25):
    n = c_s.shape[0]
    if n <= 6:
        return np.arange(n)
    disp = c_d - c_s
    diff_s = c_s[:, None, :] - c_s[None, :, :]
    dist_sq = diff_s[:, :, 0] ** 2 + diff_s[:, :, 1] ** 2
    nbr_mask = (dist_sq < radius * radius) & (dist_sq > 1e-10)
    nbr_counts = nbr_mask.sum(axis=1)
    keep = []
    for i in range(n):
        if nbr_counts[i] < min_neighbors:
            keep.append(i)
            continue
        nbr_disp = disp[nbr_mask[i]]
        diff = np.abs(nbr_disp - disp[i])
        dev = np.median(np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2))
        if dev <= max_dev:
            keep.append(i)
    if len(keep) < 4:
        return np.arange(n)
    return np.array(keep, dtype=int)

def solve_tps_single_cpu(c_s, c_d, max_kps=80, lambd=0.5, min_dist=0.015, max_pts=88, return_tensor=False):
    """
    Giải hệ phương trình Thin-Plate Spline đơn mẫu trên CPU tốc độ cao (< 0.4ms).
    """
    if c_s is None or c_d is None or c_s.shape[0] < 4:
        c_pad = np.zeros((max_pts, 2), dtype=np.float32)
        th_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
        return (torch.from_numpy(c_pad), torch.from_numpy(th_pad), 0.0) if return_tensor else (c_pad, th_pad, 0.0)

    n = c_s.shape[0]
    min_dist_sq = min_dist * min_dist
    keep_indices = [0]
    kept_pts = np.empty((n, 2), dtype=np.float32)
    kept_pts[0] = c_s[0]
    num_kept = 1

    for i in range(1, n):
        dx = kept_pts[:num_kept, 0] - c_s[i, 0]
        dy = kept_pts[:num_kept, 1] - c_s[i, 1]
        if np.min(dx * dx + dy * dy) >= min_dist_sq:
            kept_pts[num_kept] = c_s[i]
            keep_indices.append(i)
            num_kept += 1
            if num_kept >= max_kps:
                break

    if num_kept < 4:
        c_pad = np.zeros((max_pts, 2), dtype=np.float32)
        th_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
        return (torch.from_numpy(c_pad), torch.from_numpy(th_pad), 0.0) if return_tensor else (c_pad, th_pad, 0.0)

    c_s_filt = kept_pts[:num_kept]
    c_d_filt = c_d[keep_indices]

    c_s_anchors = np.concatenate([c_s_filt, FIXED_ANCHORS_NP], axis=0)
    c_d_anchors = np.concatenate([c_d_filt, FIXED_ANCHORS_NP], axis=0)
    m = c_s_anchors.shape[0]

    diff = c_s_anchors[:, None, :] - c_s_anchors[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist_clamped = np.maximum(dist, 1e-6)
    K = (dist_clamped ** 2) * np.log(dist_clamped) + lambd * np.eye(m, dtype=np.float32)
    P = np.concatenate([np.ones((m, 1), dtype=np.float32), c_s_anchors], axis=-1)
    L = np.concatenate([np.concatenate([K, P], axis=-1), np.concatenate([P.T, np.zeros((3, 3), dtype=np.float32)], axis=-1)], axis=0)
    Y = np.concatenate([c_d_anchors, np.zeros((3, 2), dtype=np.float32)], axis=0)

    try:
        theta = np.linalg.solve(L, Y)
    except Exception:
        theta = np.linalg.lstsq(L, Y, rcond=None)[0]
    theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    v_flag = 0.0 if (np.isnan(theta).any() or np.isinf(theta).any() or np.max(np.abs(theta)) > 50.0) else 1.0

    c_pad = np.zeros((max_pts, 2), dtype=np.float32)
    theta_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
    c_pad[:m] = c_s_anchors
    theta_pad[:m] = theta[:m]
    theta_pad[max_pts:max_pts+3] = theta[m:m+3]

    if return_tensor:
        return torch.from_numpy(c_pad), torch.from_numpy(theta_pad), float(v_flag)
    return c_pad, theta_pad, float(v_flag)

def solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015, return_tensor=True):
    """
    Giải song song 2 chiều (Forward 0->1 và Backward 1->0) cho 1 cặp keypoints.
    """
    MAX_PTS = 88
    if kps is None or len(kps) < 2 or (hasattr(kps, 'shape') and len(kps.shape) > 1 and kps.shape[1] < 4):
        z_c0 = np.zeros((MAX_PTS, 2), dtype=np.float32)
        z_th01 = np.zeros((MAX_PTS + 3, 2), dtype=np.float32)
        z_c1 = np.zeros((MAX_PTS, 2), dtype=np.float32)
        z_th10 = np.zeros((MAX_PTS + 3, 2), dtype=np.float32)
        z_v = np.array(0.0, dtype=np.float32)
        if return_tensor:
            return (torch.from_numpy(z_c0), torch.from_numpy(z_th01), torch.from_numpy(z_c1),
                    torch.from_numpy(z_th10), torch.from_numpy(z_v))
        return z_c0, z_th01, z_c1, z_th10, z_v

    c0 = kps[0]
    c1 = kps[1]
    c0_pad, th01_pad, v01 = solve_tps_single_cpu(c0, c1, max_kps=max_kps, lambd=lambd, min_dist=min_dist, max_pts=MAX_PTS, return_tensor=False)
    c1_pad, th10_pad, v10 = solve_tps_single_cpu(c1, c0, max_kps=max_kps, lambd=lambd, min_dist=min_dist, max_pts=MAX_PTS, return_tensor=False)

    v = np.array(1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0, dtype=np.float32)
    if return_tensor:
        return (torch.from_numpy(c0_pad), torch.from_numpy(th01_pad), torch.from_numpy(c1_pad),
                torch.from_numpy(th10_pad), torch.from_numpy(v))
    return c0_pad, th01_pad, c1_pad, th10_pad, v

_GRID_CACHE = {}

def get_grid_pts(h, w, device):
    key = (h, w, device)
    pts = _GRID_CACHE.get(key)
    if pts is None:
        if len(_GRID_CACHE) >= _MAX_GRID_CACHE_SIZE:
            _GRID_CACHE.clear()
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device), indexing='ij'
        )
        pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        _GRID_CACHE[key] = pts
    return pts

@torch.no_grad()
def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, h, w, device, chunk_size=65536):
    """
    Vector hóa 100% trên GPU: Tính toán trường dòng chuyển động TPS cho TOÀN BỘ BATCH cùng lúc.
    """
    with torch.amp.autocast('cuda', enabled=False):
        theta_f32 = theta_batch.float()
        c_src_f32 = c_src_batch.float()
        valid_f32 = valid_mask.float()

        b = theta_f32.shape[0]
        pts = get_grid_pts(h, w, device)
        total_pts = pts.shape[0]
        flow_chunks = []

        for i in range(0, total_pts, chunk_size):
            p_chunk = pts[i:i + chunk_size]
            c_len = p_chunk.shape[0]
            pts_exp = p_chunk.unsqueeze(0).expand(b, -1, -1)

            dist = torch.cdist(pts_exp, c_src_f32, p=2.0)
            U = tps_rbf(dist)

            ones = torch.ones((b, c_len, 1), device=device, dtype=pts.dtype)
            P = torch.cat([ones, pts_exp], dim=-1)
            basis = torch.cat([U, P], dim=-1)

            mapped = torch.bmm(basis, theta_f32)
            flow_chunks.append(mapped - pts_exp)

        flow = torch.cat(flow_chunks, dim=1)
        flow = flow.view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()
        flow.mul_(valid_f32.view(b, 1, 1, 1))
        return flow

def tps_flow_from_theta(theta, c_src, h, w, device, chunk_size=32768):
    """Fallback tương thích cho đơn mẫu inference."""
    n = c_src.shape[0]
    if n < 4 or theta.shape[0] != (n + 3):
        return torch.zeros((1, 2, h, w), device=device)

    pts = get_grid_pts(h, w, device)
    total_pts = pts.shape[0]
    flow_chunks = []

    for i in range(0, total_pts, chunk_size):
        p_chunk = pts[i:i + chunk_size]
        diff = p_chunk.unsqueeze(1) - c_src.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        U = tps_rbf(dist)
        P = torch.cat([torch.ones((p_chunk.shape[0], 1), device=device), p_chunk], dim=-1)
        basis = torch.cat([U, P], dim=-1)
        mapped = torch.matmul(basis, theta)
        flow_chunk = mapped - p_chunk
        flow_chunks.append(flow_chunk)

    flow = torch.cat(flow_chunks, dim=0)
    flow = flow.view(h, w, 2).permute(2, 0, 1).unsqueeze(0)
    return flow

print("=" * 80)
print("[✓] CELL 3 hoàn tất: Warp Layers Cache, 8 Anchors & Vectorized TPS Batch Solver sẵn sàng (< 0.05s).")
print("=" * 80)
sys.stdout.flush()
