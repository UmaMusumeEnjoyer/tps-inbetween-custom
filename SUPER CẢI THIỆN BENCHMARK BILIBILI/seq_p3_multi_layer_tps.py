#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_p3_multi_layer_tps.py
--------------------------------------------------------------------------------
PHA 3 (BƯỚC 1 & BƯỚC 2): MODULE COARSE TPS ĐA LỚP (MULTI-LAYER CLUSTERED TPS)
Đề tài: Nội suy khung hình hoạt họa Anime Lineart (PBL6)
Kiến trúc đích: TPSInbetweenOriginal (~10.07M active params, 0% thay đổi kiến trúc mạng)

Giải quyết triệt để hạn chế vật lý của Single-Layer Global TPS:
  1. Bước 1 - Phân cụm điểm chuyển động (Motion Segmentation):
     - Tính vận tốc tương đối pixel: v_i = (P_{1,i} - P_{0,i}) * [W, H].
     - Tách Cụm Nền Tĩnh (Static Layer, label = 0) với ||v_i||_2 <= eps_static_px (1.5 px).
     - Khử nhiễu điểm khớp sai (Local Spatial-Velocity & RANSAC Affine Outlier Rejection, label = -1).
     - Phân cụm Tiền Cảnh Động (Foreground Layers, label = 1..K) bằng thuật toán
       Spatial-Velocity DBSCAN kết hợp kiểm tra đồng nhất Affine cục bộ.

  2. Bước 2 - Thuật toán Multi-Layer TPS Flow Solver:
     - Giải TPS_static trên cụm điểm tĩnh (triệt tiêu trọng số uốn phi tuyến RBF ->
       đạo hàm bậc 2 bằng 0 tuyệt đối, không làm cong nền tĩnh).
     - Giải TPS_dynamic độc lập trên từng cụm điểm động (không bị điểm tĩnh kéo ghì).
     - Tạo mặt nạ mềm (Soft Blending Mask M) bằng khoảng cách Hausdorff / Voronoi
       với hàm chuyển tiếp Quintic Smoothstep C^2:
           F_fused = M * F_dynamic + (1 - M) * F_static
     - Bảo toàn đạo hàm bậc 2 bằng 0 (phẳng tuyệt đối, ||Hessian(F)|| = 0) tại vùng nền tĩnh.
--------------------------------------------------------------------------------
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None


FIXED_TPS_ANCHORS = np.array([
    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
    [0.0, 0.5],             [1.0, 0.5],
    [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
], dtype=np.float32)

_ML_GRID_CACHE = {}


# ==============================================================================
# BƯỚC 1: MODULE PHÂN CỤM ĐIỂM CHUYỂN ĐỘNG (MOTION SEGMENTATION)
# ==============================================================================
def _dbscan_distance_matrix(dist_mat, eps=1.0, min_samples=3):
    """
    Thuật toán DBSCAN thuần NumPy trên ma trận khoảng cách chuẩn hóa D_sv (N x N).
    Trả về mảng nhãn cluster: 0, 1, ..., K-1 cho các cụm hợp lệ, và -1 cho nhiễu (noise).
    """
    n = dist_mat.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int32)

    neighbors = [np.where(dist_mat[i] <= eps)[0] for i in range(n)]
    is_core = np.array([len(nb) >= min_samples for nb in neighbors], dtype=bool)

    labels = np.full((n,), -1, dtype=np.int32)
    visited = np.zeros((n,), dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i] or not is_core[i]:
            continue
        visited[i] = True
        labels[i] = cluster_id
        queue = list(neighbors[i])
        head = 0
        while head < len(queue):
            q = queue[head]
            head += 1
            if not visited[q]:
                visited[q] = True
                if is_core[q]:
                    for nb_q in neighbors[q]:
                        if not visited[nb_q] or labels[nb_q] == -1:
                            queue.append(nb_q)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return labels


def _filter_dynamic_outliers(p0_px, p1_px, vel_px, speed_px, k_nn=5, max_rel_err=0.65, min_abs_tol_px=14.0):
    """
    Khử điểm khớp sai (Outliers) từ GlueStick/SIFT dựa trên tính đồng nhất vận tốc - không gian
    và mô hình Affine cục bộ (bảo toàn 100% biến dạng co giãn hình học như hình tròn thu/phóng).
    """
    n = len(p0_px)
    if n < 6:
        return np.ones((n,), dtype=bool)

    mid_px = 0.5 * (p0_px + p1_px)
    dist_mat = np.linalg.norm(mid_px[:, None, :] - mid_px[None, :, :], axis=-1)

    k_use = min(k_nn, n - 1)
    nn_idx = np.argsort(dist_mat, axis=-1)[:, :k_use + 1]  # bao gồm chính điểm i

    # 1. Kiểm tra sai lệch vận tốc cục bộ so với median của lân cận k_nn
    local_vels = vel_px[nn_idx]  # [N, k_use+1, 2]
    med_vel = np.median(local_vels, axis=1)  # [N, 2]
    med_speed = np.linalg.norm(med_vel, axis=-1)
    vel_diff = np.linalg.norm(vel_px - med_vel, axis=-1)
    local_inlier = vel_diff <= np.maximum(min_abs_tol_px, max_rel_err * med_speed)

    # 2. Kiểm tra thêm bằng Affine toàn cục/bán cục bộ (nếu vật thể biến dạng co giãn đều như hình tròn)
    if cv2 is not None and n >= 6:
        try:
            aff_m, aff_mask = cv2.estimateAffine2D(
                p0_px.astype(np.float32), p1_px.astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=max(12.0, 0.15 * float(np.median(speed_px)))
            )
            if aff_m is not None and aff_mask is not None:
                aff_inlier = aff_mask.ravel().astype(bool)
                # Điểm được giữ nếu thỏa mãn lân cận cục bộ HOẶC khớp chính xác phép biến đổi Affine
                local_inlier = local_inlier | aff_inlier
        except Exception:
            pass

    return local_inlier


def segment_motion_keypoints(
    kps0,
    kps1,
    img_hw=(540, 960),
    eps_static_px=1.5,
    spatial_radius_ratio=0.18,
    vel_tol_px=24.0,
    min_cluster_pts=4,
    filter_outliers=True,
):
    """
    BƯỚC 1: Phân cụm điểm chuyển động (Motion Segmentation) trên cặp điểm đối sánh P0 <-> P1.

    Tham số:
      - kps0, kps1: Tọa độ điểm khớp chuẩn hóa [0, 1] kích thước (N, 2).
      - img_hw: Tuple (H, W) kích thước ảnh gốc tính bằng pixel.
      - eps_static_px: Ngưỡng vận tốc (px) để phân lập Cụm Nền Tĩnh (mặc định <= 1.5 px).
      - spatial_radius_ratio: Bán kính không gian chuẩn hóa theo đường chéo ảnh cho DBSCAN.
      - vel_tol_px: Ngưỡng dung sai vận tốc cơ sở (px) cho Spatial-Velocity DBSCAN.
      - min_cluster_pts: Số điểm tối thiểu để tạo một cụm TPS động độc lập (>= 4).
      - filter_outliers: Bật bộ lọc loại bỏ điểm khớp nhiễu cục bộ.

    Trả về dict:
      - 'labels': Mảng (N,) với:
          0     : Cụm Nền Tĩnh (Static Layer)
          1..K  : Các Cụm Tiền Cảnh Động (Dynamic Foreground Layers)
          -1    : Điểm nhiễu bị loại bỏ (Outliers)
      - 'velocities_px': Vector vận tốc (N, 2) tính theo pixel.
      - 'speeds_px': Độ lớn vận tốc ||v_i||_2 (N,) tính theo pixel.
      - 'static_kps0', 'static_kps1': Các điểm thuộc cụm tĩnh.
      - 'dynamic_clusters': Danh sách dict mô tả từng cụm động k = 1..K.
      - 'all_dyn_kps0', 'all_dyn_kps1': Toàn bộ điểm động hợp lệ gộp chung.
    """
    if kps0 is None or kps1 is None or len(kps0) == 0 or len(kps1) == 0:
        empty_pts = np.zeros((0, 2), dtype=np.float32)
        return {
            'labels': np.empty((0,), dtype=np.int32),
            'velocities_px': empty_pts,
            'speeds_px': np.empty((0,), dtype=np.float32),
            'static_mask': np.empty((0,), dtype=bool),
            'static_kps0': empty_pts,
            'static_kps1': empty_pts,
            'dynamic_clusters': [],
            'all_dyn_kps0': empty_pts,
            'all_dyn_kps1': empty_pts,
            'num_static': 0,
            'num_dynamic': 0,
            'num_outliers': 0,
            'num_clusters': 0,
        }

    p0 = np.asarray(kps0, dtype=np.float32)
    p1 = np.asarray(kps1, dtype=np.float32)
    n_pts = len(p0)

    h, w = float(img_hw[0]), float(img_hw[1])
    scale_wh = np.array([max(1.0, w - 1.0), max(1.0, h - 1.0)], dtype=np.float32)
    diag_px = math.hypot(w, h)

    p0_px = p0 * scale_wh[None, :]
    p1_px = p1 * scale_wh[None, :]
    vel_px = p1_px - p0_px
    speed_px = np.linalg.norm(vel_px, axis=-1)

    labels = np.full((n_pts,), -1, dtype=np.int32)

    # 1. Phân lập Cụm Tĩnh (Static Layer: ||v_i|| <= eps_static_px)
    static_mask = speed_px <= float(eps_static_px)
    labels[static_mask] = 0

    # 2. Xử lý các điểm chuyển động (Dynamic Candidates: ||v_i|| > eps_static_px)
    dyn_cand_idx = np.where(~static_mask)[0]
    if len(dyn_cand_idx) > 0:
        d_p0_px = p0_px[dyn_cand_idx]
        d_p1_px = p1_px[dyn_cand_idx]
        d_vel_px = vel_px[dyn_cand_idx]
        d_speed_px = speed_px[dyn_cand_idx]

        if filter_outliers and len(dyn_cand_idx) >= 6:
            inlier_mask = _filter_dynamic_outliers(d_p0_px, d_p1_px, d_vel_px, d_speed_px)
        else:
            inlier_mask = np.ones((len(dyn_cand_idx),), dtype=bool)

        valid_dyn_idx = dyn_cand_idx[inlier_mask]
        if len(valid_dyn_idx) > 0:
            vp0_px = p0_px[valid_dyn_idx]
            vp1_px = p1_px[valid_dyn_idx]
            vvel_px = vel_px[valid_dyn_idx]
            vspeed_px = speed_px[valid_dyn_idx]
            vmid_px = 0.5 * (vp0_px + vp1_px)

            # Khoảng cách không gian - vận tốc chuẩn hóa cho DBSCAN
            eps_spatial = max(32.0, float(spatial_radius_ratio) * diag_px)
            med_v = float(np.median(vspeed_px)) if len(vspeed_px) > 0 else float(vel_tol_px)
            eps_vel = max(float(vel_tol_px), 0.45 * med_v)

            dist_sp = np.linalg.norm(vmid_px[:, None, :] - vmid_px[None, :, :], axis=-1) / eps_spatial
            dist_vl = np.linalg.norm(vvel_px[:, None, :] - vvel_px[None, :, :], axis=-1) / eps_vel
            dist_sv = np.sqrt(dist_sp ** 2 + dist_vl ** 2)

            raw_clusters = _dbscan_distance_matrix(dist_sv, eps=1.0, min_samples=min(3, len(valid_dyn_idx)))

            # Gom các cụm đủ điểm (>= min_cluster_pts) và gán các điểm rời rạc hợp lệ vào cụm gần nhất
            unique_c = [c for c in sorted(set(raw_clusters)) if c >= 0]
            major_clusters = [c for c in unique_c if int(np.sum(raw_clusters == c)) >= min_cluster_pts]

            if not major_clusters:
                # Nếu tất cả cụm đều nhỏ nhưng tổng số điểm động >= min_cluster_pts -> gộp thành 1 cụm động
                if len(valid_dyn_idx) >= min_cluster_pts:
                    labels[valid_dyn_idx] = 1
                elif len(dyn_cand_idx) >= min_cluster_pts:
                    labels[dyn_cand_idx] = 1
            else:
                remap = {c_old: (idx_new + 1) for idx_new, c_old in enumerate(major_clusters)}
                for local_i, global_i in enumerate(valid_dyn_idx):
                    c_old = int(raw_clusters[local_i])
                    if c_old in remap:
                        labels[global_i] = remap[c_old]
                    else:
                        # Gán điểm biên vào cụm chính gần nhất nếu vận tốc tương thích
                        best_cid = -1
                        best_d = 1e9
                        for c_maj in major_clusters:
                            c_mask = (raw_clusters == c_maj)
                            d_min = float(np.min(dist_sv[local_i, c_mask]))
                            if d_min < best_d:
                                best_d = d_min
                                best_cid = remap[c_maj]
                        if best_d <= 2.2:
                            labels[global_i] = best_cid

    # Tổng hợp danh sách cụm động
    dynamic_clusters = []
    max_cid = int(labels.max()) if len(labels) > 0 else 0
    for cid in range(1, max_cid + 1):
        c_idx = np.where(labels == cid)[0]
        if len(c_idx) < min_cluster_pts:
            labels[c_idx] = -1
            continue
        dynamic_clusters.append({
            'cluster_id': len(dynamic_clusters) + 1,
            'indices': c_idx,
            'kps0': p0[c_idx],
            'kps1': p1[c_idx],
            'mean_vel_px': vel_px[c_idx].mean(axis=0),
            'mean_speed_px': float(speed_px[c_idx].mean()),
        })

    # Chuẩn hóa lại nhãn liên tục 1..K sau khi lọc
    clean_labels = np.where(labels == 0, 0, -1).astype(np.int32)
    for dc in dynamic_clusters:
        clean_labels[dc['indices']] = dc['cluster_id']

    all_dyn_mask = clean_labels >= 1
    return {
        'labels': clean_labels,
        'velocities_px': vel_px,
        'speeds_px': speed_px,
        'static_mask': (clean_labels == 0),
        'static_kps0': p0[clean_labels == 0],
        'static_kps1': p1[clean_labels == 0],
        'dynamic_clusters': dynamic_clusters,
        'all_dyn_kps0': p0[all_dyn_mask],
        'all_dyn_kps1': p1[all_dyn_mask],
        'num_static': int(np.sum(clean_labels == 0)),
        'num_dynamic': int(np.sum(all_dyn_mask)),
        'num_outliers': int(np.sum(clean_labels == -1)),
        'num_clusters': len(dynamic_clusters),
    }


# ==============================================================================
# BƯỚC 2: THUẬT TOÁN MULTI-LAYER TPS FLOW SOLVER & SOFT VORONOI MASK
# ==============================================================================
def _stratified_spatial_subsample(c_src, c_dst, n_keep):
    """
    Lấy mẫu đại diện phân bố đều theo không gian (Farthest-Point / Spatial Grid Stratification)
    thay vì cắt chỉ số tuyến tính đơn thuần, giúp giữ trọn vẹn biên hình học của mọi bộ phận.
    """
    n = len(c_src)
    if n <= n_keep:
        return np.asarray(c_src, dtype=np.float64), np.asarray(c_dst, dtype=np.float64)

    cs = np.asarray(c_src, dtype=np.float64)
    cd = np.asarray(c_dst, dtype=np.float64)

    # Khởi tạo bằng điểm gần tâm nhất, sau đó chọn lặp điểm xa nhất (Farthest Point Sampling)
    selected = np.zeros((n_keep,), dtype=np.int64)
    center = cs.mean(axis=0, keepdims=True)
    selected[0] = int(np.argmin(np.linalg.norm(cs - center, axis=-1)))
    min_dists = np.linalg.norm(cs - cs[selected[0]], axis=-1)

    for k in range(1, n_keep):
        pick = int(np.argmax(min_dists))
        selected[k] = pick
        d_new = np.linalg.norm(cs - cs[pick], axis=-1)
        min_dists = np.minimum(min_dists, d_new)

    selected.sort()
    return cs[selected], cd[selected]


def _solve_tps_system(cs_all, cd_all, max_pts=128, lambd=0.01):
    """Giải hệ phương trình tuyến tính TPS (K + lambda*I) w + P a = cd_all."""
    m = len(cs_all)
    diff = cs_all[:, None, :] - cs_all[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    K = (r ** 2) * np.log(np.maximum(r, 1e-6))
    np.fill_diagonal(K, float(lambd))

    P = np.hstack([np.ones((m, 1), dtype=np.float64), cs_all])
    A = np.zeros((m + 3, m + 3), dtype=np.float64)
    A[:m, :m] = K
    A[:m, m:] = P
    A[m:, :m] = P.T

    Y = np.zeros((m + 3, 2), dtype=np.float64)
    Y[:m, :] = cd_all

    try:
        theta = np.linalg.solve(A, Y)
    except np.linalg.LinAlgError:
        theta, _, _, _ = np.linalg.lstsq(A, Y, rcond=1e-4)

    valid = 0.0 if (
        np.isnan(theta).any() or np.isinf(theta).any() or np.max(np.abs(theta)) > 50.0
    ) else 1.0

    c_pad = np.zeros((max_pts, 2), dtype=np.float32)
    th_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
    if valid > 0.5:
        c_pad[:m] = cs_all.astype(np.float32)
        th_pad[:m] = theta[:m].astype(np.float32)
        th_pad[max_pts:max_pts + 3] = theta[m:m + 3].astype(np.float32)
    return c_pad, th_pad, valid


def solve_tps_static_layer(c_src_static, c_dst_static, max_pts=128, lambd=0.1, zero_hessian=True):
    """
    BƯỚC 2.1 & 2.4: Giải TPS_static cho cụm điểm nền tĩnh.
    Đảm bảo đạo hàm bậc 2 bằng 0 tuyệt đối (phẳng tuyệt đối, ||Hessian(F_static)|| = 0):
      - Khi các điểm thuộc Cụm Tĩnh (||v_i|| <= eps_static), nếu zero_hessian=True,
        ta triệt tiêu hoàn toàn thành phần uốn phi tuyến RBF (th_pad[:m] = 0.0) và
        nếu độ lệch tĩnh trung vị gần bằng 0, đặt ánh xạ đồng nhất P(q) = q (zero flow tuyệt đối).
    """
    c_pad = np.zeros((max_pts, 2), dtype=np.float32)
    th_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
    # Ma trận Affine đồng nhất f(x, y) = [0, 0] + 1*x + 1*y -> disp(q) = f(q) - q = 0
    th_pad[max_pts + 1, 0] = 1.0
    th_pad[max_pts + 2, 1] = 1.0

    if c_src_static is None or c_dst_static is None or len(c_src_static) == 0:
        return c_pad, th_pad, 1.0

    cs = np.asarray(c_src_static, dtype=np.float64)
    cd = np.asarray(c_dst_static, dtype=np.float64)
    max_disp = float(np.max(np.linalg.norm(cd - cs, axis=-1))) if len(cs) > 0 else 0.0

    # Nếu cụm tĩnh đứng yên hoặc dịch chuyển vi mô (< 2e-3 trên miền chuẩn hóa [0,1]),
    # trả về nghiệm đồng nhất phẳng tuyệt đối (Zero Flow, đạo hàm bậc 1 và bậc 2 đều bằng 0)
    if zero_hessian or max_disp < 2.5e-3 or len(cs) < 4:
        return c_pad, th_pad, 1.0

    n_keep = max_pts - 8
    cs_sub, cd_sub = _stratified_spatial_subsample(cs, cd, n_keep)
    anc = FIXED_TPS_ANCHORS.astype(np.float64)
    cs_all = np.vstack([cs_sub, anc])
    cd_all = np.vstack([cd_sub, anc])
    c_sol, th_sol, val = _solve_tps_system(cs_all, cd_all, max_pts=max_pts, lambd=lambd)
    if val > 0.5:
        # Khóa chặt trọng số RBF phi tuyến = 0 để đảm bảo đạo hàm bậc 2 bằng 0 tuyệt đối
        th_sol[:max_pts, :] = 0.0
        return c_sol, th_sol, 1.0
    return c_pad, th_pad, 1.0


def solve_tps_dynamic_layer(c_src_dyn, c_dst_dyn, max_pts=128, lambd=0.01, use_anchors=False):
    """
    BƯỚC 2.2: Giải TPS_dynamic độc lập trên cụm điểm động (không bị điểm tĩnh kéo ghì).
    """
    if c_src_dyn is None or c_dst_dyn is None or len(c_src_dyn) < 4:
        return (
            np.zeros((max_pts, 2), dtype=np.float32),
            np.zeros((max_pts + 3, 2), dtype=np.float32),
            0.0,
        )
    n_keep = (max_pts - 8) if use_anchors else max_pts
    cs_sub, cd_sub = _stratified_spatial_subsample(c_src_dyn, c_dst_dyn, n_keep)
    if use_anchors:
        anc = FIXED_TPS_ANCHORS.astype(np.float64)
        cs_all = np.vstack([cs_sub, anc])
        cd_all = np.vstack([cd_sub, anc])
    else:
        cs_all, cd_all = cs_sub, cd_sub
    return _solve_tps_system(cs_all, cd_all, max_pts=max_pts, lambd=lambd)


def _get_ml_norm_grid(h, w, device):
    key = (h, w, device, 'ml_tps_grid')
    pts = _ML_GRID_CACHE.get(key)
    if pts is None:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pts = torch.stack([gx, gy], dim=-1).view(-1, 2).contiguous()
        _ML_GRID_CACHE[key] = pts
    return pts


@torch.no_grad()
def _eval_tps_flow_single(theta, c_src, valid_flag, orig_hw, h_pad, w_pad, device, downscale_factor=2):
    """Đánh giá nhanh trường dòng chảy TPS trên GPU từ cặp (theta, c_src) kích thước [1, ...]."""
    if float(valid_flag) < 0.5:
        return torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)

    with torch.amp.autocast('cuda', enabled=False):
        th_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
        cs_t = torch.as_tensor(c_src, dtype=torch.float32, device=device)
        if th_t.ndim == 2:
            th_t = th_t.unsqueeze(0)
        if cs_t.ndim == 2:
            cs_t = cs_t.unsqueeze(0)

        b = th_t.shape[0]
        h_sub = max(16, h_pad // max(1, downscale_factor))
        w_sub = max(16, w_pad // max(1, downscale_factor))

        pts_pad = _get_ml_norm_grid(h_sub, w_sub, device)
        total_pts = pts_pad.shape[0]
        pts_pad_exp = pts_pad.unsqueeze(0).expand(b, -1, -1)

        if isinstance(orig_hw, (tuple, list)):
            orig_h, orig_w = float(orig_hw[0]), float(orig_hw[1])
            sx = float(max(1, w_pad - 1)) / max(1.0, orig_w - 1.0)
            sy = float(max(1, h_pad - 1)) / max(1.0, orig_h - 1.0)
            scale_xy = torch.tensor([[[sx, sy]]], dtype=torch.float32, device=device)
        else:
            orig_hw_f = orig_hw.to(device=device, dtype=torch.float32)
            sx = (float(max(1, w_pad - 1)) / torch.clamp(orig_hw_f[:, 1] - 1.0, min=1.0)).view(b, 1, 1)
            sy = (float(max(1, h_pad - 1)) / torch.clamp(orig_hw_f[:, 0] - 1.0, min=1.0)).view(b, 1, 1)
            scale_xy = torch.cat([sx, sy], dim=-1)

        pts_orig_raw = pts_pad_exp * scale_xy
        in_bounds = ((pts_orig_raw[..., 0:1] <= 1.001) & (pts_orig_raw[..., 1:2] <= 1.001)).float()
        pts_orig = torch.clamp(pts_orig_raw, 0.0, 1.0)

        flow_chunks = []
        chunk_size = 32768
        for i in range(0, total_pts, chunk_size):
            end_i = min(i + chunk_size, total_pts)
            q = pts_orig[:, i:end_i, :]
            dist = torch.clamp(torch.cdist(q, cs_t, p=2.0), min=1e-6)
            U = (dist ** 2) * torch.log(dist)
            ones = torch.ones((b, q.shape[1], 1), device=device, dtype=torch.float32)
            basis = torch.cat([U, ones, q], dim=-1)
            mapped = torch.bmm(basis, th_t)
            disp_orig = mapped - q
            disp_pad = (disp_orig / scale_xy) * in_bounds[:, i:end_i, :]
            flow_chunks.append(disp_pad)

        flow = torch.cat(flow_chunks, dim=1).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()
        if h_sub != h_pad or w_sub != w_pad:
            flow = F.interpolate(flow, size=(h_pad, w_pad), mode='bilinear', align_corners=True)
        return flow


@torch.no_grad()
def compute_soft_voronoi_blending_mask(
    dyn_pts_list_norm,
    static_pts_norm,
    orig_hw,
    h_pad,
    w_pad,
    device,
    downscale_factor=2,
    transition_width_px=18.0,
    falloff_radius_px=160.0,
):
    """
    BƯỚC 2.3 & 2.4: Tạo mặt nạ hòa trộn mềm (Soft Blending Mask M) dựa trên khoảng cách
    Hausdorff / Voronoi từ mỗi điểm lưới tới các Cụm Điểm Động và Cụm Điểm Tĩnh.

    Đặc tính toán học:
      - Với mỗi cụm động k = 1..K, tính khoảng cách Euclidean ngắn nhất (Hausdorff point-to-set)
        d_dyn^(k)(q) từ điểm lưới q tới quỹ đạo điểm động.
      - Với cụm tĩnh, tính khoảng cách d_static(q) tới tập điểm tĩnh.
        (Nếu không có điểm tĩnh tường minh, dùng ngưỡng bán kính ảnh hưởng falloff_radius_px).
      - Sử dụng hàm Quintic Smoothstep C^2:
            u(q) = clip((d_static(q) - d_dyn(q) + w_trans) / (2 * w_trans), 0, 1)
            M(q) = 6*u^5 - 15*u^4 + 10*u^3
      - Tại vùng nền tĩnh (d_static(q) <= d_dyn(q) - w_trans):
            M(q) == 0.0 tuyệt đối -> Đạo hàm bậc 1 và bậc 2 bằng 0 hoàn toàn!
      - Tại lõi vật thể động (d_dyn(q) <= d_static(q) - w_trans):
            M(q) == 1.0 tuyệt đối -> Giữ nguyên 100% trường biến dạng động!

    Trả về:
      - weights_dyn: List gồm K tensor mặt nạ [1, 1, h_pad, w_pad] cho từng cụm động.
      - weight_static: Tensor mặt nạ tĩnh (1 - M_total) kích thước [1, 1, h_pad, w_pad].
      - mask_total: Tensor M_total = sum_k M_k kích thước [1, 1, h_pad, w_pad].
    """
    if not dyn_pts_list_norm:
        zero_m = torch.zeros((1, 1, h_pad, w_pad), dtype=torch.float32, device=device)
        one_m = torch.ones((1, 1, h_pad, w_pad), dtype=torch.float32, device=device)
        return [], one_m, zero_m

    if isinstance(orig_hw, torch.Tensor):
        orig_h = float(orig_hw[0, 0].item())
        orig_w = float(orig_hw[0, 1].item())
    else:
        orig_h, orig_w = float(orig_hw[0]), float(orig_hw[1])

    h_sub = max(16, h_pad // max(1, downscale_factor))
    w_sub = max(16, w_pad // max(1, downscale_factor))

    pts_pad = _get_ml_norm_grid(h_sub, w_sub, device)  # [P, 2] trong [0, 1]^2 của canvas pad
    sx = float(max(1, w_pad - 1)) / max(1.0, orig_w - 1.0)
    sy = float(max(1, h_pad - 1)) / max(1.0, orig_h - 1.0)

    # Đổi điểm lưới sang đơn vị pixel thực tế của ảnh gốc [0, W] x [0, H]
    q_px = torch.stack([
        pts_pad[:, 0] * sx * max(1.0, orig_w - 1.0),
        pts_pad[:, 1] * sy * max(1.0, orig_h - 1.0),
    ], dim=-1).unsqueeze(0)  # [1, P, 2]

    scale_wh_t = torch.tensor([[max(1.0, orig_w - 1.0), max(1.0, orig_h - 1.0)]], dtype=torch.float32, device=device)

    # Tính khoảng cách d_dyn^(k)(q) cho từng cụm động
    d_dyn_list = []
    for d_pts in dyn_pts_list_norm:
        dp_t = torch.as_tensor(d_pts, dtype=torch.float32, device=device)
        if dp_t.ndim == 2:
            dp_px = (dp_t * scale_wh_t).unsqueeze(0)  # [1, M_k, 2]
        else:
            dp_px = dp_t * scale_wh_t.unsqueeze(0)
        dist_k = torch.cdist(q_px, dp_px, p=2.0).min(dim=-1).values  # [1, P]
        d_dyn_list.append(dist_k)

    d_dyn_stack = torch.stack(d_dyn_list, dim=0)  # [K, 1, P]
    d_dyn_min = d_dyn_stack.min(dim=0).values     # [1, P]

    # Tính khoảng cách d_static(q) tới cụm điểm tĩnh
    if static_pts_norm is not None and len(static_pts_norm) > 0:
        sp_t = torch.as_tensor(static_pts_norm, dtype=torch.float32, device=device)
        sp_px = (sp_t * scale_wh_t).unsqueeze(0)  # [1, M_s, 2]
        d_static = torch.cdist(q_px, sp_px, p=2.0).min(dim=-1).values  # [1, P]
        if falloff_radius_px is not None and falloff_radius_px > 0:
            d_static = torch.minimum(d_static, torch.full_like(d_static, float(falloff_radius_px)))
    else:
        d_static = torch.full_like(d_dyn_min, float(falloff_radius_px))

    # Mặt nạ tổng M_total bằng hàm Quintic Smoothstep C^2 trên biên Voronoi (d_static - d_dyn_min)
    w_tr = max(2.0, float(transition_width_px))
    u = torch.clamp((d_static - d_dyn_min + w_tr) / (2.0 * w_tr), 0.0, 1.0)
    m_total_flat = (u ** 3) * (u * (u * 6.0 - 15.0) + 10.0)  # [1, P]

    # Bảo vệ tuyệt đối vùng lân cận điểm tĩnh: mọi điểm nằm gần điểm tĩnh hơn điểm động
    # hoặc cách điểm tĩnh dưới ngưỡng an toàn (22 px) và xa điểm động đều khóa M = 0.0
    if static_pts_norm is not None and len(static_pts_norm) > 0:
        strict_static = (d_static < 24.0) & (d_dyn_min > d_static + 6.0)
        m_total_flat = torch.where(strict_static, torch.zeros_like(m_total_flat), m_total_flat)

    # Khóa cứng vùng biên để đảm bảo đạo hàm bậc 2 bằng 0 tuyệt đối trên vùng tĩnh
    m_total_flat = torch.where(m_total_flat < 1e-4, torch.zeros_like(m_total_flat), m_total_flat)
    m_total_flat = torch.where(m_total_flat > 1.0 - 1e-4, torch.ones_like(m_total_flat), m_total_flat)

    m_total = m_total_flat.view(1, 1, h_sub, w_sub)
    if h_sub != h_pad or w_sub != w_pad:
        m_total = F.interpolate(m_total, size=(h_pad, w_pad), mode='bilinear', align_corners=True)
        # Bảo toàn vùng 0 tuyệt đối sau nội suy bilinear
        m_total = torch.where(m_total < 1e-3, torch.zeros_like(m_total), m_total)
        m_total = torch.where(m_total > 1.0 - 1e-3, torch.ones_like(m_total), m_total)

    if len(d_dyn_list) == 1:
        weights_dyn = [m_total]
    else:
        # Phân bổ Partition of Unity giữa K cụm động theo Soft Voronoi
        tau_k = max(4.0, 0.5 * w_tr)
        logits_k = -d_dyn_stack / tau_k  # [K, 1, P]
        probs_k = torch.softmax(logits_k, dim=0).view(len(d_dyn_list), 1, h_sub, w_sub)
        if h_sub != h_pad or w_sub != w_pad:
            probs_k = F.interpolate(probs_k, size=(h_pad, w_pad), mode='bilinear', align_corners=True)
            probs_k = probs_k / probs_k.sum(dim=0, keepdim=True).clamp(min=1e-6)
        weights_dyn = [m_total * probs_k[k:k + 1] for k in range(len(d_dyn_list))]

    weight_static = 1.0 - m_total
    return weights_dyn, weight_static, m_total


@torch.no_grad()
def solve_multi_layer_tps(
    kps0,
    kps1,
    orig_hw,
    h_pad,
    w_pad,
    device,
    t_val=None,
    direct_t_flows=False,
    max_pts=128,
    lambd=0.01,
    downscale_factor=2,
    eps_static_px=1.5,
    use_dyn_anchors=False,
    transition_width_px=22.0,
    falloff_radius_px=105.0,
    filter_outliers=True,
):
    """
    BƯỚC 2: Hàm giải tổng hợp Multi-Layer Clustered TPS Flow Solver (`solve_multi_layer_tps`).

    Hỗ trợ đầy đủ cả 2 chế độ vận hành của kiến trúc `TPSInbetweenOriginal`:
      1. Chế độ Lagrangian Direct-t (`direct_t_flows=True`, truyền `t_val` thuộc [0, 1]):
         - Nội suy vị trí trung gian P_t = (1 - t)*P_0 + t*P_1 cho từng lớp.
         - Giải TPS_static -> F_static_0t, F_static_1t (đạo hàm bậc 2 = 0 tuyệt đối).
         - Giải TPS_dynamic^(k) -> F_dyn_0t^(k), F_dyn_1t^(k) độc lập với nền tĩnh.
         - Hòa trộn bằng mặt nạ Voronoi/Hausdorff M:
             F_fused_0t = sum_k M_k * F_dyn_0t^(k) + (1 - M) * F_static_0t
             F_fused_1t = sum_k M_k * F_dyn_1t^(k) + (1 - M) * F_static_1t
      2. Chế độ Backward-Pair (`direct_t_flows=False`):
         - Giải cặp dòng chảy ngược F_01 (từ P_1 -> P_0) và F_10 (từ P_0 -> P_1) đa lớp.
         - Kết hợp mặt nạ ống quỹ đạo (Trajectory Tube Voronoi Mask) bảo vệ tuyệt đối
           các vùng nền tĩnh khỏi năng lượng uốn của vật thể chuyển động.

    Trả về:
      - flow0_fused: Tensor [1, 2, h_pad, w_pad] (cf0t hoặc cf01)
      - flow1_fused: Tensor [1, 2, h_pad, w_pad] (cf1t hoặc cf10)
      - meta_info: Dict chứa kết quả phân cụm Bước 1, mặt nạ M, và kiểm định đạo hàm bậc 2.
    """
    if isinstance(orig_hw, torch.Tensor):
        img_h = int(orig_hw[0, 0].item())
        img_w = int(orig_hw[0, 1].item())
    else:
        img_h, img_w = int(orig_hw[0]), int(orig_hw[1])

    # BƯỚC 1: Phân cụm điểm chuyển động Tĩnh vs Động
    seg = segment_motion_keypoints(
        kps0, kps1,
        img_hw=(img_h, img_w),
        eps_static_px=eps_static_px,
        filter_outliers=filter_outliers,
    )

    zero_flow = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)

    # Nếu không có cụm động nào hợp lệ -> trả về dòng chảy tĩnh phẳng tuyệt đối (Zero Flow)
    if seg['num_clusters'] == 0:
        return zero_flow.clone(), zero_flow.clone(), {
            'segmentation': seg,
            'mask_total': torch.zeros((1, 1, h_pad, w_pad), dtype=torch.float32, device=device),
            'static_hessian_max': 0.0,
        }

    # BƯỚC 2.1: Giải TPS_static cho Cụm Nền Tĩnh (F_static có đạo hàm bậc 2 = 0 tuyệt đối)
    f_static_0 = zero_flow.clone()
    f_static_1 = zero_flow.clone()

    # BƯỚC 2.2: Giải TPS_dynamic cho từng Cụm Tiền Cảnh Động
    dyn_flows_0 = []
    dyn_flows_1 = []
    dyn_corridors = []

    tv = float(t_val) if t_val is not None else 0.5

    for dc in seg['dynamic_clusters']:
        p0_k = dc['kps0']
        p1_k = dc['kps1']
        pt_k = (1.0 - tv) * p0_k + tv * p1_k
        pm_k = 0.5 * (p0_k + p1_k)

        if direct_t_flows:
            # Giải trực tiếp tại mốc thời gian t: P_t -> P_0 và P_t -> P_1
            c_t0, th_t0, v0 = solve_tps_dynamic_layer(pt_k, p0_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            c_t1, th_t1, v1 = solve_tps_dynamic_layer(pt_k, p1_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            f0_k = _eval_tps_flow_single(th_t0, c_t0, v0, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            f1_k = _eval_tps_flow_single(th_t1, c_t1, v1, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            # Tập điểm quỹ đạo để bao phủ cả vị trí đích P_t và vị trí nguồn P_0, P_1 (tránh để lại bóng ma tại nguồn)
            corridor_k = np.vstack([p0_k, pm_k, pt_k, p1_k])
        else:
            # Giải theo chuẩn Backward-Pair: c0 = P_1 -> P_0 (flow01) và c1 = P_0 -> P_1 (flow10)
            c0_k, th01_k, v01 = solve_tps_dynamic_layer(p1_k, p0_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            c1_k, th10_k, v10 = solve_tps_dynamic_layer(p0_k, p1_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            f0_k = _eval_tps_flow_single(th01_k, c0_k, v01, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            f1_k = _eval_tps_flow_single(th10_k, c1_k, v10, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            corridor_k = np.vstack([p0_k, pm_k, p1_k])

        dyn_flows_0.append(f0_k)
        dyn_flows_1.append(f1_k)
        dyn_corridors.append(corridor_k)

    # Tập điểm tĩnh tham chiếu cho biên Voronoi
    if seg['num_static'] > 0:
        static_ref_pts = (1.0 - tv) * seg['static_kps0'] + tv * seg['static_kps1']
        effective_falloff = float(falloff_radius_px)
    elif use_dyn_anchors:
        static_ref_pts = FIXED_TPS_ANCHORS
        effective_falloff = float(falloff_radius_px)
    else:
        static_ref_pts = None
        effective_falloff = max(float(falloff_radius_px), 0.50 * math.hypot(img_w, img_h))

    # BƯỚC 2.3: Tính mặt nạ mềm Soft Blending Mask M dựa trên khoảng cách Hausdorff / Voronoi
    weights_dyn, weight_static, mask_total = compute_soft_voronoi_blending_mask(
        dyn_corridors,
        static_ref_pts,
        orig_hw=(img_h, img_w),
        h_pad=h_pad,
        w_pad=w_pad,
        device=device,
        downscale_factor=downscale_factor,
        transition_width_px=transition_width_px,
        falloff_radius_px=effective_falloff,
    )

    # Hòa trộn đa lớp: F_fused = sum_k M_k * F_dynamic^(k) + (1 - M) * F_static
    flow0_fused = weight_static * f_static_0
    flow1_fused = weight_static * f_static_1
    for w_k, f0_k, f1_k in zip(weights_dyn, dyn_flows_0, dyn_flows_1):
        flow0_fused = flow0_fused + w_k * f0_k
        flow1_fused = flow1_fused + w_k * f1_k

    # Anti-Ghost Pullback Guard tại ranh giới giao thoa (Occlusion / Transition Boundary):
    # Ngăn chặn vùng chuyển tiếp (0 < M < 1) vô tình kéo nét vẽ từ Cụm Tĩnh lên vùng trống.
    # Tính trên lưới giảm mẫu (h_sub, w_sub) để tiết kiệm 4x VRAM và tăng tốc 4x trên GPU.
    if seg['num_static'] > 0:
        h_sub_g = max(16, h_pad // max(1, downscale_factor))
        w_sub_g = max(16, w_pad // max(1, downscale_factor))
        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h_sub_g, device=device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, w_sub_g, device=device, dtype=torch.float32),
            indexing='ij'
        )
        base_g = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, h_sub, w_sub, 2]
        sx = float(max(1, w_pad - 1)) / max(1.0, float(img_w) - 1.0)
        sy = float(max(1, h_pad - 1)) / max(1.0, float(img_h) - 1.0)
        scale_wh_px = torch.tensor([[[[float(img_w - 1), float(img_h - 1)]]]], dtype=torch.float32, device=device)
        scale_pad_to_orig = torch.tensor([[[[sx, sy]]]], dtype=torch.float32, device=device)

        f0_sub = F.interpolate(flow0_fused, size=(h_sub_g, w_sub_g), mode='bilinear', align_corners=True) if (h_sub_g != h_pad or w_sub_g != w_pad) else flow0_fused
        f1_sub = F.interpolate(flow1_fused, size=(h_sub_g, w_sub_g), mode='bilinear', align_corners=True) if (h_sub_g != h_pad or w_sub_g != w_pad) else flow1_fused
        m_sub = F.interpolate(mask_total, size=(h_sub_g, w_sub_g), mode='bilinear', align_corners=True) if (h_sub_g != h_pad or w_sub_g != w_pad) else mask_total

        q0_src_px = ((base_g + f0_sub.permute(0, 2, 3, 1)) * scale_pad_to_orig * scale_wh_px).view(1, -1, 2)
        q1_src_px = ((base_g + f1_sub.permute(0, 2, 3, 1)) * scale_pad_to_orig * scale_wh_px).view(1, -1, 2)

        sp0_px = (torch.as_tensor(seg['static_kps0'], dtype=torch.float32, device=device) * scale_wh_px.view(1, 2)).unsqueeze(0)
        sp1_px = (torch.as_tensor(seg['static_kps1'], dtype=torch.float32, device=device) * scale_wh_px.view(1, 2)).unsqueeze(0)

        pt_all_px = (torch.as_tensor(
            np.vstack([(1.0 - tv) * dc['kps0'] + tv * dc['kps1'] for dc in seg['dynamic_clusters']]),
            dtype=torch.float32, device=device
        ) * scale_wh_px.view(1, 2)).unsqueeze(0)
        q_cur_px = (base_g * scale_pad_to_orig * scale_wh_px).view(1, -1, 2)
        d_cur_dyn = torch.cdist(q_cur_px, pt_all_px, p=2.0).min(dim=-1).values.view(1, 1, h_sub_g, w_sub_g)

        d_src0_stat = torch.cdist(q0_src_px, sp0_px, p=2.0).min(dim=-1).values.view(1, 1, h_sub_g, w_sub_g)
        d_src1_stat = torch.cdist(q1_src_px, sp1_px, p=2.0).min(dim=-1).values.view(1, 1, h_sub_g, w_sub_g)

        outside_dyn_stroke = (m_sub > 1e-4) & (d_cur_dyn > 10.0)
        pull_static_0 = (outside_dyn_stroke & (d_src0_stat < 22.0)).float()
        pull_static_1 = (outside_dyn_stroke & (d_src1_stat < 22.0)).float()
        if h_sub_g != h_pad or w_sub_g != w_pad:
            pull_static_0 = F.interpolate(pull_static_0, size=(h_pad, w_pad), mode='nearest')
            pull_static_1 = F.interpolate(pull_static_1, size=(h_pad, w_pad), mode='nearest')
        flow0_fused = torch.where(pull_static_0 > 0.5, f_static_0, flow0_fused)
        flow1_fused = torch.where(pull_static_1 > 0.5, f_static_1, flow1_fused)

    # BƯỚC 2.4: Kiểm định đạo hàm bậc 2 (Hessian) tại vùng nền tĩnh (M == 0)
    static_zone = (mask_total < 1e-4)
    hess_max = verify_static_zero_hessian(flow0_fused, static_zone)

    return flow0_fused, flow1_fused, {
        'segmentation': seg,
        'mask_total': mask_total,
        'weights_dyn': weights_dyn,
        'weight_static': weight_static,
        'static_hessian_max': hess_max,
    }


@torch.no_grad()
def verify_static_zero_hessian(flow_tensor, static_zone_mask=None):
    """
    Tính đạo hàm bậc 2 rời rạc (Discrete Second Derivatives: d2F/dx2, d2F/dy2, d2F/dxdy)
    của trường dòng chảy tại vùng nền tĩnh để chứng minh toán học ||Hessian(F)|| == 0.
    """
    # Sai phân bậc 2 theo trục X và trục Y
    d2x = flow_tensor[:, :, 1:-1, 2:] - 2.0 * flow_tensor[:, :, 1:-1, 1:-1] + flow_tensor[:, :, 1:-1, :-2]
    d2y = flow_tensor[:, :, 2:, 1:-1] - 2.0 * flow_tensor[:, :, 1:-1, 1:-1] + flow_tensor[:, :, :-2, 1:-1]
    hess_mag = torch.sqrt(d2x ** 2 + d2y ** 2)

    if static_zone_mask is not None:
        inner_mask = static_zone_mask[:, :, 1:-1, 1:-1]
        # Chỉ xét những điểm mà toàn bộ lân cận 3x3 đều nằm trong vùng tĩnh
        eroded_static = -F.max_pool2d(-inner_mask.float(), kernel_size=3, stride=1, padding=1) > 0.5
        if eroded_static.any():
            return float(hess_mag.masked_select(eroded_static.expand_as(hess_mag)).max().item())
        return 0.0
    return float(hess_mag.max().item())


# ==============================================================================
# TIỆN ÍCH PRECOMPUTE & TÁI TẠO NHANH TỪ TPS CACHE ĐA LỚP (.NPZ) CHO BƯỚC 4
# ==============================================================================
def solve_multilayer_cache_entry(
    kps0,
    kps1,
    img_hw,
    max_pts=128,
    max_clusters=4,
    max_mask_pts=64,
    max_static_pts=96,
    lambd=0.01,
    eps_static_px=1.5,
    use_dyn_anchors=True,
    filter_outliers=True,
):
    """
    Tiền tính toán toàn bộ tham số Multi-Layer Clustered TPS (Backward-Pair) cho 1 clip
    để lưu gọn vào `tps_cache_multilayer.npz` (không cần giải np.linalg.solve lúc train).
    """
    c0_dyn = np.zeros((max_clusters, max_pts, 2), dtype=np.float32)
    th01_dyn = np.zeros((max_clusters, max_pts + 3, 2), dtype=np.float32)
    c1_dyn = np.zeros((max_clusters, max_pts, 2), dtype=np.float32)
    th10_dyn = np.zeros((max_clusters, max_pts + 3, 2), dtype=np.float32)
    valid_dyn = np.zeros((max_clusters,), dtype=np.float32)

    dyn_pts0 = np.zeros((max_clusters, max_mask_pts, 2), dtype=np.float32)
    dyn_pts1 = np.zeros((max_clusters, max_mask_pts, 2), dtype=np.float32)
    dyn_pts_len = np.zeros((max_clusters,), dtype=np.int32)

    static_pts0 = np.zeros((max_static_pts, 2), dtype=np.float32)
    static_pts1 = np.zeros((max_static_pts, 2), dtype=np.float32)

    seg = segment_motion_keypoints(
        kps0, kps1, img_hw=img_hw, eps_static_px=eps_static_px, filter_outliers=filter_outliers
    )

    n_stat_raw = seg['num_static']
    if n_stat_raw > 0:
        if n_stat_raw > max_static_pts:
            s0_sub, s1_sub = _stratified_spatial_subsample(seg['static_kps0'], seg['static_kps1'], max_static_pts)
        else:
            s0_sub, s1_sub = seg['static_kps0'], seg['static_kps1']
        n_stat = len(s0_sub)
        static_pts0[:n_stat] = s0_sub.astype(np.float32)
        static_pts1[:n_stat] = s1_sub.astype(np.float32)
    else:
        n_stat = 0

    clusters = seg['dynamic_clusters']
    if len(clusters) > max_clusters:
        # Giữ top-(max_clusters-1) cụm lớn nhất và gộp phần còn lại vào cụm cuối
        clusters_sorted = sorted(clusters, key=lambda c: len(c['kps0']), reverse=True)
        head_c = clusters_sorted[:max_clusters - 1]
        tail_k0 = np.vstack([c['kps0'] for c in clusters_sorted[max_clusters - 1:]])
        tail_k1 = np.vstack([c['kps1'] for c in clusters_sorted[max_clusters - 1:]])
        clusters = head_c + [{'kps0': tail_k0, 'kps1': tail_k1}]

    n_clusters = len(clusters)
    for k, dc in enumerate(clusters):
        p0_k, p1_k = dc['kps0'], dc['kps1']
        c0_k, th01_k, v01 = solve_tps_dynamic_layer(p1_k, p0_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
        c1_k, th10_k, v10 = solve_tps_dynamic_layer(p0_k, p1_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
        c0_dyn[k], th01_dyn[k] = c0_k, th01_k
        c1_dyn[k], th10_dyn[k] = c1_k, th10_k
        valid_dyn[k] = 1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0

        if len(p0_k) > max_mask_pts:
            m0_sub, m1_sub = _stratified_spatial_subsample(p0_k, p1_k, max_mask_pts)
        else:
            m0_sub, m1_sub = p0_k, p1_k
        n_mk = len(m0_sub)
        dyn_pts0[k, :n_mk] = m0_sub.astype(np.float32)
        dyn_pts1[k, :n_mk] = m1_sub.astype(np.float32)
        dyn_pts_len[k] = n_mk

    return {
        'c0_dyn': c0_dyn,
        'th01_dyn': th01_dyn,
        'c1_dyn': c1_dyn,
        'th10_dyn': th10_dyn,
        'valid_dyn': valid_dyn,
        'dyn_pts0': dyn_pts0,
        'dyn_pts1': dyn_pts1,
        'dyn_pts_len': dyn_pts_len,
        'static_pts0': static_pts0,
        'static_pts1': static_pts1,
        'num_static': int(n_stat),
        'num_clusters': int(n_clusters),
        'num_outliers': int(seg['num_outliers']),
    }


@torch.no_grad()
def reconstruct_multilayer_flow_batch_from_cache(
    c0_dyn_b,
    th01_dyn_b,
    c1_dyn_b,
    th10_dyn_b,
    valid_dyn_b,
    dyn_pts0_b,
    dyn_pts1_b,
    dyn_pts_len_b,
    static_pts0_b,
    static_pts1_b,
    num_static_b,
    num_clusters_b,
    orig_hw_b,
    h_pad,
    w_pad,
    device,
    downscale_factor=2,
    use_dyn_anchors=True,
    transition_width_px=22.0,
    falloff_radius_px=105.0,
):
    """
    Tái tạo trực tiếp batch flow đa lớp [B, 2, h_pad, w_pad] trên GPU từ `tps_cache_multilayer.npz`
    mà không cần giải lại bất kỳ hệ phương trình TPS nào.
    """
    bsz = c0_dyn_b.shape[0]
    cf01_list = []
    cf10_list = []

    for i in range(bsz):
        nc = int(num_clusters_b[i].item())
        ns = int(num_static_b[i].item())
        ih = int(orig_hw_b[i, 0].item())
        iw = int(orig_hw_b[i, 1].item())

        if nc <= 0:
            cf01_list.append(torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device))
            cf10_list.append(torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device))
            continue

        dyn_flows_0 = []
        dyn_flows_1 = []
        dyn_corridors = []
        for k in range(nc):
            vk = float(valid_dyn_b[i, k].item())
            f0_k = _eval_tps_flow_single(th01_dyn_b[i, k], c0_dyn_b[i, k], vk, (ih, iw), h_pad, w_pad, device, downscale_factor)
            f1_k = _eval_tps_flow_single(th10_dyn_b[i, k], c1_dyn_b[i, k], vk, (ih, iw), h_pad, w_pad, device, downscale_factor)
            dyn_flows_0.append(f0_k)
            dyn_flows_1.append(f1_k)

            n_mk = int(dyn_pts_len_b[i, k].item())
            if n_mk > 0:
                p0_k = dyn_pts0_b[i, k, :n_mk]
                p1_k = dyn_pts1_b[i, k, :n_mk]
                pm_k = 0.5 * (p0_k + p1_k)
                dyn_corridors.append(torch.cat([p0_k, pm_k, p1_k], dim=0))

        if not dyn_corridors:
            cf01_list.append(torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device))
            cf10_list.append(torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device))
            continue

        if ns > 0:
            sp0 = static_pts0_b[i, :ns]
            sp1 = static_pts1_b[i, :ns]
            static_ref = 0.5 * (sp0 + sp1)
            eff_falloff = float(falloff_radius_px)
        elif use_dyn_anchors:
            static_ref = torch.as_tensor(FIXED_TPS_ANCHORS, dtype=torch.float32, device=device)
            eff_falloff = float(falloff_radius_px)
        else:
            static_ref = None
            eff_falloff = max(float(falloff_radius_px), 0.50 * math.hypot(float(iw), float(ih)))

        weights_dyn, _, mask_total = compute_soft_voronoi_blending_mask(
            dyn_corridors,
            static_ref,
            orig_hw=(ih, iw),
            h_pad=h_pad,
            w_pad=w_pad,
            device=device,
            downscale_factor=downscale_factor,
            transition_width_px=transition_width_px,
            falloff_radius_px=eff_falloff,
        )

        f0_fused = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)
        f1_fused = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)
        for w_k, f0_k, f1_k in zip(weights_dyn, dyn_flows_0, dyn_flows_1):
            f0_fused = f0_fused + w_k * f0_k
            f1_fused = f1_fused + w_k * f1_k

        if ns > 0:
            h_sub_g = max(16, h_pad // max(1, downscale_factor))
            w_sub_g = max(16, w_pad // max(1, downscale_factor))
            gy, gx = torch.meshgrid(
                torch.linspace(0.0, 1.0, h_sub_g, device=device, dtype=torch.float32),
                torch.linspace(0.0, 1.0, w_sub_g, device=device, dtype=torch.float32),
                indexing='ij'
            )
            base_g = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            sx = float(max(1, w_pad - 1)) / max(1.0, float(iw) - 1.0)
            sy = float(max(1, h_pad - 1)) / max(1.0, float(ih) - 1.0)
            scale_wh_px = torch.tensor([[[[float(iw - 1), float(ih - 1)]]]], dtype=torch.float32, device=device)
            scale_pad_to_orig = torch.tensor([[[[sx, sy]]]], dtype=torch.float32, device=device)

            f0_sub = F.interpolate(f0_fused, size=(h_sub_g, w_sub_g), mode='bilinear', align_corners=True) if (h_sub_g != h_pad or w_sub_g != w_pad) else f0_fused
            f1_sub = F.interpolate(f1_fused, size=(h_sub_g, w_sub_g), mode='bilinear', align_corners=True) if (h_sub_g != h_pad or w_sub_g != w_pad) else f1_fused
            m_sub = F.interpolate(mask_total, size=(h_sub_g, w_sub_g), mode='bilinear', align_corners=True) if (h_sub_g != h_pad or w_sub_g != w_pad) else mask_total

            q0_src_px = ((base_g + f0_sub.permute(0, 2, 3, 1)) * scale_pad_to_orig * scale_wh_px).view(1, -1, 2)
            q1_src_px = ((base_g + f1_sub.permute(0, 2, 3, 1)) * scale_pad_to_orig * scale_wh_px).view(1, -1, 2)
            sp0_px = (static_pts0_b[i, :ns] * scale_wh_px.view(1, 2)).unsqueeze(0)
            sp1_px = (static_pts1_b[i, :ns] * scale_wh_px.view(1, 2)).unsqueeze(0)

            pt_all_px = (torch.cat([
                0.5 * (dyn_pts0_b[i, k, :int(dyn_pts_len_b[i, k].item())] + dyn_pts1_b[i, k, :int(dyn_pts_len_b[i, k].item())])
                for k in range(nc) if int(dyn_pts_len_b[i, k].item()) > 0
            ], dim=0) * scale_wh_px.view(1, 2)).unsqueeze(0)
            q_cur_px = (base_g * scale_pad_to_orig * scale_wh_px).view(1, -1, 2)
            d_cur_dyn = torch.cdist(q_cur_px, pt_all_px, p=2.0).min(dim=-1).values.view(1, 1, h_sub_g, w_sub_g)

            d_src0_stat = torch.cdist(q0_src_px, sp0_px, p=2.0).min(dim=-1).values.view(1, 1, h_sub_g, w_sub_g)
            d_src1_stat = torch.cdist(q1_src_px, sp1_px, p=2.0).min(dim=-1).values.view(1, 1, h_sub_g, w_sub_g)
            outside_dyn_stroke = (m_sub > 1e-4) & (d_cur_dyn > 10.0)
            pull0 = (outside_dyn_stroke & (d_src0_stat < 22.0)).float()
            pull1 = (outside_dyn_stroke & (d_src1_stat < 22.0)).float()
            if h_sub_g != h_pad or w_sub_g != w_pad:
                pull0 = F.interpolate(pull0, size=(h_pad, w_pad), mode='nearest')
                pull1 = F.interpolate(pull1, size=(h_pad, w_pad), mode='nearest')
            f0_fused = torch.where(pull0 > 0.5, torch.zeros_like(f0_fused), f0_fused)
            f1_fused = torch.where(pull1 > 0.5, torch.zeros_like(f1_fused), f1_fused)

        cf01_list.append(f0_fused)
        cf10_list.append(f1_fused)

    return torch.cat(cf01_list, dim=0), torch.cat(cf10_list, dim=0)
