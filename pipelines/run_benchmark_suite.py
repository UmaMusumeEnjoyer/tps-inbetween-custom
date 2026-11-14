'#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_layer_tps.py
--------------------------------------------------------------------------------
PHA 3 (BÆ¯á»C 1 & BÆ¯á»C 2): MODULE COARSE TPS ÄA Lá»P (MULTI-LAYER CLUSTERED TPS)
Äá» tÃ i: Ná»i suy khung hÃ¬nh hoáº¡t há»a Anime Lineart (PBL6)
Kiáº¿n trÃºc ÄÃ­ch: TPSInbetweenOriginal (~10.07M active params, 0% thay Äá»i kiáº¿n trÃºc máº¡ng)

Giáº£i quyáº¿t triá»t Äá» háº¡n cháº¿ váº­t lÃ½ cá»§a Single-Layer Global TPS:
  1. BÆ°á»c 1 - PhÃ¢n cá»¥m Äiá»m chuyá»n Äá»ng (Motion Segmentation):
     - TÃ­nh váº­n tá»c tÆ°Æ¡ng Äá»i pixel: v_i = (P_{1,i} - P_{0,i}) * [W, H].
     - TÃ¡ch Cá»¥m Ná»n TÄ©nh (Static Layer, label = 0) vá»i ||v_i||_2 <= eps_static_px (1.5 px).
     - Khá»­ nhiá»u Äiá»m khá»p sai (Local Spatial-Velocity & RANSAC Affine Outlier Rejection, label = -1).
     - PhÃ¢n cá»¥m Tiá»n Cáº£nh Äá»ng (Foreground Layers, label = 1..K) báº±ng thuáº­t toÃ¡n
       Spatial-Velocity DBSCAN káº¿t há»£p kiá»m tra Äá»ng nháº¥t Affine cá»¥c bá».

  2. BÆ°á»c 2 - Thuáº­t toÃ¡n Multi-Layer TPS Flow Solver:
     - Giáº£i TPS_static trÃªn cá»¥m Äiá»m tÄ©nh (triá»t tiÃªu trá»ng sá» uá»n phi tuyáº¿n RBF ->
       Äáº¡o hÃ m báº­c 2 báº±ng 0 tuyá»t Äá»i, khÃ´ng lÃ m cong ná»n tÄ©nh).
     - Giáº£i TPS_dynamic Äá»c láº­p trÃªn tá»«ng cá»¥m Äiá»m Äá»ng (khÃ´ng bá» Äiá»m tÄ©nh kÃ©o ghÃ¬).
     - Táº¡o máº·t náº¡ má»m (Soft Blending Mask M) báº±ng khoáº£ng cÃ¡ch Hausdorff / Voronoi
       vá»i hÃ m chuyá»n tiáº¿p Quintic Smoothstep C^2:
           F_fused = M * F_dynamic + (1 - M) * F_static
     - Báº£o toÃ n Äáº¡o hÃ m báº­c 2 báº±ng 0 (pháº³ng tuyá»t Äá»i, ||Hessian(F)|| = 0) táº¡i vÃ¹ng ná»n tÄ©nh.
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
# BÆ¯á»C 1: MODULE PHÃN Cá»¤M ÄIá»M CHUYá»N Äá»NG (MOTION SEGMENTATION)
# ==============================================================================
def _dbscan_distance_matrix(dist_mat, eps=1.0, min_samples=3):
    """
    Thuáº­t toÃ¡n DBSCAN thuáº§n NumPy trÃªn ma tráº­n khoáº£ng cÃ¡ch chuáº©n hÃ³a D_sv (N x N).
    Tráº£ vá» máº£ng nhÃ£n cluster: 0, 1, ..., K-1 cho cÃ¡c cá»¥m há»£p lá», vÃ  -1 cho nhiá»u (noise).
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
    Khá»­ Äiá»m khá»p sai (Outliers) tá»« GlueStick/SIFT dá»±a trÃªn tÃ­nh Äá»ng nháº¥t váº­n tá»c - khÃ´ng gian
    vÃ  mÃ´ hÃ¬nh Affine cá»¥c bá» (báº£o toÃ n 100% biáº¿n dáº¡ng co giÃ£n hÃ¬nh há»c nhÆ° hÃ¬nh trÃ²n thu/phÃ³ng).
    """
    n = len(p0_px)
    if n < 6:
        return np.ones((n,), dtype=bool)

    mid_px = 0.5 * (p0_px + p1_px)
    dist_mat = np.linalg.norm(mid_px[:, None, :] - mid_px[None, :, :], axis=-1)

    k_use = min(k_nn, n - 1)
    nn_idx = np.argsort(dist_mat, axis=-1)[:, :k_use + 1]  # bao gá»m chÃ­nh Äiá»m i

    # 1. Kiá»m tra sai lá»ch váº­n tá»c cá»¥c bá» so vá»i median cá»§a lÃ¢n cáº­n k_nn
    local_vels = vel_px[nn_idx]  # [N, k_use+1, 2]
    med_vel = np.median(local_vels, axis=1)  # [N, 2]
    med_speed = np.linalg.norm(med_vel, axis=-1)
    vel_diff = np.linalg.norm(vel_px - med_vel, axis=-1)
    local_inlier = vel_diff <= np.maximum(min_abs_tol_px, max_rel_err * med_speed)

    # 2. Kiá»m tra thÃªm báº±ng Affine toÃ n cá»¥c/bÃ¡n cá»¥c bá» (náº¿u váº­t thá» biáº¿n dáº¡ng co giÃ£n Äá»u nhÆ° hÃ¬nh trÃ²n)
    if cv2 is not None and n >= 6:
        try:
            aff_m, aff_mask = cv2.estimateAffine2D(
                p0_px.astype(np.float32), p1_px.astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=max(12.0, 0.15 * float(np.median(speed_px)))
            )
            if aff_m is not None and aff_mask is not None:
                aff_inlier = aff_mask.ravel().astype(bool)
                # Äiá»m ÄÆ°á»£c giá»¯ náº¿u thá»a mÃ£n lÃ¢n cáº­n cá»¥c bá» HOáº¶C khá»p chÃ­nh xÃ¡c phÃ©p biáº¿n Äá»i Affine
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
    BÆ¯á»C 1: PhÃ¢n cá»¥m Äiá»m chuyá»n Äá»ng (Motion Segmentation) trÃªn cáº·p Äiá»m Äá»i sÃ¡nh P0 <-> P1.

    Tham sá»:
      - kps0, kps1: Tá»a Äá» Äiá»m khá»p chuáº©n hÃ³a [0, 1] kÃ­ch thÆ°á»c (N, 2).
      - img_hw: Tuple (H, W) kÃ­ch thÆ°á»c áº£nh gá»c tÃ­nh báº±ng pixel.
      - eps_static_px: NgÆ°á»¡ng váº­n tá»c (px) Äá» phÃ¢n láº­p Cá»¥m Ná»n TÄ©nh (máº·c Äá»nh <= 1.5 px).
      - spatial_radius_ratio: BÃ¡n kÃ­nh khÃ´ng gian chuáº©n hÃ³a theo ÄÆ°á»ng chÃ©o áº£nh cho DBSCAN.
      - vel_tol_px: NgÆ°á»¡ng dung sai váº­n tá»c cÆ¡ sá» (px) cho Spatial-Velocity DBSCAN.
      - min_cluster_pts: Sá» Äiá»m tá»i thiá»u Äá» táº¡o má»t cá»¥m TPS Äá»ng Äá»c láº­p (>= 4).
      - filter_outliers: Báº­t bá» lá»c loáº¡i bá» Äiá»m khá»p nhiá»u cá»¥c bá».

    Tráº£ vá» dict:
      - 'labels': Máº£ng (N,) vá»i:
          0     : Cá»¥m Ná»n TÄ©nh (Static Layer)
          1..K  : CÃ¡c Cá»¥m Tiá»n Cáº£nh Äá»ng (Dynamic Foreground Layers)
          -1    : Äiá»m nhiá»u bá» loáº¡i bá» (Outliers)
      - 'velocities_px': Vector váº­n tá»c (N, 2) tÃ­nh theo pixel.
      - 'speeds_px': Äá» lá»n váº­n tá»c ||v_i||_2 (N,) tÃ­nh theo pixel.
      - 'static_kps0', 'static_kps1': CÃ¡c Äiá»m thuá»c cá»¥m tÄ©nh.
      - 'dynamic_clusters': Danh sÃ¡ch dict mÃ´ táº£ tá»«ng cá»¥m Äá»ng k = 1..K.
      - 'all_dyn_kps0', 'all_dyn_kps1': ToÃ n bá» Äiá»m Äá»ng há»£p lá» gá»p chung.
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

    # 1. PhÃ¢n láº­p Cá»¥m TÄ©nh (Static Layer: ||v_i|| <= eps_static_px)
    static_mask = speed_px <= float(eps_static_px)
    labels[static_mask] = 0

    # 2. Xá»­ lÃ½ cÃ¡c Äiá»m chuyá»n Äá»ng (Dynamic Candidates: ||v_i|| > eps_static_px)
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

            # Khoáº£ng cÃ¡ch khÃ´ng gian - váº­n tá»c chuáº©n hÃ³a cho DBSCAN
            eps_spatial = max(32.0, float(spatial_radius_ratio) * diag_px)
            med_v = float(np.median(vspeed_px)) if len(vspeed_px) > 0 else float(vel_tol_px)
            eps_vel = max(float(vel_tol_px), 0.45 * med_v)

            dist_sp = np.linalg.norm(vmid_px[:, None, :] - vmid_px[None, :, :], axis=-1) / eps_spatial
            dist_vl = np.linalg.norm(vvel_px[:, None, :] - vvel_px[None, :, :], axis=-1) / eps_vel
            dist_sv = np.sqrt(dist_sp ** 2 + dist_vl ** 2)

            raw_clusters = _dbscan_distance_matrix(dist_sv, eps=1.0, min_samples=min(3, len(valid_dyn_idx)))

            # Gom cÃ¡c cá»¥m Äá»§ Äiá»m (>= min_cluster_pts) vÃ  gÃ¡n cÃ¡c Äiá»m rá»i ráº¡c há»£p lá» vÃ o cá»¥m gáº§n nháº¥t
            unique_c = [c for c in sorted(set(raw_clusters)) if c >= 0]
            major_clusters = [c for c in unique_c if int(np.sum(raw_clusters == c)) >= min_cluster_pts]

            if not major_clusters:
                # Náº¿u táº¥t cáº£ cá»¥m Äá»u nhá» nhÆ°ng tá»ng sá» Äiá»m Äá»ng >= min_cluster_pts -> gá»p thÃ nh 1 cá»¥m Äá»ng
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
                        # GÃ¡n Äiá»m biÃªn vÃ o cá»¥m chÃ­nh gáº§n nháº¥t náº¿u váº­n tá»c tÆ°Æ¡ng thÃ­ch
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

    # Tá»ng há»£p danh sÃ¡ch cá»¥m Äá»ng
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

    # Chuáº©n hÃ³a láº¡i nhÃ£n liÃªn tá»¥c 1..K sau khi lá»c
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
# BÆ¯á»C 2: THUáº¬T TOÃN MULTI-LAYER TPS FLOW SOLVER & SOFT VORONOI MASK
# ==============================================================================
def _stratified_spatial_subsample(c_src, c_dst, n_keep):
    """
    Láº¥y máº«u Äáº¡i diá»n phÃ¢n bá» Äá»u theo khÃ´ng gian (Farthest-Point / Spatial Grid Stratification)
    thay vÃ¬ cáº¯t chá» sá» tuyáº¿n tÃ­nh ÄÆ¡n thuáº§n, giÃºp giá»¯ trá»n váº¹n biÃªn hÃ¬nh há»c cá»§a má»i bá» pháº­n.
    """
    n = len(c_src)
    if n <= n_keep:
        return np.asarray(c_src, dtype=np.float64), np.asarray(c_dst, dtype=np.float64)

    cs = np.asarray(c_src, dtype=np.float64)
    cd = np.asarray(c_dst, dtype=np.float64)

    # Khá»i táº¡o báº±ng Äiá»m gáº§n tÃ¢m nháº¥t, sau ÄÃ³ chá»n láº·p Äiá»m xa nháº¥t (Farthest Point Sampling)
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
    """Giáº£i há» phÆ°Æ¡ng trÃ¬nh tuyáº¿n tÃ­nh TPS (K + lambda*I) w + P a = cd_all."""
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
    BÆ¯á»C 2.1 & 2.4: Giáº£i TPS_static cho cá»¥m Äiá»m ná»n tÄ©nh.
    Äáº£m báº£o Äáº¡o hÃ m báº­c 2 báº±ng 0 tuyá»t Äá»i (pháº³ng tuyá»t Äá»i, ||Hessian(F_static)|| = 0):
      - Khi cÃ¡c Äiá»m thuá»c Cá»¥m TÄ©nh (||v_i|| <= eps_static), náº¿u zero_hessian=True,
        ta triá»t tiÃªu hoÃ n toÃ n thÃ nh pháº§n uá»n phi tuyáº¿n RBF (th_pad[:m] = 0.0) vÃ 
        náº¿u Äá» lá»ch tÄ©nh trung vá» gáº§n báº±ng 0, Äáº·t Ã¡nh xáº¡ Äá»ng nháº¥t P(q) = q (zero flow tuyá»t Äá»i).
    """
    c_pad = np.zeros((max_pts, 2), dtype=np.float32)
    th_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
    # Ma tráº­n Affine Äá»ng nháº¥t f(x, y) = [0, 0] + 1*x + 1*y -> disp(q) = f(q) - q = 0
    th_pad[max_pts + 1, 0] = 1.0
    th_pad[max_pts + 2, 1] = 1.0

    if c_src_static is None or c_dst_static is None or len(c_src_static) == 0:
        return c_pad, th_pad, 1.0

    cs = np.asarray(c_src_static, dtype=np.float64)
    cd = np.asarray(c_dst_static, dtype=np.float64)
    max_disp = float(np.max(np.linalg.norm(cd - cs, axis=-1))) if len(cs) > 0 else 0.0

    # Náº¿u cá»¥m tÄ©nh Äá»©ng yÃªn hoáº·c dá»ch chuyá»n vi mÃ´ (< 2e-3 trÃªn miá»n chuáº©n hÃ³a [0,1]),
    # tráº£ vá» nghiá»m Äá»ng nháº¥t pháº³ng tuyá»t Äá»i (Zero Flow, Äáº¡o hÃ m báº­c 1 vÃ  báº­c 2 Äá»u báº±ng 0)
    if zero_hessian or max_disp < 2.5e-3 or len(cs) < 4:
        return c_pad, th_pad, 1.0

    n_keep = max_pts - 8
    cs_sub, cd_sub = _stratified_spatial_subsample(cs, cd, n_keep)
    anc = FIXED_TPS_ANCHORS.astype(np.float64)
    cs_all = np.vstack([cs_sub, anc])
    cd_all = np.vstack([cd_sub, anc])
    c_sol, th_sol, val = _solve_tps_system(cs_all, cd_all, max_pts=max_pts, lambd=lambd)
    if val > 0.5:
        # KhÃ³a cháº·t trá»ng sá» RBF phi tuyáº¿n = 0 Äá» Äáº£m báº£o Äáº¡o hÃ m báº­c 2 báº±ng 0 tuyá»t Äá»i
        th_sol[:max_pts, :] = 0.0
        return c_sol, th_sol, 1.0
    return c_pad, th_pad, 1.0


def solve_tps_dynamic_layer(c_src_dyn, c_dst_dyn, max_pts=128, lambd=0.01, use_anchors=False):
    """
    BÆ¯á»C 2.2: Giáº£i TPS_dynamic Äá»c láº­p trÃªn cá»¥m Äiá»m Äá»ng (khÃ´ng bá» Äiá»m tÄ©nh kÃ©o ghÃ¬).
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
    """ÄÃ¡nh giÃ¡ nhanh trÆ°á»ng dÃ²ng cháº£y TPS trÃªn GPU tá»« cáº·p (theta, c_src) kÃ­ch thÆ°á»c [1, ...]."""
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
    BÆ¯á»C 2.3 & 2.4: Táº¡o máº·t náº¡ hÃ²a trá»n má»m (Soft Blending Mask M) dá»±a trÃªn khoáº£ng cÃ¡ch
    Hausdorff / Voronoi tá»« má»i Äiá»m lÆ°á»i tá»i cÃ¡c Cá»¥m Äiá»m Äá»ng vÃ  Cá»¥m Äiá»m TÄ©nh.

    Äáº·c tÃ­nh toÃ¡n há»c:
      - Vá»i má»i cá»¥m Äá»ng k = 1..K, tÃ­nh khoáº£ng cÃ¡ch Euclidean ngáº¯n nháº¥t (Hausdorff point-to-set)
        d_dyn^(k)(q) tá»« Äiá»m lÆ°á»i q tá»i quá»¹ Äáº¡o Äiá»m Äá»ng.
      - Vá»i cá»¥m tÄ©nh, tÃ­nh khoáº£ng cÃ¡ch d_static(q) tá»i táº­p Äiá»m tÄ©nh.
        (Náº¿u khÃ´ng cÃ³ Äiá»m tÄ©nh tÆ°á»ng minh, dÃ¹ng ngÆ°á»¡ng bÃ¡n kÃ­nh áº£nh hÆ°á»ng falloff_radius_px).
      - Sá»­ dá»¥ng hÃ m Quintic Smoothstep C^2:
            u(q) = clip((d_static(q) - d_dyn(q) + w_trans) / (2 * w_trans), 0, 1)
            M(q) = 6*u^5 - 15*u^4 + 10*u^3
      - Táº¡i vÃ¹ng ná»n tÄ©nh (d_static(q) <= d_dyn(q) - w_trans):
            M(q) == 0.0 tuyá»t Äá»i -> Äáº¡o hÃ m báº­c 1 vÃ  báº­c 2 báº±ng 0 hoÃ n toÃ n!
      - Táº¡i lÃµi váº­t thá» Äá»ng (d_dyn(q) <= d_static(q) - w_trans):
            M(q) == 1.0 tuyá»t Äá»i -> Giá»¯ nguyÃªn 100% trÆ°á»ng biáº¿n dáº¡ng Äá»ng!

    Tráº£ vá»:
      - weights_dyn: List gá»m K tensor máº·t náº¡ [1, 1, h_pad, w_pad] cho tá»«ng cá»¥m Äá»ng.
      - weight_static: Tensor máº·t náº¡ tÄ©nh (1 - M_total) kÃ­ch thÆ°á»c [1, 1, h_pad, w_pad].
      - mask_total: Tensor M_total = sum_k M_k kÃ­ch thÆ°á»c [1, 1, h_pad, w_pad].
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

    pts_pad = _get_ml_norm_grid(h_sub, w_sub, device)  # [P, 2] trong [0, 1]^2 cá»§a canvas pad
    sx = float(max(1, w_pad - 1)) / max(1.0, orig_w - 1.0)
    sy = float(max(1, h_pad - 1)) / max(1.0, orig_h - 1.0)

    # Äá»i Äiá»m lÆ°á»i sang ÄÆ¡n vá» pixel thá»±c táº¿ cá»§a áº£nh gá»c [0, W] x [0, H]
    q_px = torch.stack([
        pts_pad[:, 0] * sx * max(1.0, orig_w - 1.0),
        pts_pad[:, 1] * sy * max(1.0, orig_h - 1.0),
    ], dim=-1).unsqueeze(0)  # [1, P, 2]

    scale_wh_t = torch.tensor([[max(1.0, orig_w - 1.0), max(1.0, orig_h - 1.0)]], dtype=torch.float32, device=device)

    # TÃ­nh khoáº£ng cÃ¡ch d_dyn^(k)(q) cho tá»«ng cá»¥m Äá»ng
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

    # TÃ­nh khoáº£ng cÃ¡ch d_static(q) tá»i cá»¥m Äiá»m tÄ©nh
    if static_pts_norm is not None and len(static_pts_norm) > 0:
        sp_t = torch.as_tensor(static_pts_norm, dtype=torch.float32, device=device)
        sp_px = (sp_t * scale_wh_t).unsqueeze(0)  # [1, M_s, 2]
        d_static = torch.cdist(q_px, sp_px, p=2.0).min(dim=-1).values  # [1, P]
        if falloff_radius_px is not None and falloff_radius_px > 0:
            d_static = torch.minimum(d_static, torch.full_like(d_static, float(falloff_radius_px)))
    else:
        d_static = torch.full_like(d_dyn_min, float(falloff_radius_px))

    # Máº·t náº¡ tá»ng M_total báº±ng hÃ m Quintic Smoothstep C^2 trÃªn biÃªn Voronoi (d_static - d_dyn_min)
    w_tr = max(2.0, float(transition_width_px))
    u = torch.clamp((d_static - d_dyn_min + w_tr) / (2.0 * w_tr), 0.0, 1.0)
    m_total_flat = (u ** 3) * (u * (u * 6.0 - 15.0) + 10.0)  # [1, P]

    # Báº£o vá» tuyá»t Äá»i vÃ¹ng lÃ¢n cáº­n Äiá»m tÄ©nh: má»i Äiá»m náº±m gáº§n Äiá»m tÄ©nh hÆ¡n Äiá»m Äá»ng
    # hoáº·c cÃ¡ch Äiá»m tÄ©nh dÆ°á»i ngÆ°á»¡ng an toÃ n (22 px) vÃ  xa Äiá»m Äá»ng Äá»u khÃ³a M = 0.0
    if static_pts_norm is not None and len(static_pts_norm) > 0:
        strict_static = (d_static < 24.0) & (d_dyn_min > d_static + 6.0)
        m_total_flat = torch.where(strict_static, torch.zeros_like(m_total_flat), m_total_flat)

    # KhÃ³a cá»©ng vÃ¹ng biÃªn Äá» Äáº£m báº£o Äáº¡o hÃ m báº­c 2 báº±ng 0 tuyá»t Äá»i trÃªn vÃ¹ng tÄ©nh
    m_total_flat = torch.where(m_total_flat < 1e-4, torch.zeros_like(m_total_flat), m_total_flat)
    m_total_flat = torch.where(m_total_flat > 1.0 - 1e-4, torch.ones_like(m_total_flat), m_total_flat)

    m_total = m_total_flat.view(1, 1, h_sub, w_sub)
    if h_sub != h_pad or w_sub != w_pad:
        m_total = F.interpolate(m_total, size=(h_pad, w_pad), mode='bilinear', align_corners=True)
        # Báº£o toÃ n vÃ¹ng 0 tuyá»t Äá»i sau ná»i suy bilinear
        m_total = torch.where(m_total < 1e-3, torch.zeros_like(m_total), m_total)
        m_total = torch.where(m_total > 1.0 - 1e-3, torch.ones_like(m_total), m_total)

    if len(d_dyn_list) == 1:
        weights_dyn = [m_total]
    else:
        # PhÃ¢n bá» Partition of Unity giá»¯a K cá»¥m Äá»ng theo Soft Voronoi
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
    BÆ¯á»C 2: HÃ m giáº£i tá»ng há»£p Multi-Layer Clustered TPS Flow Solver (`solve_multi_layer_tps`).

    Há» trá»£ Äáº§y Äá»§ cáº£ 2 cháº¿ Äá» váº­n hÃ nh cá»§a kiáº¿n trÃºc `TPSInbetweenOriginal`:
      1. Cháº¿ Äá» Lagrangian Direct-t (`direct_t_flows=True`, truyá»n `t_val` thuá»c [0, 1]):
         - Ná»i suy vá» trÃ­ trung gian P_t = (1 - t)*P_0 + t*P_1 cho tá»«ng lá»p.
         - Giáº£i TPS_static -> F_static_0t, F_static_1t (Äáº¡o hÃ m báº­c 2 = 0 tuyá»t Äá»i).
         - Giáº£i TPS_dynamic^(k) -> F_dyn_0t^(k), F_dyn_1t^(k) Äá»c láº­p vá»i ná»n tÄ©nh.
         - HÃ²a trá»n báº±ng máº·t náº¡ Voronoi/Hausdorff M:
             F_fused_0t = sum_k M_k * F_dyn_0t^(k) + (1 - M) * F_static_0t
             F_fused_1t = sum_k M_k * F_dyn_1t^(k) + (1 - M) * F_static_1t
      2. Cháº¿ Äá» Backward-Pair (`direct_t_flows=False`):
         - Giáº£i cáº·p dÃ²ng cháº£y ngÆ°á»£c F_01 (tá»« P_1 -> P_0) vÃ  F_10 (tá»« P_0 -> P_1) Äa lá»p.
         - Káº¿t há»£p máº·t náº¡ á»ng quá»¹ Äáº¡o (Trajectory Tube Voronoi Mask) báº£o vá» tuyá»t Äá»i
           cÃ¡c vÃ¹ng ná»n tÄ©nh khá»i nÄng lÆ°á»£ng uá»n cá»§a váº­t thá» chuyá»n Äá»ng.

    Tráº£ vá»:
      - flow0_fused: Tensor [1, 2, h_pad, w_pad] (cf0t hoáº·c cf01)
      - flow1_fused: Tensor [1, 2, h_pad, w_pad] (cf1t hoáº·c cf10)
      - meta_info: Dict chá»©a káº¿t quáº£ phÃ¢n cá»¥m BÆ°á»c 1, máº·t náº¡ M, vÃ  kiá»m Äá»nh Äáº¡o hÃ m báº­c 2.
    """
    if isinstance(orig_hw, torch.Tensor):
        img_h = int(orig_hw[0, 0].item())
        img_w = int(orig_hw[0, 1].item())
    else:
        img_h, img_w = int(orig_hw[0]), int(orig_hw[1])

    # BÆ¯á»C 1: PhÃ¢n cá»¥m Äiá»m chuyá»n Äá»ng TÄ©nh vs Äá»ng
    seg = segment_motion_keypoints(
        kps0, kps1,
        img_hw=(img_h, img_w),
        eps_static_px=eps_static_px,
        filter_outliers=filter_outliers,
    )

    zero_flow = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)

    # Náº¿u khÃ´ng cÃ³ cá»¥m Äá»ng nÃ o há»£p lá» -> tráº£ vá» dÃ²ng cháº£y tÄ©nh pháº³ng tuyá»t Äá»i (Zero Flow)
    if seg['num_clusters'] == 0:
        return zero_flow.clone(), zero_flow.clone(), {
            'segmentation': seg,
            'mask_total': torch.zeros((1, 1, h_pad, w_pad), dtype=torch.float32, device=device),
            'static_hessian_max': 0.0,
        }

    # BÆ¯á»C 2.1: Giáº£i TPS_static cho Cá»¥m Ná»n TÄ©nh (F_static cÃ³ Äáº¡o hÃ m báº­c 2 = 0 tuyá»t Äá»i)
    f_static_0 = zero_flow.clone()
    f_static_1 = zero_flow.clone()

    # BÆ¯á»C 2.2: Giáº£i TPS_dynamic cho tá»«ng Cá»¥m Tiá»n Cáº£nh Äá»ng
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
            # Giáº£i trá»±c tiáº¿p táº¡i má»c thá»i gian t: P_t -> P_0 vÃ  P_t -> P_1
            c_t0, th_t0, v0 = solve_tps_dynamic_layer(pt_k, p0_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            c_t1, th_t1, v1 = solve_tps_dynamic_layer(pt_k, p1_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            f0_k = _eval_tps_flow_single(th_t0, c_t0, v0, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            f1_k = _eval_tps_flow_single(th_t1, c_t1, v1, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            # Táº­p Äiá»m quá»¹ Äáº¡o Äá» bao phá»§ cáº£ vá» trÃ­ ÄÃ­ch P_t vÃ  vá» trÃ­ nguá»n P_0, P_1 (trÃ¡nh Äá» láº¡i bÃ³ng ma táº¡i nguá»n)
            corridor_k = np.vstack([p0_k, pm_k, pt_k, p1_k])
        else:
            # Giáº£i theo chuáº©n Backward-Pair: c0 = P_1 -> P_0 (flow01) vÃ  c1 = P_0 -> P_1 (flow10)
            c0_k, th01_k, v01 = solve_tps_dynamic_layer(p1_k, p0_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            c1_k, th10_k, v10 = solve_tps_dynamic_layer(p0_k, p1_k, max_pts=max_pts, lambd=lambd, use_anchors=use_dyn_anchors)
            f0_k = _eval_tps_flow_single(th01_k, c0_k, v01, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            f1_k = _eval_tps_flow_single(th10_k, c1_k, v10, (img_h, img_w), h_pad, w_pad, device, downscale_factor)
            corridor_k = np.vstack([p0_k, pm_k, p1_k])

        dyn_flows_0.append(f0_k)
        dyn_flows_1.append(f1_k)
        dyn_corridors.append(corridor_k)

    # Táº­p Äiá»m tÄ©nh tham chiáº¿u cho biÃªn Voronoi
    if seg['num_static'] > 0:
        static_ref_pts = (1.0 - tv) * seg['static_kps0'] + tv * seg['static_kps1']
        effective_falloff = float(falloff_radius_px)
    elif use_dyn_anchors:
        static_ref_pts = FIXED_TPS_ANCHORS
        effective_falloff = float(falloff_radius_px)
    else:
        static_ref_pts = None
        effective_falloff = max(float(falloff_radius_px), 0.50 * math.hypot(img_w, img_h))

    # BÆ¯á»C 2.3: TÃ­nh máº·t náº¡ má»m Soft Blending Mask M dá»±a trÃªn khoáº£ng cÃ¡ch Hausdorff / Voronoi
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

    # HÃ²a trá»n Äa lá»p: F_fused = sum_k M_k * F_dynamic^(k) + (1 - M) * F_static
    flow0_fused = weight_static * f_static_0
    flow1_fused = weight_static * f_static_1
    for w_k, f0_k, f1_k in zip(weights_dyn, dyn_flows_0, dyn_flows_1):
        flow0_fused = flow0_fused + w_k * f0_k
        flow1_fused = flow1_fused + w_k * f1_k

    # Anti-Ghost Pullback Guard táº¡i ranh giá»i giao thoa (Occlusion / Transition Boundary):
    # NgÄn cháº·n vÃ¹ng chuyá»n tiáº¿p (0 < M < 1) vÃ´ tÃ¬nh kÃ©o nÃ©t váº½ tá»« Cá»¥m TÄ©nh lÃªn vÃ¹ng trá»ng.
    # TÃ­nh trÃªn lÆ°á»i giáº£m máº«u (h_sub, w_sub) Äá» tiáº¿t kiá»m 4x VRAM vÃ  tÄng tá»c 4x trÃªn GPU.
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

    # BÆ¯á»C 2.4: Kiá»m Äá»nh Äáº¡o hÃ m báº­c 2 (Hessian) táº¡i vÃ¹ng ná»n tÄ©nh (M == 0)
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
    TÃ­nh Äáº¡o hÃ m báº­c 2 rá»i ráº¡c (Discrete Second Derivatives: d2F/dx2, d2F/dy2, d2F/dxdy)
    cá»§a trÆ°á»ng dÃ²ng cháº£y táº¡i vÃ¹ng ná»n tÄ©nh Äá» chá»©ng minh toÃ¡n há»c ||Hessian(F)|| == 0.
    """
    # Sai phÃ¢n báº­c 2 theo trá»¥c X vÃ  trá»¥c Y
    d2x = flow_tensor[:, :, 1:-1, 2:] - 2.0 * flow_tensor[:, :, 1:-1, 1:-1] + flow_tensor[:, :, 1:-1, :-2]
    d2y = flow_tensor[:, :, 2:, 1:-1] - 2.0 * flow_tensor[:, :, 1:-1, 1:-1] + flow_tensor[:, :, :-2, 1:-1]
    hess_mag = torch.sqrt(d2x ** 2 + d2y ** 2)

    if static_zone_mask is not None:
        inner_mask = static_zone_mask[:, :, 1:-1, 1:-1]
        # Chá» xÃ©t nhá»¯ng Äiá»m mÃ  toÃ n bá» lÃ¢n cáº­n 3x3 Äá»u náº±m trong vÃ¹ng tÄ©nh
        eroded_static = -F.max_pool2d(-inner_mask.float(), kernel_size=3, stride=1, padding=1) > 0.5
        if eroded_static.any():
            return float(hess_mag.masked_select(eroded_static.expand_as(hess_mag)).max().item())
        return 0.0
    return float(hess_mag.max().item())


# ==============================================================================
# TIá»N ÃCH PRECOMPUTE & TÃI Táº O NHANH Tá»ª TPS CACHE ÄA Lá»P (.NPZ) CHO BÆ¯á»C 4
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
    Tiá»n tÃ­nh toÃ¡n toÃ n bá» tham sá» Multi-Layer Clustered TPS (Backward-Pair) cho 1 clip
    Äá» lÆ°u gá»n vÃ o `tps_cache_multilayer.npz` (khÃ´ng cáº§n giáº£i np.linalg.solve lÃºc train).
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
        # Giá»¯ top-(max_clusters-1) cá»¥m lá»n nháº¥t vÃ  gá»p pháº§n cÃ²n láº¡i vÃ o cá»¥m cuá»i
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
    TÃ¡i táº¡o trá»±c tiáº¿p batch flow Äa lá»p [B, 2, h_pad, w_pad] trÃªn GPU tá»« `tps_cache_multilayer.npz`
    mÃ  khÃ´ng cáº§n giáº£i láº¡i báº¥t ká»³ há» phÆ°Æ¡ng trÃ¬nh TPS nÃ o.
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
')

build_cache_path = os.path.join(WORKING_DIR, "build_tps_cache_multilayer.py")
with open(build_cache_path, "w", encoding="utf-8") as f:
    f.write('#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tps_cache_multilayer.py
--------------------------------------------------------------------------------
PHA 3 - BÆ¯á»C 4: Táº O Bá» CACHE `tps_cache_multilayer.npz` TRÃN SAKUGA GOLDEN DATASET
VÃ KIá»M Äá»NH Äá»I CHá»¨NG TRÃN Táº¬P VAL (190 CLIPS / 420 FRAMES).

Táº¡o ra:
  1. `tps_cache_multilayer.npz` (~12-15 MB nÃ©n, 100% tÆ°Æ¡ng thÃ­ch DDP training).
  2. `train_sequences_indexed.csv` & `val_sequences_indexed.csv` (cÃ³ cá»t `cache_idx`).
  3. Thá»ng kÃª phÃ¢n cá»¥m toÃ n táº­p Train/Val & ÄÃ¡nh giÃ¡ nhanh M0/Blended trÃªn táº­p Val.
--------------------------------------------------------------------------------
"""

import os
import sys
import glob
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_DATASET_DIR = os.path.join(
    REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset",
    "sakuga_golden_dataset_sequences_tps_v2"
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from multi_layer_tps import solve_multilayer_cache_entry


def build_multilayer_cache(
    dataset_dir,
    out_npz_path=None,
    max_pts=128,
    max_clusters=4,
    max_mask_pts=64,
    max_static_pts=96,
    lambd=0.01,
    eps_static_px=1.5,
):
    if out_npz_path is None:
        out_npz_path = os.path.join(dataset_dir, "tps_cache_multilayer.npz")

    train_csv = os.path.join(dataset_dir, "train_sequences.csv")
    val_csv = os.path.join(dataset_dir, "val_sequences.csv")
    train_clips_dir = os.path.join(dataset_dir, "train")
    val_clips_dir = os.path.join(dataset_dir, "val")
    matches_dir = os.path.join(dataset_dir, "matches")

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_tr['_clip_root'] = train_clips_dir
    df_va['_clip_root'] = val_clips_dir
    df_all = pd.concat([df_tr, df_va], ignore_index=True)
    n_total = len(df_all)
    df_all['cache_idx'] = np.arange(n_total, dtype=np.int32)

    df_tr_out = df_all.iloc[:len(df_tr)].drop(columns=['_clip_root']).copy()
    df_va_out = df_all.iloc[len(df_tr):].drop(columns=['_clip_root']).copy()
    df_tr_out.to_csv(os.path.join(dataset_dir, "train_sequences_indexed.csv"), index=False)
    df_va_out.to_csv(os.path.join(dataset_dir, "val_sequences_indexed.csv"), index=False)

    c0_dyn = np.zeros((n_total, max_clusters, max_pts, 2), dtype=np.float32)
    th01_dyn = np.zeros((n_total, max_clusters, max_pts + 3, 2), dtype=np.float32)
    c1_dyn = np.zeros((n_total, max_clusters, max_pts, 2), dtype=np.float32)
    th10_dyn = np.zeros((n_total, max_clusters, max_pts + 3, 2), dtype=np.float32)
    valid_dyn = np.zeros((n_total, max_clusters), dtype=np.float32)

    dyn_pts0 = np.zeros((n_total, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dyn_pts1 = np.zeros((n_total, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dyn_pts_len = np.zeros((n_total, max_clusters), dtype=np.int32)

    static_pts0 = np.zeros((n_total, max_static_pts, 2), dtype=np.float32)
    static_pts1 = np.zeros((n_total, max_static_pts, 2), dtype=np.float32)
    num_static = np.zeros((n_total,), dtype=np.int32)
    num_clusters = np.zeros((n_total,), dtype=np.int32)
    num_outliers = np.zeros((n_total,), dtype=np.int32)
    orig_hw = np.zeros((n_total, 2), dtype=np.int32)

    def _task(item):
        idx, row = item
        cid = str(row['clip_id'])
        sl = int(row['seq_len'])
        c_root = row['_clip_root']
        img0_p = os.path.join(c_root, cid, "000001.png")
        with Image.open(img0_p) as im:
            w, h = im.size

        m_p = os.path.join(matches_dir, cid, f"match_000001_{sl:06d}.npy")
        if not os.path.exists(m_p):
            cands = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
            m_p = cands[0] if cands else ""

        if m_p and os.path.exists(m_p):
            kps = np.load(m_p).astype(np.float32)
            entry = solve_multilayer_cache_entry(
                kps[0], kps[1], img_hw=(h, w),
                max_pts=max_pts, max_clusters=max_clusters,
                max_mask_pts=max_mask_pts, max_static_pts=max_static_pts,
                lambd=lambd, eps_static_px=eps_static_px,
                use_dyn_anchors=True, filter_outliers=True,
            )
        else:
            entry = solve_multilayer_cache_entry(
                np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32),
                img_hw=(h, w), max_pts=max_pts, max_clusters=max_clusters,
                max_mask_pts=max_mask_pts, max_static_pts=max_static_pts,
            )
        return idx, h, w, entry

    t0 = time.time()
    print(f"âï¸ Äang táº¡o `tps_cache_multilayer.npz` cho {n_total:,} sequences "
          f"(MAX_PTS={max_pts}, MAX_CLUSTERS={max_clusters}, eps_static={eps_static_px}px)...", flush=True)

    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as pool:
        for idx, h, w, entry in pool.map(_task, list(df_all.iterrows())):
            c0_dyn[idx] = entry['c0_dyn']
            th01_dyn[idx] = entry['th01_dyn']
            c1_dyn[idx] = entry['c1_dyn']
            th10_dyn[idx] = entry['th10_dyn']
            valid_dyn[idx] = entry['valid_dyn']
            dyn_pts0[idx] = entry['dyn_pts0']
            dyn_pts1[idx] = entry['dyn_pts1']
            dyn_pts_len[idx] = entry['dyn_pts_len']
            static_pts0[idx] = entry['static_pts0']
            static_pts1[idx] = entry['static_pts1']
            num_static[idx] = entry['num_static']
            num_clusters[idx] = entry['num_clusters']
            num_outliers[idx] = entry['num_outliers']
            orig_hw[idx] = [h, w]

    np.savez_compressed(
        out_npz_path,
        c0_dyn=c0_dyn,
        th01_dyn=th01_dyn,
        c1_dyn=c1_dyn,
        th10_dyn=th10_dyn,
        valid_dyn=valid_dyn,
        dyn_pts0=dyn_pts0,
        dyn_pts1=dyn_pts1,
        dyn_pts_len=dyn_pts_len,
        static_pts0=static_pts0,
        static_pts1=static_pts1,
        num_static=num_static,
        num_clusters=num_clusters,
        num_outliers=num_outliers,
        orig_hw=orig_hw,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_npz_path) / (1024 ** 2)
    print(f"[â] ÄÃ£ lÆ°u thÃ nh cÃ´ng: {out_npz_path} ({size_mb:.2f} MB) trong {elapsed:.1f}s!")
    print(f"    - Sá» clips cÃ³ >= 1 cá»¥m Äá»ng há»£p lá»: {int((num_clusters >= 1).sum()):,}/{n_total:,}")
    print(f"    - Sá» clips cÃ³ >= 2 cá»¥m Äá»ng: {int((num_clusters >= 2).sum()):,}/{n_total:,}")
    print(f"    - Sá» clips cÃ³ Äiá»m ná»n tÄ©nh (>0): {int((num_static > 0).sum()):,}/{n_total:,}")
    print(f"    - Tá»ng sá» Äiá»m khá»p nhiá»u (outliers) ÄÃ£ lá»c: {int(num_outliers.sum()):,}")
    return out_npz_path


def main():
    parser = argparse.ArgumentParser(description="Táº¡o tps_cache_multilayer.npz cho BÆ°á»c 4 Pha 3")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out_npz", type=str, default=None)
    args = parser.parse_args()
    build_multilayer_cache(args.dataset_dir, args.out_npz)


if __name__ == "__main__":
    main()
')

worker_p2_path = os.path.join(WORKING_DIR, "ddp_worker_tps_sequence_phase2.py")
with open(worker_p2_path, "w", encoding="utf-8") as f:
    f.write('#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddp_worker_tps_sequence_final.py
--------------------------------------------------------------------------------
Äá»ng cÆ¡ Huáº¥n luyá»n & Kiá»m Äá»nh DDP 2x GPU Tesla T4 (Kaggle Dual-GPU) cho:
Fine-tune MÃ´ hÃ¬nh TPS-Inbetween Gá»c (model_latest.pt) trÃªn Sakuga Sequence Final Dataset
Äá» kiá»m chá»©ng khÃ¡ch quan (Zero-Bias Hypothesis Testing):
  - M0: Coarse TPS Warp (ToÃ¡n tá»­ uá»n lÆ°á»i hÃ¬nh há»c thuáº§n tÃºy, khÃ´ng máº¡ng nÆ¡-ron)
  - M1: Original TPS-Inbetween (model_latest.pt â Baseline gá»c ML240)
  - M3: Finetuned TPS-Inbetween (100% cÃ¹ng kiáº¿n trÃºc TPSInbetweenOriginal vá»i M1)

CÃ¡c chuáº©n má»±c ká»¹ thuáº­t cá»t lÃµi ÄÆ°á»£c Äáº£m báº£o trong file nÃ y:
1. 100% ZERO ARCHITECTURAL CHANGE:
   - M1 vÃ  M3 dÃ¹ng chung duy nháº¥t class TPSInbetweenOriginal (10,068,668 active params).
2. Sá»¬A TRIá»T Äá» Lá»I NGÆ¯á»¢C Dáº¤U BACKWARD-WARP TPS & DYNAMIC MULT32 PADDING:
   - KhÃ´i phá»¥c ÄÃºng quy Æ°á»c tpsinbet.py: flow01 (kps1 -> kps0), flow10 (kps0 -> kps1).
   - Xá»­ lÃ½ chÃ­nh xÃ¡c má»i chiá»u rá»ng khung hÃ¬nh W in [720, 1480] khi Äá»m mult-32.
3. Há» TRá»¢ 3 TÃY CHá»N Tá»C Äá» HUáº¤N LUYá»N (SHORT / MEDIUM / LONG):
   - Máº·c Äá»nh theo duyá»t Plan: TRAIN_MODE = "LONG" (Option 3 â Full-Trajectory
     540p, 8-term Anti-Thickening Loss: 1.0*Charb + 0.35*Sym-DT(8) + 0.20*Sobel +
     0.20*Soft-clDice + 0.15*VGG16 + 0.50*Exact-STE Density + 0.25*Stroke-Width +
     0.20*Ghost-Cleanliness, EMA vá»i adaptive warmup, Cosine LR Warmup).
   - DDP Length & Width-Bucketed Batch Sampler giá»¯ cáº£ 2 GPU á» ~98-100% táº£i.
   - Chunked Multi-frame Gradient Accumulation giá»¯ VRAM an toÃ n (~5-6 GB / 15 GB T4).
4. CHÃNH SÃCH CHECKPOINT TINH Gá»N TUYá»T Äá»I (CHá» 2 FILE):
   - model_tps_finetuned_best.pt : Checkpoint tá»t nháº¥t theo Two-Tier Gated Normalized Score.
   - model_tps_finetuned_last.pt : Checkpoint epoch gáº§n nháº¥t + tráº¡ng thÃ¡i Auto-Resume.
   - KhÃ´ng lÆ°u rá»i tá»«ng epoch! TÃ­ch há»£p RAM Watchdog + glibc malloc_trim().
--------------------------------------------------------------------------------
"""

import os
import sys
import gc
import time
import math
import json
import glob
import random
import argparse
import contextlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

# ==============================================================================
# 0. Tá»I Æ¯U Há» THá»NG, KHÃA PHÃN Máº¢NH RAM C & Cáº¤U HÃNH MÃI TRÆ¯á»NG
# ==============================================================================
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

torch.backends.cudnn.benchmark = True


def trim_memory():
    """Giáº£i phÃ³ng bá» nhá» Python vÃ  Ã©p glibc tráº£ láº¡i vÃ¹ng nhá» trá»ng cho OS (Linux/Kaggle)."""
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass


# ==============================================================================
# 1. TOÃN Tá»¬ BIáº¾N Dáº NG LÆ¯á»I & GIáº¢I TPS CHUáº¨N BACKWARD-WARP (KHá»P 100% TPSINBET.PY)
# ==============================================================================
FIXED_TPS_ANCHORS = np.array([
    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
    [0.0, 0.5],             [1.0, 0.5],
    [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
], dtype=np.float32)

_GRID_CACHE = {}
_PIXEL_GRID_CACHE = {}


def warp_normalized_flow(img, flow):
    """Backward-warp theo luá»ng Coarse TPS (tá»a Äá» chuáº©n hÃ³a [0, 1] -> [-1, 1])."""
    _, _, h, w = img.shape
    key = (h, w, flow.device)
    base_grid = _GRID_CACHE.get(key)
    if base_grid is None:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=flow.device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, w, device=flow.device, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
        _GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        grid = (base_grid + flow.float().permute(0, 2, 3, 1)) * 2.0 - 1.0
        out = F.grid_sample(
            img.float().contiguous(), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )
    return out.to(img.dtype)


def warp_pixel_flow(img, flow):
    """Backward-warp theo luá»ng quang há»c IFNet (ÄÆ¡n vá» pixel)."""
    _, _, h, w = img.shape
    key = (h, w, flow.device)
    base_grid = _PIXEL_GRID_CACHE.get(key)
    if base_grid is None:
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
        _PIXEL_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = flow.float()
        flow_norm = torch.cat([
            flow_f32[:, 0:1, :, :] / max(1.0, (w - 1.0) / 2.0),
            flow_f32[:, 1:2, :, :] / max(1.0, (h - 1.0) / 2.0)
        ], dim=1)
        grid = (base_grid + flow_norm.permute(0, 2, 3, 1)).contiguous()
        out = F.grid_sample(
            img.float().contiguous(), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )
    return out.to(img.dtype)


def tps_rbf_torch(r):
    r = torch.clamp(r, min=1e-6)
    return (r ** 2) * torch.log(r)


def _solve_tps_system(cs_all, cd_all, max_pts=128, lambd=0.01):
    """Giáº£i há» phÆ°Æ¡ng trÃ¬nh tuyáº¿n tÃ­nh TPS chung (dÃ¹ng cho cáº£ Backward-Pair vÃ  Lagrangian Direct)."""
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


def solve_tps_direct(c_src, c_dst, max_pts=128, lambd=0.01, use_anchors=False):
    """Giáº£i TPS trá»±c tiáº¿p tá»« c_src -> c_dst (dÃ¹ng cho Lagrangian Direct TPS táº¡i má»c t)."""
    if c_src is None or c_dst is None or len(c_src) < 4:
        return (
            np.zeros((max_pts, 2), dtype=np.float32),
            np.zeros((max_pts + 3, 2), dtype=np.float32),
            0.0,
        )
    n_keep = (max_pts - 8) if use_anchors else max_pts
    if len(c_src) > n_keep:
        idx = np.linspace(0, len(c_src) - 1, n_keep, dtype=int)
        cs_sub = np.asarray(c_src[idx], dtype=np.float64)
        cd_sub = np.asarray(c_dst[idx], dtype=np.float64)
    else:
        cs_sub = np.asarray(c_src, dtype=np.float64)
        cd_sub = np.asarray(c_dst, dtype=np.float64)
    if use_anchors:
        anchors64 = FIXED_TPS_ANCHORS.astype(np.float64)
        cs_all = np.vstack([cs_sub, anchors64])
        cd_all = np.vstack([cd_sub, anchors64])
    else:
        cs_all, cd_all = cs_sub, cd_sub
    return _solve_tps_system(cs_all, cd_all, max_pts=max_pts, lambd=lambd)


def solve_tps_backward_pair(kps0, kps1, max_pts=128, lambd=0.01):
    """
    Giáº£i há» phÆ°Æ¡ng trÃ¬nh TPS theo ÄÃºng chiá»u Backward-Warp cá»§a tpsinbet.py:
      - flow01 ÄÆ°á»£c ná»i suy tá»« cÃ¡c Äiá»m Äiá»u khiá»n táº¡i kps1 nháº¯m tá»i kps0 (c0=kps1 -> kps0)
      - flow10 ÄÆ°á»£c ná»i suy tá»« cÃ¡c Äiá»m Äiá»u khiá»n táº¡i kps0 nháº¯m tá»i kps1 (c1=kps0 -> kps1)
    Tá»± Äá»ng giáº£m máº«u Äá»u xuá»ng (max_pts - 8) Äiá»m + 8 Äiá»m neo biÃªn cá» Äá»nh [0, 1]^2.
    Tráº£ vá»: c0, th01, c1, th10, valid_flag.
    """
    c0, th01, v01 = solve_tps_direct(kps1, kps0, max_pts=max_pts, lambd=lambd, use_anchors=True)
    c1, th10, v10 = solve_tps_direct(kps0, kps1, max_pts=max_pts, lambd=lambd, use_anchors=True)
    v = 1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0
    return c0, th01, c1, th10, v


def _get_norm_grid_pts(h, w, device):
    key = (h, w, device, 'tps_pts_2d')
    pts = _GRID_CACHE.get(key)
    if pts is None:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pts = torch.stack([gx, gy], dim=-1).view(-1, 2).contiguous()
        _GRID_CACHE[key] = pts
    return pts


@torch.no_grad()
def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, orig_hw, h_pad, w_pad, device, downscale_factor=2):
    """
    Ná»i suy chÃ­nh xÃ¡c trÆ°á»ng dÃ²ng cháº£y Coarse TPS trÃªn GPU tá»« nghiá»m ÄÃ£ tiá»n tÃ­nh toÃ¡n,
    tá»± Äá»ng co giÃ£n chÃ­nh xÃ¡c theo tá»· lá» khung áº£nh gá»c (orig_h, orig_w) bÃªn trong canvas Äá»m (h_pad, w_pad).
    DÃ¹ng chunking (chunk_size=32768 Äiá»m) Äá» giá»¯ má»©c Äá»nh VRAM tháº¥p (<35 MB).
    """
    with torch.amp.autocast('cuda', enabled=False):
        theta_f32 = theta_batch.float()
        c_src_f32 = c_src_batch.float()
        valid_f32 = valid_mask.float()
        b = theta_f32.shape[0]

        h_sub = max(16, h_pad // max(1, downscale_factor))
        w_sub = max(16, w_pad // max(1, downscale_factor))

        pts_pad = _get_norm_grid_pts(h_sub, w_sub, device)  # [P, 2]
        total_pts = pts_pad.shape[0]
        pts_pad_exp = pts_pad.unsqueeze(0).expand(b, -1, -1)  # [B, P, 2]

        # Tá»· lá» quy Äá»i giá»¯a tá»a Äá» chuáº©n hÃ³a cá»§a canvas (h_pad, w_pad) vÃ  áº£nh gá»c (orig_h, orig_w)
        orig_hw_f = orig_hw.to(device=device, dtype=torch.float32)  # [B, 2] -> (H_orig, W_orig)
        sx = (float(max(1, w_pad - 1)) / torch.clamp(orig_hw_f[:, 1] - 1.0, min=1.0)).view(b, 1, 1)
        sy = (float(max(1, h_pad - 1)) / torch.clamp(orig_hw_f[:, 0] - 1.0, min=1.0)).view(b, 1, 1)
        scale_xy = torch.cat([sx, sy], dim=-1)  # [B, 1, 2]

        # Quy vá» tá»a Äá» gá»c [0, 1] vÃ  giá»¯ dáº¥u vÃ¹ng biÃªn há»£p lá»
        pts_orig_raw = pts_pad_exp * scale_xy
        in_bounds = ((pts_orig_raw[..., 0:1] <= 1.001) & (pts_orig_raw[..., 1:2] <= 1.001)).float()
        pts_orig = torch.clamp(pts_orig_raw, 0.0, 1.0)

        flow_chunks = []
        chunk_size = 32768
        for i in range(0, total_pts, chunk_size):
            end_i = min(i + chunk_size, total_pts)
            q = pts_orig[:, i:end_i, :]  # [B, C, 2]
            dist = torch.cdist(q, c_src_f32, p=2.0)  # [B, C, M]
            U = tps_rbf_torch(dist)
            ones = torch.ones((b, q.shape[1], 1), device=device, dtype=torch.float32)
            basis = torch.cat([U, ones, q], dim=-1)  # [B, C, M+3]
            mapped = torch.bmm(basis, theta_f32)  # [B, C, 2]
            disp_orig = mapped - q
            disp_pad = (disp_orig / scale_xy) * in_bounds[:, i:end_i, :]
            flow_chunks.append(disp_pad)

        flow = torch.cat(flow_chunks, dim=1).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()
        if h_sub != h_pad or w_sub != w_pad:
            flow = F.interpolate(flow, size=(h_pad, w_pad), mode='bilinear', align_corners=True)

        flow.mul_(valid_f32.view(b, 1, 1, 1))
        return flow


# ==============================================================================
# 2. KIáº¾N TRÃC MÃ HÃNH NGUYÃN Báº¢N 100% (TPSInbetweenOriginal: ZHU ET AL., 2024)
# ==============================================================================
def conv(in_p, out_p, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_p, out_p, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_p)
    )


def deconv(in_p, out_p, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_p, out_p, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_p)
    )


class Conv2(nn.Module):
    def __init__(self, in_p, out_p, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_p, out_p, 3, stride, 1)
        self.conv2 = conv(out_p, out_p, 3, 1, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Contextnet(nn.Module):
    def __init__(self, indim=1, c=16):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(indim, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f1 = warp_normalized_flow(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f2 = warp_normalized_flow(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f3 = warp_normalized_flow(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f4 = warp_normalized_flow(x, flow)
        return [f1, f2, f3, f4]


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(8)])
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        orig_h, orig_w = x.shape[2], x.shape[3]
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x.contiguous())
        x = self.convblock(x) + x
        with torch.amp.autocast('cuda', enabled=False):
            tmp = self.lastconv(x.float().contiguous())
            tmp = F.interpolate(tmp, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            flow = tmp[:, :4] * scale * 2
            mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)

    def forward(self, I0t, I1t, scale=(4, 2, 1)):
        img0 = I0t.repeat(1, 3, 1, 1)
        img1 = I1t.repeat(1, 3, 1, 1)
        warped_i0 = I0t
        warped_i1 = I1t
        flow = None
        mask = None
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                if flow.shape[2:] != I0t.shape[2:]:
                    flow = F.interpolate(flow, size=I0t.shape[2:], mode='bilinear', align_corners=False)
                if mask.shape[2:] != I0t.shape[2:]:
                    mask = F.interpolate(mask, size=I0t.shape[2:], mode='bilinear', align_corners=False)
                inp = torch.cat((img0, img1, warped_i0.repeat(1, 3, 1, 1), warped_i1.repeat(1, 3, 1, 1), mask), 1)
                flow_d, mask_d = stu[i](inp, flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            if i < 2:
                warped_i0 = warp_pixel_flow(I0t, flow[:, :2])
                warped_i1 = warp_pixel_flow(I1t, flow[:, 2:4])
        return flow, torch.sigmoid(mask)


class OriginalUnet(nn.Module):
    """UNet gá»c cá»§a bÃ i bÃ¡o (indim=9, outdim=1, Äáº§u ra residual [-1, 1])."""
    def __init__(self, indim=9, outdim=1, c=16):
        super(OriginalUnet, self).__init__()
        self.down0 = Conv2(indim, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, outdim, 3, 1, 1)

    def forward(self, img0, img1, flow0, flow1, warped0, warped1, mask, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped0, warped1, flow0, flow1, mask), 1))
        if s0.shape[2:] != c0[0].shape[2:]:
            s0 = F.interpolate(s0, size=c0[0].shape[2:], mode='bilinear', align_corners=False)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        if s1.shape[2:] != c0[1].shape[2:]:
            s1 = F.interpolate(s1, size=c0[1].shape[2:], mode='bilinear', align_corners=False)
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        if s2.shape[2:] != c0[2].shape[2:]:
            s2 = F.interpolate(s2, size=c0[2].shape[2:], mode='bilinear', align_corners=False)
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        if s3.shape[2:] != c0[3].shape[2:]:
            s3 = F.interpolate(s3, size=c0[3].shape[2:], mode='bilinear', align_corners=False)
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        if x.shape[2:] != s2.shape[2:]:
            x = F.interpolate(x, size=s2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat((x, s2), 1))
        if x.shape[2:] != s1.shape[2:]:
            x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat((x, s1), 1))
        if x.shape[2:] != s0.shape[2:]:
            x = F.interpolate(x, size=s0.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat((x, s0), 1))
        if x.shape[2:] != img0.shape[2:]:
            x = F.interpolate(x, size=img0.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv(x)
        return torch.sigmoid(x)


class TPSInbetweenOriginal(nn.Module):
    """
    Kiáº¿n trÃºc nguyÃªn báº£n TPS-Inbetween (Zhu et al., HIT 2024).
    DÃ¹ng chung DUY NHáº¤T 1 class nÃ y cho M1 (model_latest.pt) vÃ  M3 (Finetuned):
    13,026,543 active parameters â CAM Káº¾T 0% THAY Äá»I KIáº¾N TRÃC.
    """
    def __init__(self):
        super(TPSInbetweenOriginal, self).__init__()
        self.encoder = Contextnet(indim=1, c=16)
        self.IFNet = IFNet()
        self.unet = OriginalUnet(indim=9, outdim=1, c=16)

    def forward(self, x0, x1, t_val, coarse_flow01, coarse_flow10, direct_t_flows=False):
        t = t_val
        if direct_t_flows:
            coarse_flow0t = coarse_flow01
            coarse_flow1t = coarse_flow10
        else:
            coarse_flow0t = coarse_flow01 * (t ** 2) - (1.0 - t) * t * coarse_flow10
            coarse_flow1t = coarse_flow10 * ((1.0 - t) ** 2) - t * (1.0 - t) * coarse_flow01

        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)
        coarse_warp = torch.clamp((1.0 - t) * coarse_I0t + t * coarse_I1t, 0.0, 1.0)

        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)
        flow_bet_coar, m = self.IFNet(coarse_I0t, coarse_I1t)

        fine_I0t = warp_pixel_flow(coarse_I0t, flow_bet_coar[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_bet_coar[:, 2:4])

        refine_out = self.unet(
            x0, x1,
            coarse_flow0t + flow_bet_coar[:, :2],
            coarse_flow1t + flow_bet_coar[:, 2:4],
            fine_I0t, fine_I1t, m, feats0, feats1
        )
        res = refine_out[:, 0:1, :, :] * 2.0 - 1.0
        It = torch.clamp((1.0 - t) * fine_I0t + t * fine_I1t + res, 0.0, 1.0)
        return It, coarse_warp, fine_I0t, fine_I1t, flow_bet_coar, m


def load_tps_weights(model, ckpt_path, device='cpu', prefer_ema=False, is_rank0=True):
    """
    Náº¡p trá»ng sá» tá»« model_latest.pt gá»c (252 keys gá»m 94 dead keys) hoáº·c checkpoint finetune.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y file checkpoint táº¡i: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if prefer_ema and raw.get('ema_state') is not None:
            ema_obj = raw['ema_state']
            sd = ema_obj['shadow'] if isinstance(ema_obj, dict) and 'shadow' in ema_obj else ema_obj
        elif 'model_state' in raw:
            sd = raw['model_state']
        elif 'state_dict' in raw:
            sd = raw['state_dict']
        elif 'model' in raw:
            sd = raw['model']
        else:
            sd = raw
    else:
        sd = raw

    model_sd = model.state_dict()
    matched_sd = {}
    skipped_keys = []
    for k, v in sd.items():
        clean_k = k.replace('module.', '').replace('base_model.', '')
        if clean_k in model_sd and model_sd[clean_k].shape == v.shape:
            matched_sd[clean_k] = v
        else:
            skipped_keys.append(clean_k)

    missing, _ = model.load_state_dict(matched_sd, strict=False)
    if len(missing) > 0:
        raise RuntimeError(f"Thiáº¿u {len(missing)} active tensors khi náº¡p {ckpt_path}: {missing[:5]}")

    if is_rank0:
        tag = os.path.basename(ckpt_path)
        print(f"[â] Náº¡p trá»ng sá» ({tag}): Khá»p {len(matched_sd)}/{len(model_sd)} active tensors 100%!"
              + (f" (Bá» qua {len(skipped_keys)} unused/dead keys)" if skipped_keys else ""), flush=True)
    return model


# ==============================================================================
# 3. HÃM Máº¤T MÃT CHUYÃN Dá»¤NG CHO LINEART (8-TERM ANTI-THICKENING LOSS)
# ==============================================================================
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


def approx_gpu_distance_map(bin_mask, steps=8):
    """
    Báº£n Äá» khoáº£ng cÃ¡ch Euclidean xáº¥p xá» siÃªu nhanh báº±ng phÃ©p giÃ£n 3x3 láº·p trÃªn GPU [0, 1].
    GiÃ¡ trá» 0.0 táº¡i chÃ­nh pixel nÃ©t vÃ  tÄng tuyáº¿n tÃ­nh tá»i 1.0 ngoÃ i bÃ¡n kÃ­nh `steps` pixel.
    """
    curr = bin_mask
    d_sum = curr
    for _ in range(steps):
        curr = F.max_pool2d(curr, kernel_size=3, stride=1, padding=1)
        d_sum = d_sum + curr
    dt = 1.0 - (d_sum / float(steps + 1))
    return torch.clamp(dt, 0.0, 1.0)


class ExactSTEDensityLoss(nn.Module):
    """
    HÃ m pháº¡t Äá» dÃ y nÃ©t váº½ (Density Loss) vá»i Straight-Through Estimator (STE)
    vÃ  Leaky Surrogate Ramp táº¡i ngÆ°á»¡ng nhá» phÃ¢n tau=0.15.
    Dead-zone [0.96, 1.04], pháº¡t báº¥t Äá»i xá»©ng cá»±c máº¡nh khi dÃ y nÃ©t (dev_high).
    """
    def __init__(self, tau=0.15, dead_zone_low=0.96, dead_zone_high=1.04):
        super().__init__()
        self.tau = tau
        self.low = dead_zone_low
        self.high = dead_zone_high

    def forward(self, pred, gt):
        s_p = torch.clamp((pred - 0.06) / 0.18, 0.0, 1.0) + 0.25 * F.relu(pred - 0.24)
        h_p = (pred >= self.tau).float()
        m_p = h_p - s_p.detach() + s_p  # STE

        h_g = (gt >= self.tau).float()
        area_p = torch.sum(m_p, dim=(-2, -1))
        area_g = torch.sum(h_g, dim=(-2, -1)).detach()

        r = area_p / torch.clamp(area_g, min=1.0)

        dev_low = F.relu(self.low - r)
        dev_high = F.relu(r - self.high)
        dev_center = r - 1.0

        loss = 1.8 * dev_low + 2.8 * dev_high + 1.5 * (dev_high ** 2) + 0.06 * (dev_center ** 2)
        return loss.mean()


class BidirectionalStrokeWidthLoss(nn.Module):
    """
    HÃ m kiá»m soÃ¡t bá» rá»ng nÃ©t váº½ hai chiá»u:
      - NhÃ¡nh 1: Báº£o vá» nÃ©t máº£nh & ngá»n tÃ³c bÃªn trong lÃµi nÃ©t GT (anti-erosion).
      - NhÃ¡nh 2: Pháº¡t overspill trÃ n má»±c trÃªn vÃ nh 1-2px bÃªn ngoÃ i nÃ©t GT (anti-dilation).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        with torch.no_grad():
            gt_core = soft_skeletonize(gt, iters=2)
            gt_stroke = (gt > 0.08).float()
            gt_dilated = soft_dilate(soft_dilate(gt_stroke))
            gt_inner = (gt > 0.12).float()
            edge_ring = torch.clamp(gt_dilated - gt_inner, 0.0, 1.0)

        # NhÃ¡nh 1: Báº£o vá» lÃµi skeleton khÃ´ng bá» má»/Äá»©t nÃ©t
        loss_core = torch.sum(gt_core * F.relu(gt_core - pred), dim=(-3, -2, -1)) / torch.clamp(torch.sum(gt_core, dim=(-3, -2, -1)), min=1.0)

        # NhÃ¡nh 2: Trá»«ng pháº¡t trÃ n má»±c (overspill) trÃªn vÃ nh edge_ring quanh nÃ©t
        over_spill = F.relu(pred - torch.maximum(gt, torch.tensor(0.06, device=pred.device))) + 2.0 * F.relu(pred - 0.14)
        loss_ring = torch.sum(edge_ring * over_spill, dim=(-3, -2, -1)) / torch.clamp(torch.sum(edge_ring, dim=(-3, -2, -1)), min=1.0)

        return (loss_core + loss_ring).mean()


class GhostAndBackgroundCleanlinessLoss(nn.Module):
    """
    HÃ m triá»t tiÃªu sÆ°Æ¡ng mÃ¹ (haze), bÃ³ng ma xa (far ghost) vÃ  quáº§ng má» sÃ¡t biÃªn (halo) - Pha 2 Cleanliness Purge:
      - Zone 1: Ná»n xa (cÃ¡ch nÃ©t GT > 5px): chuáº©n hÃ³a theo diá»n tÃ­ch nÃ©t GT, bá» sung thÃ nh pháº§n pháº¡t báº­c hai (quad-penalty) cá»±c máº¡nh khi pred >= 0.035 Äá» dá»n sáº¡ch lÆ°á»£ng sÆ°Æ¡ng má» tá»n Äá»ng tá»« Pha 1.
      - Zone 2: VÃ¹ng hÃ o quang sÃ¡t nÃ©t: pháº¡t má»±c lÆ¡ lá»­ng >= 0.06.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        with torch.no_grad():
            gt_stroke_wide = soft_dilate(soft_dilate((gt > 0.04).float()))
            far_bg = torch.clamp(1.0 - gt_stroke_wide, 0.0, 1.0)
            near_halo = torch.clamp(gt_stroke_wide - (gt > 0.15).float(), 0.0, 1.0)
            gt_area = torch.clamp(torch.sum((gt > 0.15).float(), dim=(-3, -2, -1)), min=500.0)

        near_count = torch.clamp(torch.sum(near_halo, dim=(-3, -2, -1)), min=200.0)

        # Pháº¡t máº¡nh lÅ©y tiáº¿n cho cÃ¡c giÃ¡ trá» pred lÆ¡ lá»­ng trÃªn ná»n xa:
        # Sá» háº¡ng báº­c hai (pen_far_p035 ** 2) táº¡o gradient ráº¥t máº¡nh Äá»i vá»i cÃ¡c váº¿t má»±c má» >= 0.035
        pen_far_p035 = F.relu(pred - 0.035)
        pen_far = (
            1.5 * F.relu(pred - 0.012)
            + 4.0 * pen_far_p035
            + 6.0 * (pen_far_p035 ** 2)
            + 8.0 * F.relu(pred - 0.20)
        )
        loss_far = torch.sum(far_bg * pen_far, dim=(-3, -2, -1)) / gt_area

        pen_halo_p06 = F.relu(pred - 0.06)
        pen_halo = 0.8 * pen_halo_p06 + 1.2 * (pen_halo_p06 ** 2)
        loss_halo = torch.sum(near_halo * pen_halo, dim=(-3, -2, -1)) / near_count

        return (loss_far + loss_halo).mean()


class SoftClDiceLoss(nn.Module):
    def __init__(self, iters=2, smooth=1.0):
        super().__init__()
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


class VGGPerceptualOrLaplacianLoss(nn.Module):
    """
    Perceptual Loss dá»±a trÃªn VGG16 ÄÃ´ng bÄng (khá»p tinh tháº§n bÃ i bÃ¡o gá»c TPS-Inbetween).
    Tá»± Äá»ng dá»± phÃ²ng sang Multi-Scale Laplacian Pyramid Loss náº¿u cháº¡y á» mÃ´i trÆ°á»ng offline
    chÆ°a cache sáºµn weights VGG16 (khÃ´ng bao giá» gÃ¢y crash).
    """
    def __init__(self, is_rank0=True):
        super().__init__()
        self.use_vgg = False
        self.blocks = None
        try:
            import torchvision.models as tv_models
            weights_enum = getattr(tv_models, 'VGG16_Weights', None)
            if weights_enum is not None:
                vgg = tv_models.vgg16(weights=weights_enum.DEFAULT).features
            else:
                vgg = tv_models.vgg16(pretrained=True).features
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            # Láº¥y 4 táº§ng relu1_2(4), relu2_2(9), relu3_3(16), relu4_3(23)
            self.slice1 = nn.Sequential(*list(vgg[:4]))
            self.slice2 = nn.Sequential(*list(vgg[4:9]))
            self.slice3 = nn.Sequential(*list(vgg[9:16]))
            self.slice4 = nn.Sequential(*list(vgg[16:23]))
            self.use_vgg = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            if is_rank0:
                print("[â] VGG16 Perceptual Loss ÄÃ£ kÃ­ch hoáº¡t (Option 3 LONG Mode).", flush=True)
        except Exception as e:
            if is_rank0:
                print(f"[*] VGG16 chÆ°a táº£i ÄÆ°á»£c ({e}) -> Fallback sang 4-Scale Laplacian Perceptual Loss.", flush=True)

    def _lap_pyr_loss(self, pred, gt):
        loss = 0.0
        cur_p, cur_g = pred, gt
        for w in [0.4, 0.3, 0.2, 0.1]:
            blur_p = F.avg_pool2d(F.pad(cur_p, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
            blur_g = F.avg_pool2d(F.pad(cur_g, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
            loss = loss + w * F.l1_loss(cur_p - blur_p, cur_g - blur_g)
            cur_p = F.avg_pool2d(cur_p, 2, stride=2)
            cur_g = F.avg_pool2d(cur_g, 2, stride=2)
        return loss

    def forward(self, pred_ink, gt_ink):
        if not self.use_vgg:
            return self._lap_pyr_loss(pred_ink, gt_ink)

        p_rgb = (1.0 - pred_ink).repeat(1, 3, 1, 1)
        g_rgb = (1.0 - gt_ink).repeat(1, 3, 1, 1)
        p_norm = (p_rgb - self.mean) / self.std
        g_norm = (g_rgb - self.mean) / self.std

        weights = [0.20, 0.30, 0.30, 0.20]
        total = 0.0
        h_p, h_g = p_norm, g_norm
        for blk, w in zip([self.slice1, self.slice2, self.slice3, self.slice4], weights):
            h_p = blk(h_p)
            with torch.no_grad():
                h_g = blk(h_g)
            total = total + w * F.l1_loss(h_p, h_g)
        return total


class AdaptiveLineartLoss(nn.Module):
    """
    HÃ m má»¥c tiÃªu 8-term chá»ng dÃ y nÃ©t vÃ  báº£o toÃ n Äá» máº£nh:
      - 1.00*Charbonnier
      - 0.35*Symmetric-DT (steps=8, kháº£ vi 2 chiá»u)
      - 0.20*Sobel Edge Gradient
      - 0.50*Exact-STE Density Ratio (deadzone [0.96, 1.04])
      - 0.25*Bidirectional Stroke Width (core protection + ring overspill penalty)
      - 0.20*Ghost & Cleanliness (far background + near halo)
      - 0.20*Soft-clDice (cÃ¢n báº±ng alpha=0.5, beta=0.5)
      - 0.15*VGG16 Perceptual (chá» á» LONG mode)
    Há» trá»£ cáº¯t biÃªn chÃ­nh xÃ¡c (valid_hw) Äá» khÃ´ng tÃ­nh loss lÃªn pháº§n pad biÃªn pháº£i/dÆ°á»i!
    """
    def __init__(self, train_mode="LONG", is_rank0=True, w_ghost=0.45, w_temp=0.50):
        super().__init__()
        self.train_mode = train_mode.upper()
        self.dt_steps = 8
        self.w_charb = 1.0
        self.w_dt = 0.35
        self.w_sobel = 0.20
        self.w_dens = 0.50
        self.w_width = 0.25
        self.w_ghost = float(w_ghost)

        if self.train_mode == "SHORT":
            self.w_cldice = 0.0
            self.w_perc = 0.0
            self.w_temp = 0.0
        elif self.train_mode == "MEDIUM":
            self.w_cldice = 0.20
            self.w_perc = 0.0
            self.w_temp = float(w_temp) * 0.7
        else:
            self.w_cldice = 0.20
            self.w_perc = 0.15
            self.w_temp = float(w_temp)

        self.density_loss = ExactSTEDensityLoss(tau=0.15, dead_zone_low=0.96, dead_zone_high=1.04)
        self.stroke_width_loss = BidirectionalStrokeWidthLoss()
        self.ghost_loss = GhostAndBackgroundCleanlinessLoss()
        self.cldice = SoftClDiceLoss(iters=2) if self.w_cldice > 0 else None
        self.perceptual = VGGPerceptualOrLaplacianLoss(is_rank0=is_rank0) if self.w_perc > 0 else None

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, gt, valid_hw=None):
        if valid_hw is not None:
            if torch.all(valid_hw == valid_hw[0:1]):
                hi, wi = int(valid_hw[0, 0].item()), int(valid_hw[0, 1].item())
                return self._compute_single(pred[:, :, :hi, :wi], gt[:, :, :hi, :wi])
            total_l = 0.0
            details_acc = {
                'loss_charb': 0.0, 'loss_dt': 0.0, 'loss_sobel': 0.0,
                'loss_dens': 0.0, 'loss_width': 0.0, 'loss_ghost': 0.0,
                'loss_cldice': 0.0, 'loss_perc': 0.0, 'total_loss': 0.0
            }
            b = pred.shape[0]
            for i in range(b):
                hi, wi = int(valid_hw[i, 0].item()), int(valid_hw[i, 1].item())
                p_i = pred[i:i + 1, :, :hi, :wi]
                g_i = gt[i:i + 1, :, :hi, :wi]
                l_i, d_i = self._compute_single(p_i, g_i)
                total_l = total_l + l_i / float(b)
                for k in details_acc:
                    if k in d_i:
                        details_acc[k] += d_i[k] / float(b)
            details_acc['total_loss'] = float(total_l.detach().item())
            return total_l, details_acc
        return self._compute_single(pred, gt)

    def _compute_single(self, pred, gt):
        # 1. Charbonnier
        loss_charb = torch.mean(torch.sqrt((pred - gt) ** 2 + 1e-6))

        # 2. Symmetric Differentiable Distance Transform Loss (kháº£ vi 2 chiá»u)
        pred_soft = torch.clamp((pred - 0.06) / 0.18, 0.0, 1.0)
        dt_p = approx_gpu_distance_map(pred_soft, steps=self.dt_steps)
        with torch.no_grad():
            gt_bin = (gt > 0.15).float()
            dt_g = approx_gpu_distance_map(gt_bin, steps=self.dt_steps)
            if self.cldice is not None:
                skel_gt = soft_skeletonize(gt, iters=2)

        loss_dt = F.l1_loss(dt_p, dt_g) + 0.5 * torch.mean(pred * dt_g)

        # 3. Sobel Edge Gradient
        gx_p = F.conv2d(pred, self.sobel_x, padding=1)
        gy_p = F.conv2d(pred, self.sobel_y, padding=1)
        gx_g = F.conv2d(gt, self.sobel_x, padding=1)
        gy_g = F.conv2d(gt, self.sobel_y, padding=1)
        loss_sobel = F.l1_loss(gx_p, gx_g) + F.l1_loss(gy_p, gy_g)

        # 4. Anti-Thickening Terms
        loss_dens = self.density_loss(pred, gt)
        loss_width = self.stroke_width_loss(pred, gt)
        loss_ghost = self.ghost_loss(pred, gt)

        # 5. Soft-clDice
        loss_cldice = self.cldice(pred, gt, t_skel=skel_gt) if self.cldice is not None else pred.new_zeros(())

        # 6. Perceptual / VGG
        loss_perc = self.perceptual(pred, gt) if self.perceptual is not None else pred.new_zeros(())

        total_loss = (
            self.w_charb * loss_charb
            + self.w_dt * loss_dt
            + self.w_sobel * loss_sobel
            + self.w_dens * loss_dens
            + self.w_width * loss_width
            + self.w_ghost * loss_ghost
            + self.w_cldice * loss_cldice
            + self.w_perc * loss_perc
        )
        return total_loss, {
            'loss_charb': float(loss_charb.detach().item()),
            'loss_dt': float(loss_dt.detach().item()),
            'loss_sobel': float(loss_sobel.detach().item()),
            'loss_dens': float(loss_dens.detach().item()),
            'loss_width': float(loss_width.detach().item()),
            'loss_ghost': float(loss_ghost.detach().item()),
            'loss_cldice': float(loss_cldice.detach().item()),
            'loss_perc': float(loss_perc.detach().item()),
            'total_loss': float(total_loss.detach().item()),
        }

    def compute_temporal_pair_loss(self, pred_a, pred_b, gt_a, gt_b, valid_hw=None):
        """
        RÃ ng buá»c nháº¥t quÃ¡n thá»i gian (2C) giá»¯a 2 khung hÃ¬nh liÃªn tiáº¿p (k, k+1) trong cÃ¹ng chuá»i:
          - Pháº¡t sai phÃ¢n chuyá»n Äá»ng (pred_b - pred_a) lá»ch khá»i (gt_b - gt_a).
          - Pháº¡t nháº¥p nhÃ¡y/rung sÆ°Æ¡ng trÃªn vÃ¹ng cÃ¹ng lÃ  ná»n tráº¯ng cá»§a 2 frame liÃªn tiáº¿p.
        """
        if self.w_temp <= 0.0 or pred_a.shape[0] == 0:
            return pred_a.new_zeros(())

        total = pred_a.new_zeros(())
        n = pred_a.shape[0]
        for i in range(n):
            if valid_hw is not None:
                hi, wi = int(valid_hw[i, 0].item()), int(valid_hw[i, 1].item())
                pa, pb = pred_a[i:i + 1, :, :hi, :wi], pred_b[i:i + 1, :, :hi, :wi]
                ga, gb = gt_a[i:i + 1, :, :hi, :wi], gt_b[i:i + 1, :, :hi, :wi]
            else:
                pa, pb = pred_a[i:i + 1], pred_b[i:i + 1]
                ga, gb = gt_a[i:i + 1], gt_b[i:i + 1]

            dp = pb - pa
            dg = gb - ga
            loss_tdiff = torch.mean(torch.sqrt((dp - dg) ** 2 + 1e-6))

            with torch.no_grad():
                both_bg = ((ga < 0.04) & (gb < 0.04)).float()
                gt_area = torch.clamp(
                    0.5 * (torch.sum((ga > 0.15).float()) + torch.sum((gb > 0.15).float())),
                    min=500.0
                )
            loss_tflicker = torch.sum(torch.abs(dp) * both_bg) / gt_area
            total = total + (loss_tdiff + 0.5 * loss_tflicker)
        return self.w_temp * total


# ==============================================================================
# 4. Há» THá»NG ÄO LÆ¯á»NG CHá» Sá» KHÃCH QUAN KÃP (DUAL-REGIME ZERO-BIAS METRICS)
# ==============================================================================
def compute_wcd_batch_gpu(p_bin, g_bin):
    """
    Weighted Chamfer Distance trÃªn máº·t náº¡ nhá» phÃ¢n [B, 1, H, W].
    CÃ´ng thá»©c chuáº©n hÃ³a Paper Eq. (15):
      h_weight = sigmoid(|area_p - area_g| / min(area_p, area_g))
    Pháº¡t náº·ng khi máº¥t cÃ¢n Äá»i diá»n tÃ­ch nÃ©t váº½ (chá»ng dÃ y nÃ©t).
    Äá»ng thá»i tÃ­nh `wcd_legacy` (cÃ´ng thá»©c cÅ©) Äá» Äá»i chiáº¿u lá»ch sá»­.
    """
    dist_g = approx_gpu_distance_map(g_bin, steps=10)
    dist_p = approx_gpu_distance_map(p_bin, steps=10)
    d_p_to_g = torch.sum(p_bin * F.softplus(dist_g * 10.0), dim=(-2, -1)) / torch.clamp(torch.sum(p_bin, dim=(-2, -1)), min=1.0)
    d_g_to_p = torch.sum(g_bin * F.softplus(dist_p * 10.0), dim=(-2, -1)) / torch.clamp(torch.sum(g_bin, dim=(-2, -1)), min=1.0)
    area_p = torch.sum(p_bin, dim=(-2, -1))
    area_g = torch.sum(g_bin, dim=(-2, -1))

    # Chuáº©n Paper Eq. (15): sigmoid(|Ap - Ag| / min(Ap, Ag))
    diff_rel = torch.abs(area_p - area_g) / torch.clamp(torch.minimum(area_p, area_g), min=1.0)
    h_weight = torch.sigmoid(diff_rel)
    wcd = h_weight * (d_p_to_g + d_g_to_p)

    # Legacy: 0.5 * (1 + min/max)
    h_weight_legacy = 0.5 * (1.0 + torch.minimum(area_p, area_g) / torch.clamp(torch.maximum(area_p, area_g), min=1.0))
    wcd_legacy = h_weight_legacy * (d_p_to_g + d_g_to_p)

    return wcd, wcd_legacy


def compute_semd_batch_gpu(p_ink, g_ink):
    """1D-Sliced Wasserstein EMD trÃªn phÃ¢n bá» nÃ©t váº½ dá»c hai trá»¥c X/Y."""
    p_norm = p_ink / torch.clamp(torch.sum(p_ink, dim=(-2, -1), keepdim=True), min=1e-6)
    g_norm = g_ink / torch.clamp(torch.sum(g_ink, dim=(-2, -1), keepdim=True), min=1e-6)
    emd_h = torch.mean(torch.abs(torch.cumsum(torch.sum(p_norm, dim=-1), dim=-1) - torch.cumsum(torch.sum(g_norm, dim=-1), dim=-1)), dim=(-2, -1))
    emd_w = torch.mean(torch.abs(torch.cumsum(torch.sum(p_norm, dim=-2), dim=-1) - torch.cumsum(torch.sum(g_norm, dim=-2), dim=-1)), dim=(-2, -1))
    return 0.5 * (emd_h + emd_w)


_DOG_KERNEL_CACHE = {}


def _gaussian_blur_2d(img, sigma):
    key = (round(float(sigma), 4), img.device, img.dtype)
    k2d = _DOG_KERNEL_CACHE.get(key)
    if k2d is None:
        rad = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-rad, rad + 1, device=img.device, dtype=torch.float32)
        g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g1 = g1 / g1.sum()
        k2d = torch.outer(g1, g1).view(1, 1, 2 * rad + 1, 2 * rad + 1).to(img.dtype)
        _DOG_KERNEL_CACHE[key] = k2d
    rad = k2d.shape[-1] // 2
    return F.conv2d(F.pad(img, (rad, rad, rad, rad), mode='replicate'), k2d)


def compute_dog_bin_mask(ink_img, sigma=1.0, k=1.6, tau=0.02):
    """
    Bá» lá»c Difference-of-Gaussians (DoG) trÃ­ch xuáº¥t nÃ©t thuáº§n tÃºy,
    triá»t tiÃªu hoÃ n toÃ n Äá» xÃ¡m ná»n (ná»n xÃ¡m nháº¡t vs ná»n tráº¯ng tinh) Äá» ÄÃ¡nh giÃ¡ cÃ´ng báº±ng M1 vs M3.
    """
    gray = 1.0 - torch.clamp(ink_img, 0.0, 1.0)
    g1 = _gaussian_blur_2d(gray, sigma)
    g2 = _gaussian_blur_2d(gray, sigma * k)
    dog = g1 - g2
    return (dog < -tau).float()


@torch.no_grad()
def evaluate_metrics_dual_regime(pred_ink, gt_ink):
    """
    Äo lÆ°á»ng 10+ chá» sá» khÃ¡ch quan trÃªn cÃ¹ng 1 cáº·p áº£nh (pred_ink, gt_ink) shape [B, 1, H, W]:
      - Regime A (Raw Continuous): PSNR, SSIM, IE_MAE (0..255), PWCR, RGD, Ghost_Px, Dark_Ghost_Px, Val_L1
      - Regime B (Thresholded tau=0.15 & DoG): WCD (Eq.15), WCD_DoG, WCD_Legacy, WCD_DoG_Legacy,
                                               S-EMD, Skeleton Recall, Hair-Tip Recall,
                                               Stroke Recall, Stroke Precision, Stroke F1, Density Ratio
    """
    p = torch.clamp(pred_ink.float(), 0.0, 1.0)
    g = torch.clamp(gt_ink.float(), 0.0, 1.0)

    # ---- REGIME A: RAW CONTINUOUS ----
    mse = torch.mean((p - g) ** 2, dim=(-3, -2, -1))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

    mu_p = F.avg_pool2d(p, 11, stride=1, padding=5)
    mu_g = F.avg_pool2d(g, 11, stride=1, padding=5)
    sig_p = F.avg_pool2d(p * p, 11, stride=1, padding=5) - mu_p ** 2
    sig_g = F.avg_pool2d(g * g, 11, stride=1, padding=5) - mu_g ** 2
    sig_pg = F.avg_pool2d(p * g, 11, stride=1, padding=5) - mu_p * mu_g
    ssim_map = ((2 * mu_p * mu_g + 1e-4) * (2 * sig_pg + 9e-4)) / ((mu_p ** 2 + mu_g ** 2 + 1e-4) * (sig_p + sig_g + 9e-4))
    ssim = torch.mean(ssim_map, dim=(-3, -2, -1))

    val_l1 = torch.mean(torch.abs(p - g), dim=(-3, -2, -1))
    ie_mae = val_l1 * 255.0

    # Pixel brightness domain [0, 255]
    p_255 = (1.0 - p) * 255.0
    g_255 = (1.0 - g) * 255.0
    pwcr = torch.mean((p_255 >= 250.0).float(), dim=(-3, -2, -1))
    rgd = torch.mean(((p_255 >= 180.0) & (p_255 < 250.0)).float(), dim=(-3, -2, -1))

    gt_white_safe = (g_255 > 250.0).float()
    gt_stroke_safe = soft_dilate(soft_dilate((g > 0.08).float()))
    outside_gt = (1.0 - gt_stroke_safe) * gt_white_safe
    ghost_px = torch.sum(((p_255 >= 180.0) & (p_255 <= 250.0)).float() * gt_white_safe, dim=(-3, -2, -1))
    dark_ghost_px = torch.sum((p > 0.35).float() * (1.0 - gt_stroke_safe), dim=(-3, -2, -1))

    # ---- REGIME B: BINARIZED TOPOLOGY (TH=0.15 + DoG) ----
    p_bin = (p > 0.15).float()
    g_bin = (g > 0.15).float()
    p_dog = compute_dog_bin_mask(p)
    g_dog = compute_dog_bin_mask(g)

    wcd, wcd_legacy = compute_wcd_batch_gpu(p_bin, g_bin)
    wcd_dog, wcd_dog_legacy = compute_wcd_batch_gpu(p_dog, g_dog)
    semd = compute_semd_batch_gpu(p, g)

    # Stroke Recall / Precision / F1 (dung sai 2px dilation cho nÃ©t máº£nh 1px)
    g_dil = soft_dilate(soft_dilate(g_bin))
    p_dil = soft_dilate(soft_dilate(p_bin))
    tp_r = torch.sum(g_bin * p_dil, dim=(-3, -2, -1))
    tp_p = torch.sum(p_bin * g_dil, dim=(-3, -2, -1))
    cnt_p = torch.sum(p_bin, dim=(-3, -2, -1))
    cnt_g = torch.sum(g_bin, dim=(-3, -2, -1))
    recall = tp_r / torch.clamp(cnt_g, min=1.0)
    precision = tp_p / torch.clamp(cnt_p, min=1.0)
    f1 = 2.0 * recall * precision / torch.clamp(recall + precision, min=1e-6)
    density_ratio = cnt_p / torch.clamp(cnt_g, min=1.0)

    # Skeleton Recall & Hair-Tip Recall
    skel_g = soft_skeletonize(g, iters=2)
    skel_mask = (skel_g > 0.20).float()
    p_dil1 = soft_dilate(p_bin)
    skel_recall = torch.sum(p_dil1 * skel_mask, dim=(-3, -2, -1)) / torch.clamp(torch.sum(skel_mask, dim=(-3, -2, -1)), min=1.0)

    nbrs = F.avg_pool2d(skel_g, kernel_size=3, stride=1, padding=1) * 9.0 - skel_g
    endpoints = ((nbrs <= 1.8) & (skel_g > 0.20)).float()
    tip_recall = torch.sum(p_dil1 * endpoints, dim=(-3, -2, -1)) / torch.clamp(torch.sum(endpoints, dim=(-3, -2, -1)), min=1.0)

    return {
        'psnr': psnr.squeeze(-1) if psnr.ndim > 1 else psnr,
        'ssim': ssim.squeeze(-1) if ssim.ndim > 1 else ssim,
        'val_l1': val_l1.squeeze(-1) if val_l1.ndim > 1 else val_l1,
        'ie_mae': ie_mae.squeeze(-1) if ie_mae.ndim > 1 else ie_mae,
        'pwcr': pwcr.squeeze(-1) if pwcr.ndim > 1 else pwcr,
        'rgd': rgd.squeeze(-1) if rgd.ndim > 1 else rgd,
        'ghost': ghost_px.squeeze(-1) if ghost_px.ndim > 1 else ghost_px,
        'dark_ghost': dark_ghost_px.squeeze(-1) if dark_ghost_px.ndim > 1 else dark_ghost_px,
        'wcd': wcd.squeeze(-1) if wcd.ndim > 1 else wcd,
        'wcd_legacy': wcd_legacy.squeeze(-1) if wcd_legacy.ndim > 1 else wcd_legacy,
        'wcd_dog': wcd_dog.squeeze(-1) if wcd_dog.ndim > 1 else wcd_dog,
        'wcd_dog_legacy': wcd_dog_legacy.squeeze(-1) if wcd_dog_legacy.ndim > 1 else wcd_dog_legacy,
        'semd': semd.squeeze(-1) if semd.ndim > 1 else semd,
        'recall': recall.squeeze(-1) if recall.ndim > 1 else recall,
        'precision': precision.squeeze(-1) if precision.ndim > 1 else precision,
        'f1': f1.squeeze(-1) if f1.ndim > 1 else f1,
        'density_ratio': density_ratio.squeeze(-1) if density_ratio.ndim > 1 else density_ratio,
        'skel_recall': skel_recall.squeeze(-1) if skel_recall.ndim > 1 else skel_recall,
        'tip_recall': tip_recall.squeeze(-1) if tip_recall.ndim > 1 else tip_recall,
    }


# ==============================================================================
# 5. DATASET CHUá»I SEQUENCE ÄA KHUNG HÃNH N={3,5,7} & DDP BUCKET SAMPLER
# ==============================================================================
def pad_to_canvas(arr_2d, target_h, target_w):
    """Äá»m áº£nh ink [H, W] báº±ng ná»n tráº¯ng (0.0) á» ÄÃ¡y/pháº£i tá»i (target_h, target_w)."""
    h, w = arr_2d.shape
    if h == target_h and w == target_w:
        return arr_2d
    out = np.zeros((target_h, target_w), dtype=arr_2d.dtype)
    out[:h, :w] = arr_2d
    return out


def _generate_synthetic_geometric_sequence(seq_len, orig_h=540, orig_w=960, max_pts=128, lambd=0.01):
    """
    Sinh Äá»ng chuá»i hÃ¬nh há»c nÃ©t máº£nh 540p kÃ¨m Äiá»m Äiá»u khiá»n Lagrangian Direct TPS:
      - HÃ¬nh trÃ²n / elip dá»ch chuyá»n xa (60..420 px) & co giÃ£n kÃ­ch thÆ°á»c (0.40x..2.25x)
      - HÃ¬nh trÃ²n bay + ÄÆ°á»ng ná»n / há»p tÄ©nh
      - Cung cong xoay & tá»nh tiáº¿n
      - Cáº·p 2 hÃ¬nh trÃ²n chuyá»n Äá»ng Äá»c láº­p
    """
    def _ellipse_pts(cx, cy, rx, ry, rot_rad, n_pts=32):
        ang = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
        lx = rx * np.cos(ang)
        ly = ry * np.sin(ang)
        ca, sa = math.cos(rot_rad), math.sin(rot_rad)
        px = cx + lx * ca - ly * sa
        py = cy + lx * sa + ly * ca
        pts = np.stack([px / float(orig_w - 1), py / float(orig_h - 1)], axis=1)
        return pts.astype(np.float32)

    mode_roll = random.random()
    frames = []
    kps_per_t = []
    t_list = [float(k) / float(seq_len - 1) for k in range(seq_len)]

    if mode_roll < 0.65:
        # Mode 1 (40%) & Mode 2 (25%): Shrinking/expanding circle/oval (+ optional static ground/box)
        has_static = (mode_roll >= 0.40)
        y_max_c = 340 if has_static else 420
        rx0 = random.uniform(26.0, 92.0)
        ry0 = rx0 * random.uniform(0.75, 1.25)
        scale_r = random.uniform(0.42, 2.20)
        rx1 = float(np.clip(rx0 * scale_r, 18.0, 96.0))
        ry1 = float(np.clip(ry0 * scale_r * random.uniform(0.85, 1.15), 18.0, 96.0))
        cx0 = random.uniform(110.0, 420.0)
        cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(80.0, 420.0), 100.0, orig_w - 100.0))
        if random.random() < 0.5:
            cx0, cx1 = cx1, cx0
        cy0 = random.uniform(115.0, y_max_c - 60.0)
        cy1 = float(np.clip(cy0 + random.uniform(-90.0, 90.0), 105.0, y_max_c - 40.0))
        rot0 = random.uniform(-0.35, 0.35)
        rot1 = rot0 + random.uniform(-0.35, 0.35)
        th0 = random.uniform(2.0, 5.0)
        th1 = random.uniform(2.0, 5.0)

        # Static element (ground line or static rectangle)
        st_type = random.choice(["line", "rect"])
        st_y = random.uniform(410.0, 490.0)
        st_x0, st_x1 = random.uniform(80.0, 200.0), random.uniform(orig_w - 200.0, orig_w - 80.0)
        st_th = int(round(random.uniform(2.5, 4.5)))
        if st_type == "line":
            xs_st = np.linspace(st_x0, st_x1, 24)
            pts_st = np.stack([xs_st / float(orig_w - 1), np.full_like(xs_st, st_y) / float(orig_h - 1)], axis=1).astype(np.float32)
        else:
            bx0, by0 = random.uniform(120.0, 740.0), st_y - 45.0
            bw, bh = random.uniform(70.0, 140.0), random.uniform(40.0, 65.0)
            pts_st = np.vstack([
                np.stack([np.linspace(bx0, bx0 + bw, 7)[:-1], np.full(6, by0)], axis=1),
                np.stack([np.full(6, bx0 + bw), np.linspace(by0, by0 + bh, 7)[:-1]], axis=1),
                np.stack([np.linspace(bx0 + bw, bx0, 7)[:-1], np.full(6, by0 + bh)], axis=1),
                np.stack([np.full(6, bx0), np.linspace(by0 + bh, by0, 7)[:-1]], axis=1),
            ])
            pts_st[:, 0] /= float(orig_w - 1)
            pts_st[:, 1] /= float(orig_h - 1)
            pts_st = pts_st.astype(np.float32)

        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            cx = (1.0 - t) * cx0 + t * cx1
            cy = (1.0 - t) * cy0 + t * cy1
            rx = (1.0 - t) * rx0 + t * rx1
            ry = (1.0 - t) * ry0 + t * ry1
            rot = (1.0 - t) * rot0 + t * rot1
            th = max(2, int(round((1.0 - t) * th0 + t * th1)))
            if cv2 is not None:
                cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), math.degrees(rot), 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                if has_static:
                    if st_type == "line":
                        cv2.line(im, (int(st_x0), int(st_y)), (int(st_x1), int(st_y)), 1.0, st_th, lineType=cv2.LINE_AA)
                    else:
                        cv2.rectangle(im, (int(bx0), int(by0)), (int(bx0 + bw), int(by0 + bh)), 1.0, st_th, lineType=cv2.LINE_AA)
            pts_mv = _ellipse_pts(cx, cy, rx, ry, rot, n_pts=32)
            pts_all = np.vstack([pts_mv, pts_st]) if has_static else pts_mv
            frames.append(im)
            kps_per_t.append(pts_all)
    elif mode_roll < 0.85:
        # Mode 3 (20%): Smooth curved arc / open stroke translating + rotating + scaling
        n_ctrl = 24
        s_u = np.linspace(-1.0, 1.0, n_ctrl)
        base_len = random.uniform(60.0, 140.0)
        bend0 = random.uniform(-55.0, 55.0)
        bend1 = random.uniform(-55.0, 55.0)
        cx0, cy0 = random.uniform(160.0, 440.0), random.uniform(140.0, 400.0)
        cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(60.0, 320.0), 140.0, orig_w - 140.0))
        cy1 = float(np.clip(cy0 + random.uniform(-100.0, 100.0), 120.0, orig_h - 120.0))
        sc0, sc1 = random.uniform(0.75, 1.25), random.uniform(0.65, 1.45)
        ang0 = random.uniform(-0.7, 0.7)
        ang1 = ang0 + random.uniform(-0.6, 0.6)
        th = int(round(random.uniform(2.0, 4.5)))

        def _arc_px(t):
            cx = (1.0 - t) * cx0 + t * cx1
            cy = (1.0 - t) * cy0 + t * cy1
            sc = (1.0 - t) * sc0 + t * sc1
            ang = (1.0 - t) * ang0 + t * ang1
            bend = (1.0 - t) * bend0 + t * bend1
            lx = s_u * base_len * sc
            ly = bend * sc * (1.0 - s_u ** 2)
            ca, sa = math.cos(ang), math.sin(ang)
            return np.stack([cx + lx * ca - ly * sa, cy + lx * sa + ly * ca], axis=1)

        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            p_px = _arc_px(t)
            if cv2 is not None:
                cv2.polylines(im, [np.round(p_px).astype(np.int32)], False, 1.0, th, lineType=cv2.LINE_AA)
            p_norm = np.stack([p_px[:, 0] / float(orig_w - 1), p_px[:, 1] / float(orig_h - 1)], axis=1).astype(np.float32)
            frames.append(im)
            kps_per_t.append(p_norm)
    else:
        # Mode 4 (15%): Two independent moving circles/ovals in upper & lower lanes
        lanes = [(145.0, 40.0), (385.0, 40.0)]
        params = []
        for cy_base, cy_jit in lanes:
            rx0 = random.uniform(24.0, 62.0)
            ry0 = rx0 * random.uniform(0.8, 1.2)
            sc = random.uniform(0.55, 1.80)
            rx1 = float(np.clip(rx0 * sc, 18.0, 72.0))
            ry1 = float(np.clip(ry0 * sc, 18.0, 72.0))
            cx0 = random.uniform(120.0, 450.0)
            cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(80.0, 350.0), 100.0, orig_w - 100.0))
            cy0 = cy_base + random.uniform(-cy_jit, cy_jit)
            cy1 = cy_base + random.uniform(-cy_jit, cy_jit)
            params.append((cx0, cy0, rx0, ry0, cx1, cy1, rx1, ry1))
        th = int(round(random.uniform(2.5, 4.5)))
        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            pts_two = []
            for (cx0, cy0, rx0, ry0, cx1, cy1, rx1, ry1) in params:
                cx = (1.0 - t) * cx0 + t * cx1
                cy = (1.0 - t) * cy0 + t * cy1
                rx = (1.0 - t) * rx0 + t * rx1
                ry = (1.0 - t) * ry0 + t * ry1
                if cv2 is not None:
                    cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                pts_two.append(_ellipse_pts(cx, cy, rx, ry, 0.0, n_pts=24))
            frames.append(im)
            kps_per_t.append(np.vstack(pts_two))

    x0 = frames[0]
    x1 = frames[-1]
    xk_all = np.stack(frames[1:-1], axis=0)
    kps0 = kps_per_t[0]
    kps1 = kps_per_t[-1]
    t_mid = np.array(t_list[1:-1], dtype=np.float32)

    m_steps = seq_len - 2
    c_t0_all = np.zeros((m_steps, max_pts, 2), dtype=np.float32)
    th_t0_all = np.zeros((m_steps, max_pts + 3, 2), dtype=np.float32)
    c_t1_all = np.zeros((m_steps, max_pts, 2), dtype=np.float32)
    th_t1_all = np.zeros((m_steps, max_pts + 3, 2), dtype=np.float32)
    valid_all = np.zeros((m_steps,), dtype=np.float32)

    for m_idx in range(m_steps):
        kpt = kps_per_t[m_idx + 1]
        c_t0, th_t0, v0 = solve_tps_direct(kpt, kps0, max_pts=max_pts, lambd=lambd, use_anchors=False)
        c_t1, th_t1, v1 = solve_tps_direct(kpt, kps1, max_pts=max_pts, lambd=lambd, use_anchors=False)
        c_t0_all[m_idx] = c_t0
        th_t0_all[m_idx] = th_t0
        c_t1_all[m_idx] = c_t1
        th_t1_all[m_idx] = th_t1
        valid_all[m_idx] = 1.0 if (v0 > 0.5 and v1 > 0.5) else 0.0

    return {
        'is_sequence': True,
        'is_direct_t_flows': True,
        'x0': torch.from_numpy(x0).unsqueeze(0),
        'x1': torch.from_numpy(x1).unsqueeze(0),
        'xk_all': torch.from_numpy(xk_all).unsqueeze(1),
        't_vals': torch.from_numpy(t_mid),
        'c_t0_all': torch.from_numpy(c_t0_all),
        'th_t0_all': torch.from_numpy(th_t0_all),
        'c_t1_all': torch.from_numpy(c_t1_all),
        'th_t1_all': torch.from_numpy(th_t1_all),
        'valid_all': torch.from_numpy(valid_all),
        'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
        'seq_len': seq_len,
        'clip_id': f"synth_N{seq_len}",
        'scene_id': "synth_geom",
        'mean_disp': 120.0,
    }


class SakugaSequenceDataset(Dataset):
    """
    Dataset Sequence cho cáº£ 3 cháº¿ Äá» (SHORT / MEDIUM / LONG) vÃ  Validation:
      - `mode == 'LONG'`: tráº£ vá» NGUYÃN CHUá»I (x0, x1, xk_all[M], t_vals[M]) á» Äá» phÃ¢n giáº£i gá»c 540p,
        kÃ¨m tÃ¹y chá»n trá»n cÃ¡c chuá»i hÃ¬nh há»c tá»ng há»£p (`synth_count`) dÃ¹ng Lagrangian Direct TPS.
      - `mode == 'SHORT'`: má»i epoch láº¥y 1 frame trung gian ngáº«u nhiÃªn/chuá»i + Ink Crop 384x384.
      - `mode == 'MEDIUM'`: má» rá»ng thÃ nh toÃ n bá» 3,169 máº«u ÄÆ¡n + Motion-BBox Crop 512x512.
    """
    def __init__(self, metadata_csv, clips_dir, tps_cache_path, train_mode="LONG", is_train=True, seed=42, synth_count=0):
        super().__init__()
        self.clips_dir = clips_dir
        self.train_mode = train_mode.upper()
        self.is_train = is_train
        self.df = pd.read_csv(metadata_csv).reset_index(drop=True)

        # Náº¡p nháº¹ toÃ n bá» máº£ng TPS (~7 MB) vÃ o RAM vÃ  ÄÃ³ng ngay file .npz Äá» an toÃ n tuyá»t Äá»i khi fork num_workers>0
        self.tps_cache_path = tps_cache_path
        with np.load(tps_cache_path) as cache_data:
            self.c0_arr = np.array(cache_data['c0'])
            self.th01_arr = np.array(cache_data['th01'])
            self.c1_arr = np.array(cache_data['c1'])
            self.th10_arr = np.array(cache_data['th10'])
            self.valid_arr = np.array(cache_data['valid'])
            self.orig_hw_arr = np.array(cache_data['orig_hw'])

        self.samples = []
        if self.is_train and self.train_mode == "LONG":
            # 1 item = toÃ n bá» 1 sequence (x0, x1, toÃ n bá» N-2 frames trung gian)
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                t_list = [float(k) / float(seq_len - 1) for k in range(1, seq_len - 1)]
                self.samples.append({
                    'is_synth': False,
                    'clip_id': cid,
                    'scene_id': str(row.get('scene_id', '')),
                    'seq_len': seq_len,
                    'cache_idx': cache_idx,
                    'orig_h': int(hw[0]),
                    'orig_w': int(hw[1]),
                    't_list': t_list,
                    'mean_disp': float(row.get('mean_disp', 0.0)),
                })
            # Bá» sung chuá»i hÃ¬nh há»c tá»ng há»£p phÃ¢n bá» Äá»u cho N={3, 5, 7}
            n_synth = int(synth_count) if (cv2 is not None and synth_count > 0) else 0
            for s_i in range(n_synth):
                sl = (3, 5, 7)[s_i % 3]
                self.samples.append({
                    'is_synth': True,
                    'clip_id': f"synth_{s_i:04d}_N{sl}",
                    'scene_id': "synth_geom",
                    'seq_len': sl,
                    'cache_idx': -1,
                    'orig_h': 540,
                    'orig_w': 960,
                    't_list': [float(k) / float(sl - 1) for k in range(1, sl - 1)],
                    'mean_disp': 120.0,
                })
        elif self.is_train and self.train_mode == "SHORT":
            # 1 item = 1 chuá»i (tá»± chá»n 1 frame ngáº«u nhiÃªn má»i epoch trong __getitem__)
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                self.samples.append({
                    'is_synth': False,
                    'clip_id': cid,
                    'scene_id': str(row.get('scene_id', '')),
                    'seq_len': seq_len,
                    'cache_idx': cache_idx,
                    'orig_h': int(hw[0]),
                    'orig_w': int(hw[1]),
                    'frame_k': None,
                    'mean_disp': float(row.get('mean_disp', 0.0)),
                })
        else:
            # Validation hoáº·c MEDIUM Train: má» rá»ng thÃ nh má»i frame trung gian (t_k, I_k)
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                for k in range(1, seq_len - 1):
                    self.samples.append({
                        'is_synth': False,
                        'clip_id': cid,
                        'scene_id': str(row.get('scene_id', '')),
                        'seq_len': seq_len,
                        'cache_idx': cache_idx,
                        'orig_h': int(hw[0]),
                        'orig_w': int(hw[1]),
                        'frame_k': k + 1,  # 1-based file index: 000002..00000(N-1)
                        't_val': float(k) / float(seq_len - 1),
                        'mean_disp': float(row.get('mean_disp', 0.0)),
                    })

    def __len__(self):
        return len(self.samples)

    def _read_ink(self, clip_id, frame_idx_1b):
        p = os.path.join(self.clips_dir, clip_id, f"{frame_idx_1b:06d}.png")
        with Image.open(p) as im:
            arr = np.asarray(im.convert('L'), dtype=np.float32)
        return (255.0 - arr) / 255.0

    def __getitem__(self, idx):
        s = self.samples[idx]
        if s.get('is_synth', False):
            return _generate_synthetic_geometric_sequence(s['seq_len'], orig_h=s['orig_h'], orig_w=s['orig_w'])

        cid = s['clip_id']
        seq_len = s['seq_len']
        cache_idx = s['cache_idx']

        x0 = self._read_ink(cid, 1)
        x1 = self._read_ink(cid, seq_len)

        c0 = np.array(self.c0_arr[cache_idx], copy=True)
        th01 = np.array(self.th01_arr[cache_idx], copy=True)
        c1 = np.array(self.c1_arr[cache_idx], copy=True)
        th10 = np.array(self.th10_arr[cache_idx], copy=True)
        valid = float(self.valid_arr[cache_idx])
        orig_h, orig_w = x0.shape

        if self.is_train and self.train_mode == "LONG":
            # Táº£i Äá»ng thá»i toÃ n bá» N-2 khung hÃ¬nh trung gian cá»§a chuá»i
            xk_list = [self._read_ink(cid, f_idx) for f_idx in range(2, seq_len)]
            xk_all = np.stack(xk_list, axis=0)  # [M, H, W]
            t_arr = np.array(s['t_list'], dtype=np.float32)

            # Data Augmentation: Temporal Reversal (Äáº£o chiá»u I0 <-> I1, xk[::-1], t <-> 1-t)
            if random.random() < 0.5:
                x0, x1 = x1, x0
                c0, c1 = c1, c0
                th01, th10 = th10, th01
                xk_all = xk_all[::-1].copy()
                t_arr = (1.0 - t_arr[::-1]).copy()

            return {
                'is_sequence': True,
                'is_direct_t_flows': False,
                'x0': torch.from_numpy(x0).unsqueeze(0),               # [1, H, W]
                'x1': torch.from_numpy(x1).unsqueeze(0),               # [1, H, W]
                'xk_all': torch.from_numpy(xk_all).unsqueeze(1),       # [M, 1, H, W]
                't_vals': torch.from_numpy(t_arr),                     # [M]
                'c0': torch.from_numpy(c0),
                'th01': torch.from_numpy(th01),
                'c1': torch.from_numpy(c1),
                'th10': torch.from_numpy(th10),
                'valid': torch.tensor(valid, dtype=torch.float32),
                'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
                'seq_len': seq_len,
                'clip_id': cid,
                'scene_id': s['scene_id'],
                'mean_disp': s['mean_disp'],
            }

        # SINGLE FRAME (SHORT / MEDIUM / VAL)
        if s['frame_k'] is None:
            k_rel = random.randint(1, seq_len - 2)
            fk = k_rel + 1
            t_val = float(k_rel) / float(seq_len - 1)
        else:
            fk = s['frame_k']
            t_val = s['t_val']

        xk = self._read_ink(cid, fk)
        if self.is_train and random.random() < 0.5:
            x0, x1 = x1, x0
            c0, c1 = c1, c0
            th01, th10 = th10, th01
            t_val = 1.0 - t_val

        return {
            'is_sequence': False,
            'is_direct_t_flows': False,
            'x0': torch.from_numpy(x0).unsqueeze(0),
            'x1': torch.from_numpy(x1).unsqueeze(0),
            'xk': torch.from_numpy(xk).unsqueeze(0),
            't_val': torch.tensor([t_val], dtype=torch.float32),
            'c0': torch.from_numpy(c0),
            'th01': torch.from_numpy(th01),
            'c1': torch.from_numpy(c1),
            'th10': torch.from_numpy(th10),
            'valid': torch.tensor(valid, dtype=torch.float32),
            'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
            'seq_len': seq_len,
            'frame_k': fk,
            'clip_id': cid,
            'scene_id': s['scene_id'],
            'mean_disp': s['mean_disp'],
        }


def pad_collate_fn(batch):
    """
    Collate tá»± Äá»ng Äá»m cÃ¡c áº£nh trong cÃ¹ng micro-batch lÃªn kÃ­ch thÆ°á»c chia háº¿t cho 32
    mÃ  KHÃNG lÃ m biáº¿n dáº¡ng tá»· lá» nÃ©t váº½. Giá»¯ nguyÃªn `orig_hw` Äá» TPS tÃ­nh chuáº©n xÃ¡c.
    """
    max_h = max(item['orig_hw'][0].item() for item in batch)
    max_w = max(item['orig_hw'][1].item() for item in batch)
    h_pad = ((max_h + 31) // 32) * 32
    w_pad = ((max_w + 31) // 32) * 32

    is_seq = batch[0]['is_sequence']
    is_direct = bool(batch[0].get('is_direct_t_flows', False))

    def _pad_t(t, th, tw):
        dh = th - t.shape[-2]
        dw = tw - t.shape[-1]
        if dh == 0 and dw == 0:
            return t
        return F.pad(t, (0, dw, 0, dh), mode='constant', value=0.0)

    out = {
        'is_sequence': is_seq,
        'is_direct_t_flows': is_direct,
        'h_pad': h_pad,
        'w_pad': w_pad,
        'x0': torch.stack([_pad_t(b['x0'], h_pad, w_pad) for b in batch], dim=0),
        'x1': torch.stack([_pad_t(b['x1'], h_pad, w_pad) for b in batch], dim=0),
        'orig_hw': torch.stack([b['orig_hw'] for b in batch], dim=0),
        'seq_len': torch.tensor([b['seq_len'] for b in batch], dtype=torch.int32),
        'clip_id': [b['clip_id'] for b in batch],
        'scene_id': [b['scene_id'] for b in batch],
        'mean_disp': [b['mean_disp'] for b in batch],
    }
    if is_direct:
        out['c_t0_all'] = torch.stack([b['c_t0_all'] for b in batch], dim=0)
        out['th_t0_all'] = torch.stack([b['th_t0_all'] for b in batch], dim=0)
        out['c_t1_all'] = torch.stack([b['c_t1_all'] for b in batch], dim=0)
        out['th_t1_all'] = torch.stack([b['th_t1_all'] for b in batch], dim=0)
        out['valid_all'] = torch.stack([b['valid_all'] for b in batch], dim=0)
    else:
        out['c0'] = torch.stack([b['c0'] for b in batch], dim=0)
        out['th01'] = torch.stack([b['th01'] for b in batch], dim=0)
        out['c1'] = torch.stack([b['c1'] for b in batch], dim=0)
        out['th10'] = torch.stack([b['th10'] for b in batch], dim=0)
        out['valid'] = torch.stack([b['valid'] for b in batch], dim=0)

    if is_seq:
        # LÆ°u Ã½: Sampler Äáº£m báº£o táº¥t cáº£ item trong cÃ¹ng sequence batch cÃ³ cÃ¹ng seq_len -> cÃ¹ng M!
        out['xk_all'] = torch.stack([_pad_t(b['xk_all'], h_pad, w_pad) for b in batch], dim=0)
        out['t_vals'] = torch.stack([b['t_vals'] for b in batch], dim=0)
    else:
        out['xk'] = torch.stack([_pad_t(b['xk'], h_pad, w_pad) for b in batch], dim=0)
        out['t_val'] = torch.stack([b['t_val'] for b in batch], dim=0)
        out['frame_k'] = [b['frame_k'] for b in batch]
    return out


class DDPLengthWidthBatchSampler(Sampler):
    """
    Sampler cÃ¢n báº±ng táº£i tuyá»t Äá»i giá»¯a 2 GPU (97-100% GPU Utilization):
      1. NhÃ³m cÃ¡c sequence theo `(is_synth, seq_len)` vÃ  sáº¯p xáº¿p theo chiá»u rá»ng `orig_w`
         Äá» khÃ´ng lÃ£ng phÃ­ padding vÃ  tÃ¡ch riÃªng batch thá»±c vs tá»ng há»£p.
      2. Táº¡o micro-batch kÃ­ch thÆ°á»c chuáº©n cho tá»«ng N:
         - N=3 (1 frame giá»¯a): batch = 4 seqs (4 frames / GPU forward)
         - N=5 (3 frames giá»¯a): batch = 2 seqs (6 frames / GPU) [hoáº·c 1 náº¿u W > 1120]
         - N=7 (5 frames giá»¯a): batch = 1 seq  (5 frames / GPU)
      3. GhÃ©p tá»«ng cáº·p batch cÃ³ cÃ¹ng `(is_synth, seq_len)` cho (Rank 0, Rank 1) Äá» hai GPU khÃ´ng bao giá»
         lá»ch táº£i khi Äá»ng bá» NCCL all_reduce!
    """
    def __init__(self, samples, world_size=1, rank=0, batch_n3=4, batch_n5=2, batch_n7=1, seed=42):
        self.samples = samples
        self.world_size = max(1, world_size)
        self.rank = rank
        self.bs_map = {3: batch_n3, 5: batch_n5, 7: batch_n7}
        self.seed = seed
        self.epoch = 0
        self._rank_batches = []
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = random.Random(self.seed + epoch * 997)

        # PhÃ¢n loáº¡i theo (is_synth, seq_len)
        by_len = {}
        for idx, s in enumerate(self.samples):
            grp = (1 if s.get('is_synth', False) else 0, s['seq_len'])
            by_len.setdefault(grp, []).append((s['orig_w'], rng.random(), idx))

        # Gom thÃ nh cÃ¡c micro-batch gáº§n tÆ°Æ¡ng Äá»ng chiá»u rá»ng cho tá»«ng nhÃ³m
        global_micros = {}
        for grp, items in sorted(by_len.items()):
            sl = grp[1]
            items_sorted = sorted(items, key=lambda x: (((x[0] + 31) // 32) * 32, x[1]))
            indices = [x[2] for x in items_sorted]
            bs_base = self.bs_map.get(sl, 1)

            batches_for_len = []
            cur = []
            for idx in indices:
                w = self.samples[idx]['orig_w']
                bs_eff = 1 if (sl >= 5 and w > 1120 and bs_base > 1) else bs_base
                cur.append(idx)
                if len(cur) >= bs_eff:
                    batches_for_len.append((grp, self.samples[idx]['orig_w'], cur))
                    cur = []
            if cur:
                if batches_for_len and len(cur) < max(1, bs_base // 2):
                    pass
                else:
                    batches_for_len.append((grp, self.samples[cur[-1]]['orig_w'], cur))
            global_micros[grp] = batches_for_len

        # GhÃ©p cáº·p micro-batches theo world_size Äá» má»i rank cÃ³ cÃ¹ng (is_synth, seq_len) á» má»i step
        sync_steps = []
        for grp, b_list in global_micros.items():
            rem = len(b_list) % self.world_size
            if rem != 0 and len(b_list) >= self.world_size:
                b_list = b_list[:-rem]
            for j in range(0, len(b_list), self.world_size):
                tuple_batches = [b_list[j + r][2] for r in range(self.world_size)]
                sync_steps.append(tuple_batches)

        rng.shuffle(sync_steps)
        self._rank_batches = [step[self.rank] for step in sync_steps]

    def __iter__(self):
        return iter(self._rank_batches)

    def __len__(self):
        return len(self._rank_batches)


# ==============================================================================
# 6. EMA SHADOW & CHECKPOINT MANAGER (TUYá»T Äá»I CHá» LÆ¯U 2 FILE: BEST VÃ LAST)
# ==============================================================================
class AdaptiveEMA:
    """
    Exponential Moving Average nháº¹ trÃªn GPU vá»i Adaptive Warmup Decay:
      decay(step) = min(max_decay, (1 + step) / (10 + step))
    GiÃºp trá»ng sá» thÃ­ch á»©ng nhanh á» 3 epoch Äáº§u rá»i lÃ m mÆ°á»£t cá»±c tá»t á» cuá»i ká»³.
    """
    def __init__(self, model, max_decay=0.995):
        self.max_decay = max_decay
        self.num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.max_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))
            else:
                self.shadow[k].copy_(v)

    def apply_shadow(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None

    def state_dict(self):
        return {'num_updates': self.num_updates, 'shadow': self.shadow}

    def load_state_dict(self, state, device):
        if not state:
            return
        self.num_updates = int(state.get('num_updates', 0))
        loaded_sh = state.get('shadow', {})
        for k in self.shadow:
            if k in loaded_sh and self.shadow[k].shape == loaded_sh[k].shape:
                self.shadow[k].copy_(loaded_sh[k].to(device))


class TwoFileCheckpointManager:
    """
    ChÃ­nh sÃ¡ch lÆ°u trá»¯ tá»i giáº£n tuyá»t Äá»i theo yÃªu cáº§u Äá» Ã¡n:
    CHá» DUY TRÃ ÄÃNG 2 FILE TRONG `output_dir`:
      1. model_tps_finetuned_best.pt : LÆ°u trá»ng sá» tá»t nháº¥t trÃªn táº­p Val (EMA weights).
      2. model_tps_finetuned_last.pt : LÆ°u trá»ng sá» Epoch gáº§n nháº¥t KÃM tráº¡ng thÃ¡i Resume Äáº§y Äá»§.
    Tuyá»t Äá»i KHÃNG sinh thÃªm file thá»© 3 hoáº·c cÃ¡c file rá»i tá»«ng epoch!
    """
    def __init__(self, output_dir, is_rank0=True):
        self.output_dir = output_dir
        self.is_rank0 = is_rank0
        self.best_score = float('inf')
        self.best_epoch = 0
        os.makedirs(output_dir, exist_ok=True)
        self.best_path = os.path.join(output_dir, "model_tps_finetuned_best.pt")
        self.last_path = os.path.join(output_dir, "model_tps_finetuned_last.pt")

    def _atomic_save(self, obj, dst_path):
        tmp_path = dst_path + ".tmp"
        torch.save(obj, tmp_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.rename(tmp_path, dst_path)

    def save_epoch(self, epoch, base_model, ema, optimizer, scaler, score, history, is_best):
        if not self.is_rank0:
            return

        if is_best:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            best_payload = {
                'epoch': int(epoch),
                'best_score': float(score),
                'model_state': {k: v.detach().cpu() for k, v in ema.shadow.items()},
                'raw_model_state': {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
                'history': history,
            }
            self._atomic_save(best_payload, self.best_path)
            print(f"ð [Ká»¶ Lá»¤C Má»I]: Cáº­p nháº­t {os.path.basename(self.best_path)} táº¡i Epoch {epoch:02d} (Composite Score = {score:.4f})!", flush=True)

        # LÆ°u gá»p Epoch gáº§n nháº¥t + tráº¡ng thÃ¡i Resume vÃ o duy nháº¥t file last.pt
        last_payload = {
            'epoch': int(epoch),
            'model_state': {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
            'ema_state': {
                'num_updates': ema.num_updates,
                'shadow': {k: v.detach().cpu() for k, v in ema.shadow.items()},
            },
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'best_score': float(self.best_score),
            'best_epoch': int(self.best_epoch),
            'history': history,
        }
        self._atomic_save(last_payload, self.last_path)

    def try_resume(self, base_model, ema, optimizer, scaler, map_location):
        if not os.path.exists(self.last_path):
            return 0, []
        ckpt = torch.load(self.last_path, map_location=map_location, weights_only=False)
        last_ep = int(ckpt.get('epoch', 0))
        self.best_score = float(ckpt.get('best_score', float('inf')))
        self.best_epoch = int(ckpt.get('best_epoch', 0))
        history = ckpt.get('history', [])

        if 'model_state' in ckpt:
            base_model.load_state_dict(ckpt['model_state'], strict=True)
        if 'ema_state' in ckpt and ckpt['ema_state'] is not None:
            ema.load_state_dict(ckpt['ema_state'], map_location)
        if optimizer is not None and ckpt.get('optimizer_state') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scaler is not None and ckpt.get('scaler_state') is not None:
            scaler.load_state_dict(ckpt['scaler_state'])
        return last_ep, history


# ==============================================================================
# 7. VALIDATION Äá»I CHá»¨NG & CHá»N CHECKPOINT (KHÃNG BIAS, THUáº¦N FP32)
# ==============================================================================
@torch.no_grad()
def run_validation_suite(model, val_loader, device, downscale_factor=2, is_ddp=False, evaluate_m0=False):
    model.eval()
    keys_sum = ['psnr', 'ssim', 'val_l1', 'ie_mae', 'pwcr', 'rgd', 'ghost', 'dark_ghost',
                'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy', 'semd',
                'recall', 'precision', 'f1', 'density_ratio', 'skel_recall', 'tip_recall']
    acc_main = {k: 0.0 for k in keys_sum}
    acc_main['count'] = 0.0
    acc_m0 = {k: 0.0 for k in keys_sum}
    acc_m0['count'] = 0.0

    brk_keys = ['count', 'psnr', 'ssim', 'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy',
                'semd', 'skel_recall', 'tip_recall', 'f1', 'ghost', 'density_ratio']
    breakdown = {n: {k: 0.0 for k in brk_keys} for n in (3, 5, 7)}

    for batch in val_loader:
        h_pad, w_pad = batch['h_pad'], batch['w_pad']
        x0 = batch['x0'].to(device, non_blocking=True)
        x1 = batch['x1'].to(device, non_blocking=True)
        xk = batch['xk'].to(device, non_blocking=True)
        t_val = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
        c0 = batch['c0'].to(device, non_blocking=True)
        th01 = batch['th01'].to(device, non_blocking=True)
        c1 = batch['c1'].to(device, non_blocking=True)
        th10 = batch['th10'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        orig_hw = batch['orig_hw']
        seq_lens = batch['seq_len']

        cf01 = tps_flow_from_theta_batch(th01, c0, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)
        cf10 = tps_flow_from_theta_batch(th10, c1, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)

        # Validation cháº¡y hoÃ n toÃ n trÃªn FP32 thuáº§n tÃºy (loáº¡i bá» nhiá»u lÆ°á»£ng tá»­ hÃ³a ná»­a Äá» chÃ­nh xÃ¡c)
        with torch.amp.autocast('cuda', enabled=False):
            pred_it, coarse_warp, *_ = model(x0.float(), x1.float(), t_val.float(), cf01.float(), cf10.float())

        b = x0.shape[0]
        for i in range(b):
            hi, wi = int(orig_hw[i, 0].item()), int(orig_hw[i, 1].item())
            sl = int(seq_lens[i].item())
            gt_i = xk[i:i + 1, :, :hi, :wi]
            pr_i = pred_it[i:i + 1, :, :hi, :wi]

            m = evaluate_metrics_dual_regime(pr_i, gt_i)
            acc_main['count'] += 1.0
            for k in keys_sum:
                acc_main[k] += float(m[k].item())

            if sl in breakdown:
                breakdown[sl]['count'] += 1.0
                for bk in brk_keys:
                    if bk != 'count':
                        breakdown[sl][bk] += float(m[bk].item())

            if evaluate_m0:
                m0_pr = coarse_warp[i:i + 1, :, :hi, :wi]
                mm0 = evaluate_metrics_dual_regime(m0_pr, gt_i)
                acc_m0['count'] += 1.0
                for k in keys_sum:
                    acc_m0[k] += float(mm0[k].item())

    def _finalize(acc, brk=None):
        if is_ddp:
            vec = torch.tensor([acc['count']] + [acc[k] for k in keys_sum], dtype=torch.float64, device=device)
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            cnt = max(1.0, float(vec[0].item()))
            out = {k: float(vec[idx + 1].item()) / cnt for idx, k in enumerate(keys_sum)}
            out['total_samples'] = int(cnt)
        else:
            cnt = max(1.0, acc['count'])
            out = {k: acc[k] / cnt for k in keys_sum}
            out['total_samples'] = int(cnt)

        if brk is not None:
            brk_out = {}
            for sl in (3, 5, 7):
                if is_ddp:
                    bvec = torch.tensor([brk[sl][k] for k in brk_keys], dtype=torch.float64, device=device)
                    dist.all_reduce(bvec, op=dist.ReduceOp.SUM)
                    bcnt = max(1.0, float(bvec[0].item()))
                    brk_out[sl] = {k: float(bvec[idx].item()) / bcnt for idx, k in enumerate(brk_keys) if k != 'count'}
                    brk_out[sl]['count'] = int(bvec[0].item())
                else:
                    bcnt = max(1.0, brk[sl]['count'])
                    brk_out[sl] = {k: brk[sl][k] / bcnt for k in brk_keys if k != 'count'}
                    brk_out[sl]['count'] = int(brk[sl]['count'])
            out['breakdown'] = brk_out
        return out

    res_main = _finalize(acc_main, breakdown)
    if evaluate_m0:
        return res_main, _finalize(acc_m0)
    return res_main


# ==============================================================================
# 8. Äá»C Táº¬P BENCHMARK Äá»C Láº¬P SONG SONG TRÃN 2 GPU
# ==============================================================================
@torch.no_grad()
def evaluate_per_sample_benchmark(model, bench_loader, device, downscale_factor=2):
    """Cháº¡y mÃ´ hÃ¬nh trÃªn dataloader Benchmark vÃ  tráº£ vá» dict keyed by (clip_id, frame_k)."""
    model.eval()
    results = {}
    for batch in bench_loader:
        h_pad, w_pad = batch['h_pad'], batch['w_pad']
        x0 = batch['x0'].to(device, non_blocking=True)
        x1 = batch['x1'].to(device, non_blocking=True)
        xk = batch['xk'].to(device, non_blocking=True)
        t_val = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
        c0 = batch['c0'].to(device, non_blocking=True)
        th01 = batch['th01'].to(device, non_blocking=True)
        c1 = batch['c1'].to(device, non_blocking=True)
        th10 = batch['th10'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        orig_hw = batch['orig_hw']

        cf01 = tps_flow_from_theta_batch(th01, c0, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)
        cf10 = tps_flow_from_theta_batch(th10, c1, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_start = time.time()
        with torch.amp.autocast('cuda', enabled=False):
            pred_it, coarse_warp, *_ = model(x0.float(), x1.float(), t_val.float(), cf01.float(), cf10.float())
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        lat_ms = ((time.time() - t_start) * 1000.0) / max(1, x0.shape[0])

        for i in range(x0.shape[0]):
            hi, wi = int(orig_hw[i, 0].item()), int(orig_hw[i, 1].item())
            gt_i = xk[i:i + 1, :, :hi, :wi]
            pr_i = pred_it[i:i + 1, :, :hi, :wi]
            m0_i = coarse_warp[i:i + 1, :, :hi, :wi]

            m_pred = evaluate_metrics_dual_regime(pr_i, gt_i)
            m_coarse = evaluate_metrics_dual_regime(m0_i, gt_i)

            rec = {
                'clip_id': batch['clip_id'][i],
                'scene_id': batch['scene_id'][i],
                'seq_len': int(batch['seq_len'][i].item()),
                'frame_k': int(batch['frame_k'][i]),
                't_val': float(batch['t_val'][i].item()),
                'mean_disp': float(batch['mean_disp'][i]),
                'latency_ms': float(lat_ms),
            }
            for k, v in m_pred.items():
                rec[k] = float(v.item())
            for k, v in m_coarse.items():
                rec[f"m0_{k}"] = float(v.item())
            results[(rec['clip_id'], rec['frame_k'])] = rec
    return results


# ==============================================================================
# 9. CHÆ¯Æ NG TRÃNH HUáº¤N LUYá»N CHÃNH (DDP WORKER)
# ==============================================================================
def _apply_paired_spatial_aug_gpu(x0, x1, xk_tensor, cf01, cf10, orig_hw, is_sequence):
    """Ãp dá»¥ng Paired H-Flip (p=0.50) vÃ  V-Flip (p=0.25) Äá»ng bá» áº£nh + TPS flow bÃªn trong orig_hw."""
    b = x0.shape[0]
    for i in range(b):
        hi, wi = int(orig_hw[i, 0].item()), int(orig_hw[i, 1].item())
        if random.random() < 0.50:
            x0[i:i + 1, :, :hi, :wi] = torch.flip(x0[i:i + 1, :, :hi, :wi], dims=[-1])
            x1[i:i + 1, :, :hi, :wi] = torch.flip(x1[i:i + 1, :, :hi, :wi], dims=[-1])
            if is_sequence:
                xk_tensor[i:i + 1, :, :, :hi, :wi] = torch.flip(xk_tensor[i:i + 1, :, :, :hi, :wi], dims=[-1])
            else:
                xk_tensor[i:i + 1, :, :hi, :wi] = torch.flip(xk_tensor[i:i + 1, :, :hi, :wi], dims=[-1])
            cf01[i:i + 1, :, :hi, :wi] = torch.flip(cf01[i:i + 1, :, :hi, :wi], dims=[-1])
            cf01[i, 0, :hi, :wi].neg_()
            cf10[i:i + 1, :, :hi, :wi] = torch.flip(cf10[i:i + 1, :, :hi, :wi], dims=[-1])
            cf10[i, 0, :hi, :wi].neg_()

        if random.random() < 0.25:
            x0[i:i + 1, :, :hi, :wi] = torch.flip(x0[i:i + 1, :, :hi, :wi], dims=[-2])
            x1[i:i + 1, :, :hi, :wi] = torch.flip(x1[i:i + 1, :, :hi, :wi], dims=[-2])
            if is_sequence:
                xk_tensor[i:i + 1, :, :, :hi, :wi] = torch.flip(xk_tensor[i:i + 1, :, :, :hi, :wi], dims=[-2])
            else:
                xk_tensor[i:i + 1, :, :hi, :wi] = torch.flip(xk_tensor[i:i + 1, :, :hi, :wi], dims=[-2])
            cf01[i:i + 1, :, :hi, :wi] = torch.flip(cf01[i:i + 1, :, :hi, :wi], dims=[-2])
            cf01[i, 1, :hi, :wi].neg_()
            cf10[i:i + 1, :, :hi, :wi] = torch.flip(cf10[i:i + 1, :, :hi, :wi], dims=[-2])
            cf10[i, 1, :hi, :wi].neg_()


def main():
    parser = argparse.ArgumentParser(description="DDP Worker Fine-tune TPS-Inbetween (Pha 1: Temporal + Ghost + Scale)")
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--train_clips_dir', type=str, required=True, help="ThÆ° má»¥c train/")
    parser.add_argument('--val_clips_dir', type=str, required=True, help="ThÆ° má»¥c val/")
    parser.add_argument('--tps_cache', type=str, required=True)
    parser.add_argument('--m1_checkpoint', type=str, required=True, help="Checkpoint gá»c M1 dÃ¹ng Äo baseline Äá»i chuáº©n")
    parser.add_argument('--init_checkpoint', type=str, default="", help="Checkpoint warm-start tÃ¹y chá»n (vd: Seq2 Ep15)")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/checkpoints_tps_finetune")
    parser.add_argument('--train_mode', type=str, default="LONG", choices=["SHORT", "MEDIUM", "LONG"])
    parser.add_argument('--epochs', type=int, default=5, help="Sá» epoch cho Pha 2 Cleanliness Polish (máº·c Äá»nh: 5)")
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--max_chunk_frames', type=int, default=3, help="Sá» frame tá»i Äa má»i forward micro-chunk")
    parser.add_argument('--lr_backbone', type=float, default=1.5e-5, help="LR Backbone tháº¥p Äá» báº£o toÃ n nÃ©t (Pha 2: 1.5e-5)")
    parser.add_argument('--lr_unet', type=float, default=3.5e-5, help="LR Unet tháº¥p Äá» tinh chá»nh dá»n ná»n (Pha 2: 3.5e-5)")
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--synth_count', type=int, default=120, help="Sá» chuá»i hÃ¬nh há»c tá»ng há»£p má»i epoch á» LONG mode")
    parser.add_argument('--w_ghost', type=float, default=1.10, help="Trá»ng sá» quÃ©t sáº¡ch ná»n Pha 2 (máº·c Äá»nh: 1.10)")
    parser.add_argument('--w_temp', type=float, default=0.60, help="Trá»ng sá» nháº¥t quÃ¡n thá»i gian (máº·c Äá»nh: 0.60)")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ram_limit_gb', type=float, default=23.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true', help="Cháº¡y kiá»m tra 2 bÆ°á»c khá»i Äá»ng nhanh rá»i thoÃ¡t")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = 'nccl' if dist.is_nccl_available() else 'gloo'
        else:
            device = torch.device('cpu')
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method=os.environ.get('DDP_INIT_METHOD', 'env://'))
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank0 = (local_rank == 0)
    random.seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    mode = args.train_mode.upper()
    tps_downscale = 2 if mode == "LONG" else 4

    if is_rank0:
        init_tag = os.path.basename(args.init_checkpoint) if (args.init_checkpoint and os.path.exists(args.init_checkpoint)) else os.path.basename(args.m1_checkpoint)
        print("=" * 84)
        print(f"ð KHá»I Äá»NG HUáº¤N LUYá»N PHA 1 DDP TPS-INBETWEEN (MODE={mode} | INIT={init_tag})")
        print(f"   World Size = {world_size} GPU(s) | Device = {device} | TPS Downscale = 1/{tps_downscale}")
        print(f"   Epochs = {args.epochs} | Synth = {args.synth_count} | w_ghost={args.w_ghost} | w_temp={args.w_temp}")
        print(f"   Checkpoint Output = {args.output_dir} (Chá» lÆ°u 2 file: best.pt & last.pt)")
        print("=" * 84, flush=True)

    ckpt_mgr = TwoFileCheckpointManager(args.output_dir, is_rank0=is_rank0)
    base_model = TPSInbetweenOriginal().to(device)
    load_tps_weights(base_model, args.m1_checkpoint, device=device, prefer_ema=False, is_rank0=is_rank0)

    # Dataset & Loader
    train_ds = SakugaSequenceDataset(
        args.train_csv, args.train_clips_dir, args.tps_cache,
        train_mode=mode, is_train=True, seed=args.seed, synth_count=args.synth_count
    )
    val_ds = SakugaSequenceDataset(
        args.val_csv, args.val_clips_dir, args.tps_cache,
        train_mode="VAL", is_train=False, seed=args.seed, synth_count=0
    )

    if mode == "LONG":
        train_sampler = DDPLengthWidthBatchSampler(
            train_ds.samples, world_size=world_size, rank=local_rank,
            batch_n3=4, batch_n5=2, batch_n7=1, seed=args.seed
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, collate_fn=pad_collate_fn,
            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0)
        )
    else:
        bs_single = 12 if mode == "SHORT" else 8
        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
        train_loader = DataLoader(
            train_ds, batch_size=bs_single, shuffle=(train_sampler is None), sampler=train_sampler,
            collate_fn=pad_collate_fn, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'), persistent_workers=(args.num_workers > 0), drop_last=True
        )

    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, sampler=val_sampler,
        collate_fn=pad_collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), persistent_workers=(args.num_workers > 0), drop_last=False
    )

    if is_rank0:
        print(f"[â] Dá»¯ liá»u sáºµn sÃ ng: Train Items = {len(train_ds):,} ({len(train_loader):,} batches/GPU) | Val Samples = {len(val_ds):,}", flush=True)

    # Äo Baseline M0 & M1 táº¡i Epoch 0 trÃªn ÄÃºng model_latest.pt trÆ°á»c khi náº¡p warm-start
    baseline_json_path = os.path.join(os.path.dirname(args.output_dir), "val_baseline_m0_m1.json")
    has_resume = os.path.exists(ckpt_mgr.last_path)
    need_baseline = (not has_resume and not args.dry_run and not os.path.exists(baseline_json_path))
    if is_ddp:
        base_flag = torch.tensor([1 if need_baseline else 0], dtype=torch.int32, device=device)
        dist.broadcast(base_flag, src=0)
        need_baseline = bool(base_flag.item())

    m0_val_cache, m1_val_cache = None, None
    if need_baseline:
        if is_rank0:
            print("\nð¬ ÄANG ÄO BASELINE EPOCH 0 TRÃN Táº¬P VAL (M0 COARSE TPS & M1 ORIGINAL)...", flush=True)
        m1_val, m0_val = run_validation_suite(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=True)
        m0_val_cache, m1_val_cache = m0_val, m1_val
        if is_rank0:
            with open(baseline_json_path, 'w', encoding='utf-8') as f:
                json.dump({'M0_Coarse': m0_val, 'M1_Original': m1_val}, f, indent=2, ensure_ascii=False)
            print(f"ð M0 (Coarse TPS): PSNR={m0_val['psnr']:.2f}dB | SSIM={m0_val['ssim']:.4f} | WCD_DoG={m0_val['wcd_dog']:.4f} | Ghost={m0_val['ghost']:.0f}", flush=True)
            print(f"ð M1 (Orig ML240): PSNR={m1_val['psnr']:.2f}dB | SSIM={m1_val['ssim']:.4f} | WCD_DoG={m1_val['wcd_dog']:.4f} | Ghost={m1_val['ghost']:.0f}\n", flush=True)
        if is_ddp:
            dist.barrier()
    elif os.path.exists(baseline_json_path):
        try:
            with open(baseline_json_path, 'r', encoding='utf-8') as f:
                b_all = json.load(f)
                m0_val_cache = b_all.get('M0_Coarse', {})
                m1_val_cache = b_all.get('M1_Original', {})
        except Exception:
            m0_val_cache, m1_val_cache = None, None

    # Náº¿u cÃ³ --init_checkpoint (vd: Seq2 Ep15), náº¡p sau khi ÄÃ£ Äo xong M1 Baseline Epoch 0
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        if os.path.abspath(args.init_checkpoint) != os.path.abspath(args.m1_checkpoint):
            load_tps_weights(base_model, args.init_checkpoint, device=device, prefer_ema=True, is_rank0=is_rank0)

    if mode == "SHORT":
        for p in base_model.IFNet.block0.parameters():
            p.requires_grad = False

    # Optimizer, EMA, Loss & Scaler
    ema = AdaptiveEMA(base_model, max_decay=args.ema_decay)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in base_model.encoder.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.IFNet.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.unet.parameters() if p.requires_grad], 'lr': args.lr_unet, 'weight_decay': 1e-4},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    if is_ddp:
        if is_rank0:
            criterion = AdaptiveLineartLoss(train_mode=mode, is_rank0=True, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)
        dist.barrier()
        if not is_rank0:
            criterion = AdaptiveLineartLoss(train_mode=mode, is_rank0=False, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)
        dist.barrier()
    else:
        criterion = AdaptiveLineartLoss(train_mode=mode, is_rank0=True, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)

    start_epoch = 1
    history = []
    if not args.dry_run:
        last_ep, hist = ckpt_mgr.try_resume(base_model, ema, optimizer, scaler, map_location=device)
        if last_ep > 0:
            start_epoch = last_ep + 1
            history = hist
            if is_rank0:
                print(f"[â»ï¸ AUTO-RESUME] KhÃ´i phá»¥c thÃ nh cÃ´ng Epoch {last_ep:02d}. Tiáº¿p tá»¥c tá»« Epoch {start_epoch:02d} (Best Ep={ckpt_mgr.best_epoch:02d}).", flush=True)

    # Má»c chuáº©n M1 Epoch 0 Äá» chuáº©n hÃ³a Composite Score
    m1_ref = m1_val_cache or {}
    m0_ref = m0_val_cache or {}
    base_wcd_dog = float(m1_ref.get('wcd_dog', 1.4620))
    base_semd = float(m1_ref.get('semd', 0.0131))
    base_skel = float(m1_ref.get('skel_recall', 0.8164))
    base_tip = float(m1_ref.get('tip_recall', 0.8204))
    base_ghost = float(m1_ref.get('ghost', 1409.0))
    m0_ghost_ref = float(m0_ref.get('ghost', 17975.0))
    best_ghost_gate = 9500.0  # Má»c khá»i Äiá»m Pha 2 (káº¿ thá»«a Pha 1 Ep14 ~9185 px), Gate sáº½ tá»± Äá»ng siáº¿t cháº·t dáº§n khi Ghost giáº£m
    if is_rank0:
        print(f"âï¸ Má»C Äá»I CHUáº¨N EPOCH 0 (M1): WCD_DoG={base_wcd_dog:.4f} | SkelRec={base_skel*100:.1f}% | Ghost={base_ghost:.0f} | GhostGateStart={best_ghost_gate:.0f}", flush=True)

    if is_ddp:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == 'cuda' else {}
        ddp_model = DDP(base_model, find_unused_parameters=(mode == "SHORT"), **ddp_kwargs)
    else:
        ddp_model = base_model

    history_csv_path = os.path.join(os.path.dirname(args.output_dir), "training_history_finetune.csv")
    watchdog_paused = False
    t_total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep0 = time.time()
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Lá»ch Cosine Warmup 2 epochs
        if epoch == 1:
            lr_scale = 0.35
        elif epoch == 2:
            lr_scale = 0.70
        elif epoch == 3:
            lr_scale = 1.00
        else:
            prog = float(epoch - 3) / float(max(1, args.epochs - 3))
            lr_scale = max(0.10, 0.5 * (1.0 + math.cos(math.pi * prog)))

        optimizer.param_groups[0]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[1]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[2]['lr'] = args.lr_unet * lr_scale

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_frames = 0
        num_batches = len(train_loader)
        grad_accum = max(1, args.grad_accum)

        for b_idx, batch in enumerate(train_loader):
            h_pad, w_pad = batch['h_pad'], batch['w_pad']
            is_direct = bool(batch.get('is_direct_t_flows', False))
            x0 = batch['x0'].to(device, non_blocking=True)
            x1 = batch['x1'].to(device, non_blocking=True)
            orig_hw = batch['orig_hw'].to(device, non_blocking=True)

            if is_direct:
                # Chuá»i hÃ¬nh há»c tá»ng há»£p dÃ¹ng trá»±c tiáº¿p Lagrangian Direct TPS (cf0t, cf1t)
                xk_all = batch['xk_all'].to(device, non_blocking=True)
                t_vals = batch['t_vals'].to(device, non_blocking=True)
                bsz, m_steps = xk_all.shape[0], xk_all.shape[1]
                c_t0_f = batch['c_t0_all'].to(device, non_blocking=True).reshape(bsz * m_steps, -1, 2)
                th_t0_f = batch['th_t0_all'].to(device, non_blocking=True).reshape(bsz * m_steps, -1, 2)
                c_t1_f = batch['c_t1_all'].to(device, non_blocking=True).reshape(bsz * m_steps, -1, 2)
                th_t1_f = batch['th_t1_all'].to(device, non_blocking=True).reshape(bsz * m_steps, -1, 2)
                val_f = batch['valid_all'].to(device, non_blocking=True).reshape(bsz * m_steps)
                hw_f = orig_hw.unsqueeze(1).expand(-1, m_steps, -1).reshape(bsz * m_steps, 2)

                cf01_f = tps_flow_from_theta_batch(th_t0_f, c_t0_f, val_f, hw_f, h_pad, w_pad, device, downscale_factor=tps_downscale)
                cf10_f = tps_flow_from_theta_batch(th_t1_f, c_t1_f, val_f, hw_f, h_pad, w_pad, device, downscale_factor=tps_downscale)

                x0_f = x0.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
                x1_f = x1.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
                xk_f = xk_all.reshape(bsz * m_steps, 1, h_pad, w_pad)
                t_f = t_vals.reshape(bsz * m_steps, 1, 1, 1)
                seq_ids_f = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, m_steps).reshape(-1)
            else:
                c0 = batch['c0'].to(device, non_blocking=True)
                th01 = batch['th01'].to(device, non_blocking=True)
                c1 = batch['c1'].to(device, non_blocking=True)
                th10 = batch['th10'].to(device, non_blocking=True)
                valid = batch['valid'].to(device, non_blocking=True)

                cf01 = tps_flow_from_theta_batch(th01, c0, valid, orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale)
                cf10 = tps_flow_from_theta_batch(th10, c1, valid, orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale)

                if batch['is_sequence']:
                    xk_all = batch['xk_all'].to(device, non_blocking=True)
                    t_vals = batch['t_vals'].to(device, non_blocking=True)
                    _apply_paired_spatial_aug_gpu(x0, x1, xk_all, cf01, cf10, orig_hw, is_sequence=True)
                    bsz, m_steps = xk_all.shape[0], xk_all.shape[1]

                    x0_f = x0.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
                    x1_f = x1.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
                    cf01_f = cf01.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                    cf10_f = cf10.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                    xk_f = xk_all.reshape(bsz * m_steps, 1, h_pad, w_pad)
                    t_f = t_vals.reshape(bsz * m_steps, 1, 1, 1)
                    hw_f = orig_hw.unsqueeze(1).expand(-1, m_steps, -1).reshape(bsz * m_steps, 2)
                    seq_ids_f = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, m_steps).reshape(-1)
                else:
                    xk_f = batch['xk'].to(device, non_blocking=True)
                    _apply_paired_spatial_aug_gpu(x0, x1, xk_f, cf01, cf10, orig_hw, is_sequence=False)
                    t_f = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
                    x0_f, x1_f, cf01_f, cf10_f, hw_f = x0, x1, cf01, cf10, orig_hw
                    seq_ids_f = torch.arange(x0.shape[0], device=device)

            total_f = x0_f.shape[0]
            total_pairs = int((seq_ids_f[1:] == seq_ids_f[:-1]).sum().item()) if total_f >= 2 else 0
            do_opt_step = ((b_idx + 1) % grad_accum == 0) or ((b_idx + 1) == num_batches)
            chunk_sz = max(1, args.max_chunk_frames)

            batch_loss_sum = 0.0
            prev_pred_last = None
            prev_gt_last = None
            prev_hw_last = None
            prev_seq_id = None

            for c_start in range(0, total_f, chunk_sz):
                c_end = min(c_start + chunk_sz, total_f)
                is_last_chunk = (c_end == total_f)
                need_sync = (do_opt_step and is_last_chunk)

                sync_ctx = (ddp_model.no_sync() if (is_ddp and not need_sync) else contextlib.nullcontext())
                with sync_ctx:
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        pred_c, *_ = ddp_model(
                            x0_f[c_start:c_end], x1_f[c_start:c_end],
                            t_f[c_start:c_end], cf01_f[c_start:c_end], cf10_f[c_start:c_end],
                            direct_t_flows=is_direct
                        )
                        gt_c = xk_f[c_start:c_end]
                        hw_c = hw_f[c_start:c_end]
                        seq_c = seq_ids_f[c_start:c_end]
                        l_c, d_c = criterion(pred_c, gt_c, valid_hw=hw_c)
                        weight_scale = float(c_end - c_start) / float(total_f * grad_accum)
                        scaled_l = l_c * weight_scale

                        # Temporal Consistency Loss (2C) cho cÃ¡c cáº·p ká» nhau cÃ¹ng sequence
                        if total_pairs > 0 and criterion.w_temp > 0.0:
                            l_temp_sum = pred_c.new_zeros(())
                            if pred_c.shape[0] >= 2:
                                same_m = (seq_c[1:] == seq_c[:-1])
                                if same_m.any():
                                    l_temp_sum = l_temp_sum + criterion.compute_temporal_pair_loss(
                                        pred_c[:-1][same_m], pred_c[1:][same_m],
                                        gt_c[:-1][same_m], gt_c[1:][same_m],
                                        valid_hw=hw_c[1:][same_m]
                                    )
                            if prev_pred_last is not None and int(seq_c[0].item()) == prev_seq_id:
                                l_temp_sum = l_temp_sum + criterion.compute_temporal_pair_loss(
                                    prev_pred_last, pred_c[0:1], prev_gt_last, gt_c[0:1], valid_hw=hw_c[0:1]
                                )
                            scaled_l = scaled_l + (l_temp_sum / float(total_pairs * grad_accum))

                    scaler.scale(scaled_l).backward()
                    batch_loss_sum += d_c['total_loss'] * float(c_end - c_start)
                    prev_pred_last = pred_c[-1:].detach()
                    prev_gt_last = gt_c[-1:].detach()
                    prev_hw_last = hw_c[-1:].detach()
                    prev_seq_id = int(seq_c[-1].item())

            acc_loss += batch_loss_sum
            acc_frames += total_f

            if do_opt_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(base_model)

            if args.dry_run and b_idx >= 1:
                break

        train_min = (time.time() - t_ep0) / 60.0
        if args.dry_run:
            if is_rank0:
                print(f"[â DRY-RUN THÃNH CÃNG] ÄÃ£ kiá»m chá»©ng {acc_frames} frames qua DDP forward/backward + L_temp + SpatialAug + EMA! Avg Loss={acc_loss/max(1,acc_frames):.4f}", flush=True)
            break

        # Kiá»m Äá»nh báº±ng trá»ng sá» EMA trÃªn toÃ n bá» táº­p Val (táº¡m trÃ¡o EMA -> restore, 0 tá»n VRAM)
        t_val0 = time.time()
        ema.apply_shadow(base_model)
        val_m = run_validation_suite(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=False)
        ema.restore(base_model)
        val_min = (time.time() - t_val0) / 60.0

        # ThÆ°á»c Äo Epoch-0 Normalized + Two-Tier Gating cÃ¢n báº±ng Ghost & Density
        dens = float(val_m['density_ratio'])
        skel = float(val_m['skel_recall'])
        tip = float(val_m['tip_recall'])
        wcd_dog = float(val_m['wcd_dog'])
        semd = float(val_m['semd'])
        ghost = float(val_m['ghost'])

        # Pha 2: Dynamic Monotonic Ghost Gate
        gate_density = (0.93 <= dens <= 1.07)
        gate_skel = (skel >= max(0.840, base_skel + 0.015))
        gate_wcd = (wcd_dog <= base_wcd_dog * 1.02)
        gate_ghost = (ghost < best_ghost_gate)
        tier1_passed = bool(gate_density and gate_skel and gate_wcd and gate_ghost)

        term_wcd = 0.25 * (wcd_dog / max(1e-4, base_wcd_dog))
        term_semd = 0.08 * (semd / max(1e-4, base_semd))
        term_skel = 0.18 * ((1.0 - skel) / max(1e-4, 1.0 - base_skel))
        term_tip = 0.08 * ((1.0 - tip) / max(1e-4, 1.0 - base_tip))
        term_dens = 0.16 * (abs(dens - 1.00) / 0.10)
        term_ghost = 0.25 * (ghost / max(1500.0, base_ghost * 2.0))
        s_val = term_wcd + term_semd + term_skel + term_tip + term_dens + term_ghost

        if tier1_passed:
            comp_score = s_val
            best_ghost_gate = ghost  # Cáº­p nháº­t má»c Ghost ká»· lá»¥c má»i Äá» siáº¿t cháº·t cÃ¡c epoch sau!
        else:
            over_dens = max(0.0, dens - 1.07)
            under_dens = max(0.0, 0.93 - dens)
            over_ghost = max(0.0, (ghost - best_ghost_gate) / 1000.0)
            comp_score = s_val + 10.0 + 5.0 * over_dens + 3.0 * under_dens + 1.0 * over_ghost

        is_best = comp_score < ckpt_mgr.best_score

        if is_rank0:
            row = {
                'epoch': epoch,
                'lr_scale': round(lr_scale, 4),
                'train_loss': acc_loss / max(1, acc_frames),
                'val_l1': val_m['val_l1'],
                'psnr': val_m['psnr'],
                'ssim': val_m['ssim'],
                'wcd': val_m['wcd'],
                'wcd_legacy': val_m['wcd_legacy'],
                'wcd_dog': val_m['wcd_dog'],
                'wcd_dog_legacy': val_m['wcd_dog_legacy'],
                'semd': val_m['semd'],
                'skel_recall': val_m['skel_recall'],
                'tip_recall': val_m['tip_recall'],
                'recall': val_m['recall'],
                'precision': val_m['precision'],
                'f1': val_m['f1'],
                'density_ratio': val_m['density_ratio'],
                'pwcr': val_m['pwcr'],
                'rgd': val_m['rgd'],
                'ghost': val_m['ghost'],
                's_val_norm': s_val,
                'score': comp_score,
                'gate_passed': tier1_passed,
                'is_best': is_best,
                'train_time_min': train_min,
                'val_time_min': val_min,
            }
            for sl in (3, 5, 7):
                if sl in val_m.get('breakdown', {}):
                    row[f'psnr_N{sl}'] = val_m['breakdown'][sl]['psnr']
                    row[f'wcd_N{sl}'] = val_m['breakdown'][sl]['wcd']
                    row[f'wcd_dog_N{sl}'] = val_m['breakdown'][sl]['wcd_dog']
                    row[f'skel_recall_N{sl}'] = val_m['breakdown'][sl]['skel_recall']
                    row[f'tip_recall_N{sl}'] = val_m['breakdown'][sl]['tip_recall']
                    row[f'ghost_N{sl}'] = val_m['breakdown'][sl]['ghost']
                    row[f'density_N{sl}'] = val_m['breakdown'][sl]['density_ratio']

            history.append(row)
            pd.DataFrame(history).to_csv(history_csv_path, index=False)
            ckpt_mgr.save_epoch(epoch, base_model, ema, optimizer, scaler, comp_score, history, is_best=is_best)

            gate_badge = "â PASS" if tier1_passed else "â ï¸ GATE_BREACH"
            print(
                f"ð EP {epoch:02d}/{args.epochs:02d} [{mode}] | Train={train_min:.2f}m | Val={val_min:.2f}m | "
                f"Loss={row['train_loss']:.4f} | PSNR={val_m['psnr']:.2f}dB | "
                f"WCD_DoG={val_m['wcd_dog']:.4f} | SkelRec={val_m['skel_recall']*100:.1f}% | "
                f"Dens={dens:.3f} | Ghost={ghost:.0f} | Gate={gate_badge} | Score={comp_score:.4f}",
                flush=True
            )

        trim_memory()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
        if is_rank0:
            try:
                import psutil
                mem_gb = psutil.virtual_memory().used / (1024 ** 3)
                if mem_gb >= args.ram_limit_gb:
                    print(f"\nâ ï¸ [RAM WATCHDOG]: Host RAM Äáº¡t {mem_gb:.2f} GB >= {args.ram_limit_gb} GB. ThoÃ¡t an toÃ n mÃ£ 42!", flush=True)
                    stop_flag.fill_(1)
            except Exception:
                pass

        if is_ddp:
            dist.broadcast(stop_flag, src=0)
            dist.barrier()

        if int(stop_flag.item()) == 1:
            watchdog_paused = True
            break

    if is_ddp:
        dist.destroy_process_group()

    if watchdog_paused:
        sys.exit(42)
    if is_rank0 and not args.dry_run:
        print(f"\nð HOÃN Táº¤T HUáº¤N LUYá»N TRONG {(time.time() - t_total_start)/60.0:.1f} PHÃT!", flush=True)


if __name__ == '__main__':
    main()
')

worker_p3_path = os.path.join(WORKING_DIR, "ddp_worker_tps_sequence_phase3.py")
with open(worker_p3_path, "w", encoding="utf-8") as f:
    f.write('#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddp_worker_tps_sequence_phase3.py
--------------------------------------------------------------------------------
Äá»NG CÆ  HUáº¤N LUYá»N DDP 2x GPU PHA 3 â COARSE TPS ÄA Lá»P (MULTI-LAYER CLUSTERED TPS)
Káº¿ thá»«a 100% kiáº¿n trÃºc `TPSInbetweenOriginal` (10,068,668 active params â 0% sá»­a kiáº¿n trÃºc)
vÃ  tinh chá»nh nháº¹ (2-3 Epochs) tá»« `model_tps_blended.pt` Äá» máº¡ng neural thÃ­ch nghi hoÃ n toÃ n
vá»i Äáº·c tÃ­nh trÆ°á»ng dÃ²ng cháº£y Äa lá»p (Static Layer + Clustered Dynamic Layers + Voronoi Mask).

CÃ¡c nÃ¢ng cáº¥p cá»t lÃµi cá»§a Pha 3 (BÆ°á»c 4):
  1. Náº¡p trá»±c tiáº¿p bá» cache `tps_cache_multilayer.npz` (hoáº·c tÆ°Æ¡ng thÃ­ch fallback single-layer)
     vÃ  tÃ¡i táº¡o batch flow Äa lá»p siÃªu tá»c trÃªn GPU (`reconstruct_multilayer_flow_batch_from_cache`).
  2. LÃ m giÃ u dá»¯ liá»u cÃ³ chá»§ ÄÃ­ch táº¡i chá» (Zero-cost Compositional Multi-Layer Enrichment):
     - Static-Background Compositing (`--static_aug_prob 0.25`) trÃªn chuá»i Sakuga thá»±c táº¿:
       trá»n ná»n kiáº¿n trÃºc/ÄÆ°á»ng chÃ¢n trá»i tÄ©nh vÃ o (x0, x1, xk_all) kÃ¨m static keypoints.
     - NÃ¢ng cáº¥p 100% chuá»i hÃ¬nh há»c tá»ng há»£p (`--synth_count 180`) sang Multi-Layer TPS vá»i 6 Mode,
       bá» sung Mode Occlusion Crossing (váº­t thá» Äá»ng lÆ°á»t qua nÃ©t tÄ©nh) & Multi-Cluster Crossing.
  3. CÃ¢n báº±ng trá»ng sá» Loss Pha 3 (`w_ghost=0.65, w_temp=0.55`) + LR pháº«u thuáº­t (`1.0e-5 / 2.5e-5`)
     Äá» vá»«a lÃ m sáº¡ch ranh giá»i giao thoa, vá»«a giá»¯ vá»¯ng máº­t Äá» nÃ©t chuáº©n ~1.00 cá»§a Blended Model.
--------------------------------------------------------------------------------
"""

import os
import sys
import gc
import time
import math
import json
import glob
import random
import argparse
import contextlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

torch.backends.cudnn.benchmark = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in (SCRIPT_DIR, os.path.join(REPO_ROOT, "pipelines")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from multi_layer_tps import (
    FIXED_TPS_ANCHORS,
    solve_multilayer_cache_entry,
    reconstruct_multilayer_flow_batch_from_cache,
)

# Import cÃ¡c thÃ nh pháº§n lÃµi khÃ´ng thay Äá»i (TPSInbetweenOriginal, Loss, Metrics, Sampler, EMA, Manager)
_P2_DIR = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_phase2")
if os.path.isdir(_P2_DIR) and _P2_DIR not in sys.path:
    sys.path.insert(0, _P2_DIR)

try:
    from ddp_worker_tps_sequence_phase2 import (
        trim_memory,
        TPSInbetweenOriginal,
        load_tps_weights,
        solve_tps_backward_pair,
        tps_flow_from_theta_batch,
        AdaptiveLineartLoss,
        BidirectionalStrokeWidthLoss,
        SoftClDiceLoss,
        VGGPerceptualOrLaplacianLoss,
        approx_gpu_distance_map,
        soft_dilate,
        soft_skeletonize,
        evaluate_metrics_dual_regime,
        DDPLengthWidthBatchSampler,
        AdaptiveEMA,
        TwoFileCheckpointManager,
        _apply_paired_spatial_aug_gpu,
    )
    _IMPORTED_P2 = True
except ImportError:
    _IMPORTED_P2 = False

if not _IMPORTED_P2:
    # Khi cháº¡y Äá»c láº­p trÃªn Kaggle, ddp_worker_tps_sequence_phase2.py ÄÆ°á»£c Cell 1 ghi chung cÃ¹ng thÆ° má»¥c
    raise ImportError("KhÃ´ng tÃ¬m tháº¥y module ná»n táº£ng `ddp_worker_tps_sequence_phase2.py`!")


# ==============================================================================
# 0b. HÃM Máº¤T MÃT PHA 3 CÃN Báº°NG: Báº¢O TOÃN SKEL / DENSITY / WCD_DOG & Sáº CH Ná»N XA
# ==============================================================================
class Phase3ExactDensityLoss(nn.Module):
    """
    Siáº¿t cháº·t sÃ n máº­t Äá» nÃ©t (dead_zone_low=0.99) vÃ  tÄng pháº¡t lÃ m nháº¡t/máº£nh nÃ©t (3.2*dev_low + 1.5*dev_low^2)
    Äá» ngÄn cháº·n hoÃ n toÃ n hiá»n tÆ°á»£ng trÃ´i máº£nh nÃ©t khi fine-tune tá»« `model_tps_blended.pt`.
    """
    def __init__(self, tau=0.15, dead_zone_low=0.99, dead_zone_high=1.04):
        super().__init__()
        self.tau = tau
        self.low = dead_zone_low
        self.high = dead_zone_high

    def forward(self, pred, gt):
        s_p = torch.clamp((pred - 0.06) / 0.18, 0.0, 1.0) + 0.25 * F.relu(pred - 0.24)
        h_p = (pred >= self.tau).float()
        m_p = h_p - s_p.detach() + s_p

        h_g = (gt >= self.tau).float()
        area_p = torch.sum(m_p, dim=(-2, -1))
        area_g = torch.sum(h_g, dim=(-2, -1)).detach()
        r = area_p / torch.clamp(area_g, min=1.0)

        dev_low = F.relu(self.low - r)
        dev_high = F.relu(r - self.high)
        dev_center = r - 1.0
        loss = (
            3.2 * dev_low + 1.5 * (dev_low ** 2)
            + 2.8 * dev_high + 1.5 * (dev_high ** 2)
            + 0.12 * (dev_center ** 2)
        )
        return loss.mean()


class Phase3ZoneCleanlinessLoss(nn.Module):
    """
    LÃ m sáº¡ch cÃ³ chá»n lá»c (Surgical Background Cleanliness):
      - Ná»n xa (> 2px khá»i nÃ©t GT): dá»n sÆ°Æ¡ng má» vÃ  Äáº·c biá»t pháº¡t máº¡nh má»±c Äáº­m sai vá» trÃ­ (Dark Ghost >= 0.20).
      - VÃ nh sÃ¡t nÃ©t (near_halo): chá» pháº¡t má»±c >= 0.12 (ná»i tá»« 0.06 -> 0.12) Äá» báº£o toÃ n 100% tÃ³c máº£nh anti-aliased!
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        with torch.no_grad():
            gt_stroke_wide = soft_dilate(soft_dilate((gt > 0.04).float()))
            far_bg = torch.clamp(1.0 - gt_stroke_wide, 0.0, 1.0)
            near_halo = torch.clamp(gt_stroke_wide - (gt > 0.12).float(), 0.0, 1.0)
            gt_area = torch.clamp(torch.sum((gt > 0.15).float(), dim=(-3, -2, -1)), min=500.0)

        near_count = torch.clamp(torch.sum(near_halo, dim=(-3, -2, -1)), min=200.0)
        pen_far_p045 = F.relu(pred - 0.045)
        pen_far = (
            1.0 * F.relu(pred - 0.018)
            + 2.5 * pen_far_p045
            + 3.0 * (pen_far_p045 ** 2)
            + 9.0 * F.relu(pred - 0.20)
        )
        loss_far = torch.sum(far_bg * pen_far, dim=(-3, -2, -1)) / gt_area

        pen_halo = F.relu(pred - 0.12)
        loss_halo = 0.35 * torch.sum(near_halo * pen_halo, dim=(-3, -2, -1)) / near_count
        return (loss_far + loss_halo).mean()


class Phase3BalancedLineartLoss(AdaptiveLineartLoss):
    """
    Káº¿ thá»«a `AdaptiveLineartLoss` nhÆ°ng thay tháº¿ cÃ¡c thÃ nh pháº§n gÃ¢y táº©y máº£nh nÃ©t báº±ng cáº¥u hÃ¬nh cÃ¢n báº±ng Pha 3 v2:
      - w_dt = 0.40 (tá»i Æ°u WCD_DoG)
      - w_dens = 0.70 (giá»¯ cháº·t Density Ratio ~ 0.98 - 1.02)
      - w_width = 0.35 (báº£o vá» lÃµi skeleton)
      - w_cldice = 0.28 (báº£o vá» ngá»n tÃ³c máº£nh & Skeleton Recall)
      - w_ghost = 0.32 (vá»«a dá»n Dark Ghost, khÃ´ng lÃ m má» nÃ©t)
    """
    def __init__(self, train_mode="LONG", is_rank0=True, w_ghost=0.32, w_temp=0.55):
        super().__init__(train_mode=train_mode, is_rank0=is_rank0, w_ghost=w_ghost, w_temp=w_temp)
        self.w_dt = 0.40
        self.w_dens = 0.70
        self.w_width = 0.35
        self.w_cldice = 0.28 if self.train_mode == "LONG" else self.w_cldice
        self.density_loss = Phase3ExactDensityLoss(tau=0.15, dead_zone_low=0.99, dead_zone_high=1.04)
        self.ghost_loss = Phase3ZoneCleanlinessLoss()
        self.stroke_width_loss = BidirectionalStrokeWidthLoss()


# ==============================================================================
# 1. LÃM GIÃU Dá»® LIá»U ÄA Lá»P Táº I CHá» (ON-THE-FLY MULTI-LAYER ENRICHMENT)
# ==============================================================================
def _inject_static_background_on_sequence(x0, x1, xk_all, orig_h, orig_w, cache_entry):
    """
    Trá»n lá»p ná»n ÄÆ°á»ng tháº³ng tÄ©nh máº£nh/thÆ°a (sparse static background lineart, 1.5-2.5px)
    vÃ o chuá»i Sakuga thá»±c táº¿ Äá» khÃ´ng lÃ m loÃ£ng máº­t Äá» má»±c nhÃ¢n váº­t mÃ  váº«n rÃ¨n luyá»n
    xá»­ lÃ½ ranh giá»i giao thoa che khuáº¥t (Occlusion Crossing).
    """
    if cv2 is None or cache_entry['num_clusters'] <= 0:
        return x0, x1, xk_all, cache_entry

    bg = np.zeros((orig_h, orig_w), dtype=np.float32)
    pts_st = []
    style = random.choice(["ground_horizon", "corner_frame", "single_pillar"])
    th = int(round(random.uniform(1.5, 2.5)))
    intensity = random.uniform(0.80, 1.0)

    if style == "ground_horizon":
        y_l = random.uniform(0.68 * orig_h, 0.88 * orig_h)
        x_a, x_b = random.uniform(0.08 * orig_w, 0.20 * orig_w), random.uniform(0.80 * orig_w, 0.92 * orig_w)
        cv2.line(bg, (int(x_a), int(y_l)), (int(x_b), int(y_l)), intensity, th, lineType=cv2.LINE_AA)
        for xv in np.linspace(x_a, x_b, 20):
            pts_st.append([xv / max(1.0, orig_w - 1), y_l / max(1.0, orig_h - 1)])
    elif style == "corner_frame":
        bx0 = random.uniform(0.10 * orig_w, 0.25 * orig_w)
        by0 = random.uniform(0.60 * orig_h, 0.78 * orig_h)
        bx1 = random.uniform(0.75 * orig_w, 0.90 * orig_w)
        by1 = random.uniform(0.14 * orig_h, 0.35 * orig_h)
        cv2.line(bg, (int(bx0), int(by0)), (int(bx1), int(by0)), intensity, th, lineType=cv2.LINE_AA)
        cv2.line(bg, (int(bx0), int(by0)), (int(bx0), int(by1)), intensity, th, lineType=cv2.LINE_AA)
        for xv in np.linspace(bx0, bx1, 12):
            pts_st.append([xv / max(1.0, orig_w - 1), by0 / max(1.0, orig_h - 1)])
        for yv in np.linspace(by1, by0, 10):
            pts_st.append([bx0 / max(1.0, orig_w - 1), yv / max(1.0, orig_h - 1)])
    else:
        xv = random.uniform(0.18 * orig_w, 0.82 * orig_w)
        ya, yb = random.uniform(0.12 * orig_h, 0.24 * orig_h), random.uniform(0.76 * orig_h, 0.88 * orig_h)
        cv2.line(bg, (int(xv), int(ya)), (int(xv), int(yb)), intensity, th, lineType=cv2.LINE_AA)
        for yv in np.linspace(ya, yb, 16):
            pts_st.append([xv / max(1.0, orig_w - 1), yv / max(1.0, orig_h - 1)])

    x0_aug = np.maximum(x0, bg)
    x1_aug = np.maximum(x1, bg)
    xk_aug = np.maximum(xk_all, bg[None, :, :])

    pts_st_np = np.asarray(pts_st, dtype=np.float32)
    max_s = cache_entry['static_pts0'].shape[0]
    cur_ns = int(cache_entry['num_static'])
    sp0 = cache_entry['static_pts0'].copy()
    sp1 = cache_entry['static_pts1'].copy()
    avail = max(0, max_s - cur_ns)
    n_add = min(avail, len(pts_st_np))
    if n_add > 0:
        sp0[cur_ns:cur_ns + n_add] = pts_st_np[:n_add]
        sp1[cur_ns:cur_ns + n_add] = pts_st_np[:n_add]
        cache_entry = dict(cache_entry)
        cache_entry['static_pts0'] = sp0
        cache_entry['static_pts1'] = sp1
        cache_entry['num_static'] = cur_ns + n_add

    return x0_aug, x1_aug, xk_aug, cache_entry


def _generate_multilayer_synthetic_sequence(seq_len, orig_h=540, orig_w=960, max_pts=128, max_clusters=4, lambd=0.01):
    """
    Sinh chuá»i hÃ¬nh há»c tá»ng há»£p 540p sá»­ dá»¥ng 100% Multi-Layer Clustered TPS (Direct-t) vá»i 6 ká»ch báº£n:
      - Mode 1 (25%): HÃ¬nh trÃ²n/elip thu phÃ³ng & dá»ch xa
      - Mode 2 (25%): HÃ¬nh trÃ²n/elip bay + ÄÆ°á»ng ná»n / khung phÃ²ng tÄ©nh (Case 3)
      - Mode 3 (15%): Cung cong má» xoay & biáº¿n dáº¡ng
      - Mode 4 (15%): Hai hÃ¬nh trÃ²n chuyá»n Äá»ng song song Äá»c láº­p
      - Mode 5 (10%): Occlusion Crossing â váº­t thá» chuyá»n Äá»ng cáº¯t ngang trá»±c diá»n qua ÄÆ°á»ng tÄ©nh
      - Mode 6 (10%): Multi-Cluster Crossing â hai váº­t thá» chuyá»n Äá»ng ngÆ°á»£c/chÃ©o chiá»u nhau
    """
    def _ellipse_pts(cx, cy, rx, ry, rot_rad, n_pts=32):
        ang = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
        lx, ly = rx * np.cos(ang), ry * np.sin(ang)
        ca, sa = math.cos(rot_rad), math.sin(rot_rad)
        px = cx + lx * ca - ly * sa
        py = cy + lx * sa + ly * ca
        return np.stack([px / float(orig_w - 1), py / float(orig_h - 1)], axis=1).astype(np.float32)

    mode_roll = random.random()
    frames = []
    kps_per_t = []
    t_list = [float(k) / float(seq_len - 1) for k in range(seq_len)]

    if mode_roll < 0.50:
        # Mode 1 (25%) & Mode 2 (25%): Circle/oval + optional static ground/box
        has_static = (mode_roll >= 0.25)
        y_max_c = 340 if has_static else 420
        rx0 = random.uniform(26.0, 92.0)
        ry0 = rx0 * random.uniform(0.75, 1.25)
        scale_r = random.uniform(0.42, 2.20)
        rx1 = float(np.clip(rx0 * scale_r, 18.0, 96.0))
        ry1 = float(np.clip(ry0 * scale_r * random.uniform(0.85, 1.15), 18.0, 96.0))
        cx0 = random.uniform(110.0, 420.0)
        cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(80.0, 420.0), 100.0, orig_w - 100.0))
        if random.random() < 0.5:
            cx0, cx1 = cx1, cx0
        cy0 = random.uniform(115.0, y_max_c - 60.0)
        cy1 = float(np.clip(cy0 + random.uniform(-90.0, 90.0), 105.0, y_max_c - 40.0))
        rot0 = random.uniform(-0.35, 0.35)
        rot1 = rot0 + random.uniform(-0.35, 0.35)
        th0, th1 = random.uniform(2.0, 5.0), random.uniform(2.0, 5.0)

        st_type = random.choice(["line", "rect"])
        st_y = random.uniform(410.0, 490.0)
        st_x0, st_x1 = random.uniform(80.0, 200.0), random.uniform(orig_w - 200.0, orig_w - 80.0)
        st_th = int(round(random.uniform(2.5, 4.5)))
        if st_type == "line":
            xs_st = np.linspace(st_x0, st_x1, 24)
            pts_st = np.stack([xs_st / float(orig_w - 1), np.full_like(xs_st, st_y) / float(orig_h - 1)], axis=1).astype(np.float32)
        else:
            bx0, by0 = random.uniform(120.0, 740.0), st_y - 45.0
            bw, bh = random.uniform(70.0, 140.0), random.uniform(40.0, 65.0)
            pts_st = np.vstack([
                np.stack([np.linspace(bx0, bx0 + bw, 7)[:-1], np.full(6, by0)], axis=1),
                np.stack([np.full(6, bx0 + bw), np.linspace(by0, by0 + bh, 7)[:-1]], axis=1),
                np.stack([np.linspace(bx0 + bw, bx0, 7)[:-1], np.full(6, by0 + bh)], axis=1),
                np.stack([np.full(6, bx0), np.linspace(by0 + bh, by0, 7)[:-1]], axis=1),
            ])
            pts_st[:, 0] /= float(orig_w - 1)
            pts_st[:, 1] /= float(orig_h - 1)
            pts_st = pts_st.astype(np.float32)

        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
            rx, ry = (1.0 - t) * rx0 + t * rx1, (1.0 - t) * ry0 + t * ry1
            rot = (1.0 - t) * rot0 + t * rot1
            th = max(2, int(round((1.0 - t) * th0 + t * th1)))
            if cv2 is not None:
                cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), math.degrees(rot), 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                if has_static:
                    if st_type == "line":
                        cv2.line(im, (int(st_x0), int(st_y)), (int(st_x1), int(st_y)), 1.0, st_th, lineType=cv2.LINE_AA)
                    else:
                        cv2.rectangle(im, (int(bx0), int(by0)), (int(bx0 + bw), int(by0 + bh)), 1.0, st_th, lineType=cv2.LINE_AA)
            pts_mv = _ellipse_pts(cx, cy, rx, ry, rot, n_pts=32)
            frames.append(im)
            kps_per_t.append(np.vstack([pts_mv, pts_st]) if has_static else pts_mv)
    elif mode_roll < 0.65:
        # Mode 3 (15%): Smooth curved arc
        n_ctrl = 24
        s_u = np.linspace(-1.0, 1.0, n_ctrl)
        base_len = random.uniform(60.0, 140.0)
        bend0, bend1 = random.uniform(-55.0, 55.0), random.uniform(-55.0, 55.0)
        cx0, cy0 = random.uniform(160.0, 440.0), random.uniform(140.0, 400.0)
        cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(60.0, 320.0), 140.0, orig_w - 140.0))
        cy1 = float(np.clip(cy0 + random.uniform(-100.0, 100.0), 120.0, orig_h - 120.0))
        sc0, sc1 = random.uniform(0.75, 1.25), random.uniform(0.65, 1.45)
        ang0 = random.uniform(-0.7, 0.7)
        ang1 = ang0 + random.uniform(-0.6, 0.6)
        th = int(round(random.uniform(2.0, 4.5)))

        def _arc_px(t):
            cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
            sc, ang, bend = (1.0 - t) * sc0 + t * sc1, (1.0 - t) * ang0 + t * ang1, (1.0 - t) * bend0 + t * bend1
            lx, ly = s_u * base_len * sc, bend * sc * (1.0 - s_u ** 2)
            ca, sa = math.cos(ang), math.sin(ang)
            return np.stack([cx + lx * ca - ly * sa, cy + lx * sa + ly * ca], axis=1)

        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            p_px = _arc_px(t)
            if cv2 is not None:
                cv2.polylines(im, [np.round(p_px).astype(np.int32)], False, 1.0, th, lineType=cv2.LINE_AA)
            frames.append(im)
            kps_per_t.append(np.stack([p_px[:, 0] / float(orig_w - 1), p_px[:, 1] / float(orig_h - 1)], axis=1).astype(np.float32))
    elif mode_roll < 0.80:
        # Mode 4 (15%): Two independent moving circles/ovals
        lanes = [(145.0, 40.0), (385.0, 40.0)]
        params = []
        for cy_base, cy_jit in lanes:
            rx0 = random.uniform(24.0, 62.0)
            ry0 = rx0 * random.uniform(0.8, 1.2)
            sc = random.uniform(0.55, 1.80)
            rx1, ry1 = float(np.clip(rx0 * sc, 18.0, 72.0)), float(np.clip(ry0 * sc, 18.0, 72.0))
            cx0 = random.uniform(120.0, 450.0)
            cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(80.0, 350.0), 100.0, orig_w - 100.0))
            params.append((cx0, cy_base + random.uniform(-cy_jit, cy_jit), rx0, ry0, cx1, cy_base + random.uniform(-cy_jit, cy_jit), rx1, ry1))
        th = int(round(random.uniform(2.5, 4.5)))
        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            pts_two = []
            for (cx0, cy0, rx0, ry0, cx1, cy1, rx1, ry1) in params:
                cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
                rx, ry = (1.0 - t) * rx0 + t * rx1, (1.0 - t) * ry0 + t * ry1
                if cv2 is not None:
                    cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                pts_two.append(_ellipse_pts(cx, cy, rx, ry, 0.0, n_pts=24))
            frames.append(im)
            kps_per_t.append(np.vstack(pts_two))
    elif mode_roll < 0.90:
        # Mode 5 (10%): Occlusion Crossing â hÃ¬nh trÃ²n/elip chuyá»n Äá»ng cáº¯t qua ÄÆ°á»ng káº» tÄ©nh
        rx0 = random.uniform(35.0, 70.0)
        ry0 = rx0 * random.uniform(0.85, 1.15)
        rx1, ry1 = rx0 * random.uniform(0.65, 1.35), ry0 * random.uniform(0.65, 1.35)
        cx0, cx1 = random.uniform(140.0, 260.0), random.uniform(680.0, 820.0)
        if random.random() < 0.5:
            cx0, cx1 = cx1, cx0
        st_y = random.uniform(210.0, 330.0)
        cy0 = st_y + random.choice([-1.0, 1.0]) * random.uniform(60.0, 120.0)
        cy1 = st_y - (cy0 - st_y) * random.uniform(0.6, 1.1)
        xs_st = np.linspace(100.0, orig_w - 100.0, 28)
        pts_st = np.stack([xs_st / float(orig_w - 1), np.full_like(xs_st, st_y) / float(orig_h - 1)], axis=1).astype(np.float32)
        th = int(round(random.uniform(2.5, 4.0)))
        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
            rx, ry = (1.0 - t) * rx0 + t * rx1, (1.0 - t) * ry0 + t * ry1
            if cv2 is not None:
                cv2.line(im, (100, int(round(st_y))), (orig_w - 100, int(round(st_y))), 1.0, th, lineType=cv2.LINE_AA)
                cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
            frames.append(im)
            kps_per_t.append(np.vstack([_ellipse_pts(cx, cy, rx, ry, 0.0, n_pts=32), pts_st]))
    else:
        # Mode 6 (10%): Multi-Cluster Crossing â hai hÃ¬nh trÃ²n cháº¡y ngÆ°á»£c/chÃ©o chiá»u
        rx_a, ry_a = random.uniform(30.0, 55.0), random.uniform(30.0, 55.0)
        rx_b, ry_b = random.uniform(28.0, 50.0), random.uniform(28.0, 50.0)
        p_a = (150.0, random.uniform(130.0, 220.0), 800.0, random.uniform(310.0, 410.0))
        p_b = (800.0, random.uniform(130.0, 220.0), 150.0, random.uniform(310.0, 410.0))
        th = int(round(random.uniform(2.5, 4.0)))
        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            cxa, cya = (1.0 - t) * p_a[0] + t * p_a[2], (1.0 - t) * p_a[1] + t * p_a[3]
            cxb, cyb = (1.0 - t) * p_b[0] + t * p_b[2], (1.0 - t) * p_b[1] + t * p_b[3]
            if cv2 is not None:
                cv2.ellipse(im, (int(round(cxa)), int(round(cya))), (int(round(rx_a)), int(round(ry_a))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                cv2.ellipse(im, (int(round(cxb)), int(round(cyb))), (int(round(rx_b)), int(round(ry_b))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
            pts_ab = np.vstack([
                _ellipse_pts(cxa, cya, rx_a, ry_a, 0.0, n_pts=24),
                _ellipse_pts(cxb, cyb, rx_b, ry_b, 0.0, n_pts=24),
            ])
            frames.append(im)
            kps_per_t.append(pts_ab)

    x0, x1 = frames[0], frames[-1]
    xk_all = np.stack(frames[1:-1], axis=0)
    kps0, kps1 = kps_per_t[0], kps_per_t[-1]
    t_mid = np.array(t_list[1:-1], dtype=np.float32)

    # Giáº£i Multi-Layer TPS Cache Entry (Direct-t) cho tá»«ng khung hÃ¬nh trung gian m_idx
    m_steps = seq_len - 2
    max_mask_pts, max_static_pts = 64, 96
    c0_m = np.zeros((m_steps, max_clusters, max_pts, 2), dtype=np.float32)
    th01_m = np.zeros((m_steps, max_clusters, max_pts + 3, 2), dtype=np.float32)
    c1_m = np.zeros((m_steps, max_clusters, max_pts, 2), dtype=np.float32)
    th10_m = np.zeros((m_steps, max_clusters, max_pts + 3, 2), dtype=np.float32)
    val_m = np.zeros((m_steps, max_clusters), dtype=np.float32)
    dpts0_m = np.zeros((m_steps, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dpts1_m = np.zeros((m_steps, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dlen_m = np.zeros((m_steps, max_clusters), dtype=np.int32)
    spts0_m = np.zeros((m_steps, max_static_pts, 2), dtype=np.float32)
    spts1_m = np.zeros((m_steps, max_static_pts, 2), dtype=np.float32)
    nstat_m = np.zeros((m_steps,), dtype=np.int32)
    nclus_m = np.zeros((m_steps,), dtype=np.int32)

    from multi_layer_tps import segment_motion_keypoints, solve_tps_dynamic_layer, _stratified_spatial_subsample
    seg = segment_motion_keypoints(kps0, kps1, img_hw=(orig_h, orig_w), eps_static_px=1.5, filter_outliers=False)
    clusters = seg['dynamic_clusters'][:max_clusters]
    if seg['num_static'] > 0:
        if seg['num_static'] > max_static_pts:
            s0_sub, s1_sub = _stratified_spatial_subsample(seg['static_kps0'], seg['static_kps1'], max_static_pts)
        else:
            s0_sub, s1_sub = seg['static_kps0'], seg['static_kps1']
        n_stat = len(s0_sub)
    else:
        s0_sub = s1_sub = np.zeros((0, 2), dtype=np.float32)
        n_stat = 0

    for m_idx in range(m_steps):
        tv = float(t_mid[m_idx])
        if n_stat > 0:
            spts0_m[m_idx, :n_stat] = s0_sub.astype(np.float32)
            spts1_m[m_idx, :n_stat] = s1_sub.astype(np.float32)
        nstat_m[m_idx] = n_stat
        nclus_m[m_idx] = len(clusters)

        for k, dc in enumerate(clusters):
            p0_k, p1_k = dc['kps0'], dc['kps1']
            pt_k = (1.0 - tv) * p0_k + tv * p1_k
            ct0, tht0, v0 = solve_tps_dynamic_layer(pt_k, p0_k, max_pts=max_pts, lambd=lambd, use_anchors=False)
            ct1, tht1, v1 = solve_tps_dynamic_layer(pt_k, p1_k, max_pts=max_pts, lambd=lambd, use_anchors=False)
            c0_m[m_idx, k], th01_m[m_idx, k] = ct0, tht0
            c1_m[m_idx, k], th10_m[m_idx, k] = ct1, tht1
            val_m[m_idx, k] = 1.0 if (v0 > 0.5 and v1 > 0.5) else 0.0

            # ÄÆ°a pt_k vÃ o dyn_pts0/dyn_pts1 Äá» máº·t náº¡ Voronoi Äáº·t tÃ¢m táº¡i ÄÃºng quá»¹ Äáº¡o t
            if len(pt_k) > max_mask_pts:
                pt_sub, _ = _stratified_spatial_subsample(pt_k, pt_k, max_mask_pts)
            else:
                pt_sub = pt_k
            n_mk = len(pt_sub)
            dpts0_m[m_idx, k, :n_mk] = pt_sub.astype(np.float32)
            dpts1_m[m_idx, k, :n_mk] = pt_sub.astype(np.float32)
            dlen_m[m_idx, k] = n_mk

    return {
        'is_sequence': True,
        'is_direct_t_flows': True,
        'is_multilayer_cache': True,
        'x0': torch.from_numpy(x0).unsqueeze(0),
        'x1': torch.from_numpy(x1).unsqueeze(0),
        'xk_all': torch.from_numpy(xk_all).unsqueeze(1),
        't_vals': torch.from_numpy(t_mid),
        'c0_dyn_all': torch.from_numpy(c0_m),
        'th01_dyn_all': torch.from_numpy(th01_m),
        'c1_dyn_all': torch.from_numpy(c1_m),
        'th10_dyn_all': torch.from_numpy(th10_m),
        'valid_dyn_all': torch.from_numpy(val_m),
        'dyn_pts0_all': torch.from_numpy(dpts0_m),
        'dyn_pts1_all': torch.from_numpy(dpts1_m),
        'dyn_pts_len_all': torch.from_numpy(dlen_m),
        'static_pts0_all': torch.from_numpy(spts0_m),
        'static_pts1_all': torch.from_numpy(spts1_m),
        'num_static_all': torch.from_numpy(nstat_m),
        'num_clusters_all': torch.from_numpy(nclus_m),
        'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
        'seq_len': seq_len,
        'clip_id': f"synth_ml_N{seq_len}",
        'scene_id': "synth_geom_ml",
        'mean_disp': 120.0,
    }


# ==============================================================================
# 2. DATASET ÄA Lá»P (Äá»C TPS_CACHE_MULTILAYER.NPZ + STATIC BG AUGMENTATION)
# ==============================================================================
class SakugaMultiLayerDataset(Dataset):
    def __init__(self, metadata_csv, clips_dir, tps_cache_path, train_mode="LONG", is_train=True, seed=42, synth_count=0, static_aug_prob=0.25):
        super().__init__()
        self.clips_dir = clips_dir
        self.train_mode = train_mode.upper()
        self.is_train = is_train
        self.static_aug_prob = float(static_aug_prob) if is_train else 0.0
        self.df = pd.read_csv(metadata_csv).reset_index(drop=True)

        self.tps_cache_path = tps_cache_path
        with np.load(tps_cache_path) as cache_data:
            self.is_multilayer = ('c0_dyn' in cache_data.files)
            self.orig_hw_arr = np.array(cache_data['orig_hw'])
            if self.is_multilayer:
                self.c0_dyn = np.array(cache_data['c0_dyn'])
                self.th01_dyn = np.array(cache_data['th01_dyn'])
                self.c1_dyn = np.array(cache_data['c1_dyn'])
                self.th10_dyn = np.array(cache_data['th10_dyn'])
                self.valid_dyn = np.array(cache_data['valid_dyn'])
                self.dyn_pts0 = np.array(cache_data['dyn_pts0'])
                self.dyn_pts1 = np.array(cache_data['dyn_pts1'])
                self.dyn_pts_len = np.array(cache_data['dyn_pts_len'])
                self.static_pts0 = np.array(cache_data['static_pts0'])
                self.static_pts1 = np.array(cache_data['static_pts1'])
                self.num_static = np.array(cache_data['num_static'])
                self.num_clusters = np.array(cache_data['num_clusters'])
            else:
                self.c0_arr = np.array(cache_data['c0'])
                self.th01_arr = np.array(cache_data['th01'])
                self.c1_arr = np.array(cache_data['c1'])
                self.th10_arr = np.array(cache_data['th10'])
                self.valid_arr = np.array(cache_data['valid'])

        self.samples = []
        if self.is_train and self.train_mode == "LONG":
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                t_list = [float(k) / float(seq_len - 1) for k in range(1, seq_len - 1)]
                self.samples.append({
                    'is_synth': False,
                    'clip_id': cid,
                    'scene_id': str(row.get('scene_id', '')),
                    'seq_len': seq_len,
                    'cache_idx': cache_idx,
                    'orig_h': int(hw[0]),
                    'orig_w': int(hw[1]),
                    't_list': t_list,
                    'mean_disp': float(row.get('mean_disp', 0.0)),
                })
            n_synth = int(synth_count) if (cv2 is not None and synth_count > 0) else 0
            for s_i in range(n_synth):
                sl = (3, 5, 7)[s_i % 3]
                self.samples.append({
                    'is_synth': True,
                    'clip_id': f"synth_ml_{s_i:04d}_N{sl}",
                    'scene_id': "synth_geom_ml",
                    'seq_len': sl,
                    'cache_idx': -1,
                    'orig_h': 540,
                    'orig_w': 960,
                    't_list': [float(k) / float(sl - 1) for k in range(1, sl - 1)],
                    'mean_disp': 120.0,
                })
        else:
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                for k in range(1, seq_len - 1):
                    self.samples.append({
                        'is_synth': False,
                        'clip_id': cid,
                        'scene_id': str(row.get('scene_id', '')),
                        'seq_len': seq_len,
                        'cache_idx': cache_idx,
                        'orig_h': int(hw[0]),
                        'orig_w': int(hw[1]),
                        'frame_k': k + 1,
                        't_val': float(k) / float(seq_len - 1),
                        'mean_disp': float(row.get('mean_disp', 0.0)),
                    })

    def __len__(self):
        return len(self.samples)

    def _read_ink(self, clip_id, frame_idx_1b):
        p = os.path.join(self.clips_dir, clip_id, f"{frame_idx_1b:06d}.png")
        with Image.open(p) as im:
            arr = np.asarray(im.convert('L'), dtype=np.float32)
        return (255.0 - arr) / 255.0

    def __getitem__(self, idx):
        s = self.samples[idx]
        if s.get('is_synth', False):
            return _generate_multilayer_synthetic_sequence(s['seq_len'], orig_h=s['orig_h'], orig_w=s['orig_w'])

        cid = s['clip_id']
        seq_len = s['seq_len']
        cache_idx = s['cache_idx']

        x0 = self._read_ink(cid, 1)
        x1 = self._read_ink(cid, seq_len)
        orig_h, orig_w = x0.shape

        if self.is_multilayer:
            c_entry = {
                'c0_dyn': np.array(self.c0_dyn[cache_idx], copy=True),
                'th01_dyn': np.array(self.th01_dyn[cache_idx], copy=True),
                'c1_dyn': np.array(self.c1_dyn[cache_idx], copy=True),
                'th10_dyn': np.array(self.th10_dyn[cache_idx], copy=True),
                'valid_dyn': np.array(self.valid_dyn[cache_idx], copy=True),
                'dyn_pts0': np.array(self.dyn_pts0[cache_idx], copy=True),
                'dyn_pts1': np.array(self.dyn_pts1[cache_idx], copy=True),
                'dyn_pts_len': np.array(self.dyn_pts_len[cache_idx], copy=True),
                'static_pts0': np.array(self.static_pts0[cache_idx], copy=True),
                'static_pts1': np.array(self.static_pts1[cache_idx], copy=True),
                'num_static': int(self.num_static[cache_idx]),
                'num_clusters': int(self.num_clusters[cache_idx]),
            }

            if self.is_train and self.train_mode == "LONG":
                xk_list = [self._read_ink(cid, f_idx) for f_idx in range(2, seq_len)]
                xk_all = np.stack(xk_list, axis=0)
                t_arr = np.array(s['t_list'], dtype=np.float32)

                # LÃ m giÃ u cÃ³ chá»§ ÄÃ­ch: trá»n ná»n tÄ©nh nhÃ¢n táº¡o vÃ o chuá»i Sakuga thá»±c táº¿
                if self.static_aug_prob > 0.0 and random.random() < self.static_aug_prob:
                    x0, x1, xk_all, c_entry = _inject_static_background_on_sequence(
                        x0, x1, xk_all, orig_h, orig_w, c_entry
                    )

                # Temporal Reversal (Äáº£o chiá»u I0 <-> I1)
                if random.random() < 0.5:
                    x0, x1 = x1, x0
                    c_entry['c0_dyn'], c_entry['c1_dyn'] = c_entry['c1_dyn'], c_entry['c0_dyn']
                    c_entry['th01_dyn'], c_entry['th10_dyn'] = c_entry['th10_dyn'], c_entry['th01_dyn']
                    c_entry['dyn_pts0'], c_entry['dyn_pts1'] = c_entry['dyn_pts1'], c_entry['dyn_pts0']
                    c_entry['static_pts0'], c_entry['static_pts1'] = c_entry['static_pts1'], c_entry['static_pts0']
                    xk_all = xk_all[::-1].copy()
                    t_arr = (1.0 - t_arr[::-1]).copy()

                return {
                    'is_sequence': True,
                    'is_direct_t_flows': False,
                    'is_multilayer_cache': True,
                    'x0': torch.from_numpy(x0).unsqueeze(0),
                    'x1': torch.from_numpy(x1).unsqueeze(0),
                    'xk_all': torch.from_numpy(xk_all).unsqueeze(1),
                    't_vals': torch.from_numpy(t_arr),
                    'c0_dyn': torch.from_numpy(c_entry['c0_dyn']),
                    'th01_dyn': torch.from_numpy(c_entry['th01_dyn']),
                    'c1_dyn': torch.from_numpy(c_entry['c1_dyn']),
                    'th10_dyn': torch.from_numpy(c_entry['th10_dyn']),
                    'valid_dyn': torch.from_numpy(c_entry['valid_dyn']),
                    'dyn_pts0': torch.from_numpy(c_entry['dyn_pts0']),
                    'dyn_pts1': torch.from_numpy(c_entry['dyn_pts1']),
                    'dyn_pts_len': torch.from_numpy(c_entry['dyn_pts_len']),
                    'static_pts0': torch.from_numpy(c_entry['static_pts0']),
                    'static_pts1': torch.from_numpy(c_entry['static_pts1']),
                    'num_static': torch.tensor(c_entry['num_static'], dtype=torch.int32),
                    'num_clusters': torch.tensor(c_entry['num_clusters'], dtype=torch.int32),
                    'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
                    'seq_len': seq_len,
                    'clip_id': cid,
                    'scene_id': s['scene_id'],
                    'mean_disp': s['mean_disp'],
                }

            fk = s['frame_k']
            t_val = s['t_val']
            xk = self._read_ink(cid, fk)
            return {
                'is_sequence': False,
                'is_direct_t_flows': False,
                'is_multilayer_cache': True,
                'x0': torch.from_numpy(x0).unsqueeze(0),
                'x1': torch.from_numpy(x1).unsqueeze(0),
                'xk': torch.from_numpy(xk).unsqueeze(0),
                't_val': torch.tensor([t_val], dtype=torch.float32),
                'c0_dyn': torch.from_numpy(c_entry['c0_dyn']),
                'th01_dyn': torch.from_numpy(c_entry['th01_dyn']),
                'c1_dyn': torch.from_numpy(c_entry['c1_dyn']),
                'th10_dyn': torch.from_numpy(c_entry['th10_dyn']),
                'valid_dyn': torch.from_numpy(c_entry['valid_dyn']),
                'dyn_pts0': torch.from_numpy(c_entry['dyn_pts0']),
                'dyn_pts1': torch.from_numpy(c_entry['dyn_pts1']),
                'dyn_pts_len': torch.from_numpy(c_entry['dyn_pts_len']),
                'static_pts0': torch.from_numpy(c_entry['static_pts0']),
                'static_pts1': torch.from_numpy(c_entry['static_pts1']),
                'num_static': torch.tensor(c_entry['num_static'], dtype=torch.int32),
                'num_clusters': torch.tensor(c_entry['num_clusters'], dtype=torch.int32),
                'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
                'seq_len': seq_len,
                'frame_k': fk,
                'clip_id': cid,
                'scene_id': s['scene_id'],
                'mean_disp': s['mean_disp'],
            }

        raise RuntimeError("Dataset Pha 3 yÃªu cáº§u bá» cache Äa lá»p `tps_cache_multilayer.npz`!")


def pad_collate_fn_multilayer(batch):
    max_h = max(item['orig_hw'][0].item() for item in batch)
    max_w = max(item['orig_hw'][1].item() for item in batch)
    h_pad = ((max_h + 31) // 32) * 32
    w_pad = ((max_w + 31) // 32) * 32

    is_seq = batch[0]['is_sequence']
    is_direct = bool(batch[0].get('is_direct_t_flows', False))

    def _pad_t(t, th, tw):
        dh = th - t.shape[-2]
        dw = tw - t.shape[-1]
        if dh == 0 and dw == 0:
            return t
        return F.pad(t, (0, dw, 0, dh), mode='constant', value=0.0)

    out = {
        'is_sequence': is_seq,
        'is_direct_t_flows': is_direct,
        'is_multilayer_cache': True,
        'h_pad': h_pad,
        'w_pad': w_pad,
        'x0': torch.stack([_pad_t(b['x0'], h_pad, w_pad) for b in batch], dim=0),
        'x1': torch.stack([_pad_t(b['x1'], h_pad, w_pad) for b in batch], dim=0),
        'orig_hw': torch.stack([b['orig_hw'] for b in batch], dim=0),
        'seq_len': torch.tensor([b['seq_len'] for b in batch], dtype=torch.int32),
        'clip_id': [b['clip_id'] for b in batch],
        'scene_id': [b['scene_id'] for b in batch],
        'mean_disp': [b['mean_disp'] for b in batch],
    }

    if is_direct:
        for k in ['c0_dyn_all', 'th01_dyn_all', 'c1_dyn_all', 'th10_dyn_all', 'valid_dyn_all',
                  'dyn_pts0_all', 'dyn_pts1_all', 'dyn_pts_len_all',
                  'static_pts0_all', 'static_pts1_all', 'num_static_all', 'num_clusters_all']:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    else:
        for k in ['c0_dyn', 'th01_dyn', 'c1_dyn', 'th10_dyn', 'valid_dyn',
                  'dyn_pts0', 'dyn_pts1', 'dyn_pts_len',
                  'static_pts0', 'static_pts1', 'num_static', 'num_clusters']:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    if is_seq:
        out['xk_all'] = torch.stack([_pad_t(b['xk_all'], h_pad, w_pad) for b in batch], dim=0)
        out['t_vals'] = torch.stack([b['t_vals'] for b in batch], dim=0)
    else:
        out['xk'] = torch.stack([_pad_t(b['xk'], h_pad, w_pad) for b in batch], dim=0)
        out['t_val'] = torch.stack([b['t_val'] for b in batch], dim=0)
        out['frame_k'] = [b['frame_k'] for b in batch]
    return out


# ==============================================================================
# 3. VALIDATION ÄA Lá»P (THUáº¦N FP32)
# ==============================================================================
@torch.no_grad()
def run_validation_suite_multilayer(model, val_loader, device, downscale_factor=2, is_ddp=False, evaluate_m0=False):
    model.eval()
    keys_sum = ['psnr', 'ssim', 'val_l1', 'ie_mae', 'pwcr', 'rgd', 'ghost', 'dark_ghost',
                'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy', 'semd',
                'recall', 'precision', 'f1', 'density_ratio', 'skel_recall', 'tip_recall']
    acc_main = {k: 0.0 for k in keys_sum}
    acc_main['count'] = 0.0
    acc_m0 = {k: 0.0 for k in keys_sum}
    acc_m0['count'] = 0.0

    brk_keys = ['count', 'psnr', 'ssim', 'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy',
                'semd', 'skel_recall', 'tip_recall', 'f1', 'ghost', 'density_ratio']
    breakdown = {n: {k: 0.0 for k in brk_keys} for n in (3, 5, 7)}

    for batch in val_loader:
        h_pad, w_pad = batch['h_pad'], batch['w_pad']
        x0 = batch['x0'].to(device, non_blocking=True)
        x1 = batch['x1'].to(device, non_blocking=True)
        xk = batch['xk'].to(device, non_blocking=True)
        t_val = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
        orig_hw = batch['orig_hw']
        seq_lens = batch['seq_len']

        cf01, cf10 = reconstruct_multilayer_flow_batch_from_cache(
            batch['c0_dyn'].to(device, non_blocking=True),
            batch['th01_dyn'].to(device, non_blocking=True),
            batch['c1_dyn'].to(device, non_blocking=True),
            batch['th10_dyn'].to(device, non_blocking=True),
            batch['valid_dyn'].to(device, non_blocking=True),
            batch['dyn_pts0'].to(device, non_blocking=True),
            batch['dyn_pts1'].to(device, non_blocking=True),
            batch['dyn_pts_len'].to(device, non_blocking=True),
            batch['static_pts0'].to(device, non_blocking=True),
            batch['static_pts1'].to(device, non_blocking=True),
            batch['num_static'].to(device, non_blocking=True),
            batch['num_clusters'].to(device, non_blocking=True),
            orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor,
        )

        with torch.amp.autocast('cuda', enabled=False):
            pred_it, coarse_warp, *_ = model(x0.float(), x1.float(), t_val.float(), cf01.float(), cf10.float())

        b = x0.shape[0]
        for i in range(b):
            hi, wi = int(orig_hw[i, 0].item()), int(orig_hw[i, 1].item())
            sl = int(seq_lens[i].item())
            gt_i = xk[i:i + 1, :, :hi, :wi]
            pr_i = pred_it[i:i + 1, :, :hi, :wi]

            m = evaluate_metrics_dual_regime(pr_i, gt_i)
            acc_main['count'] += 1.0
            for k in keys_sum:
                acc_main[k] += float(m[k].item())

            if sl in breakdown:
                breakdown[sl]['count'] += 1.0
                for bk in brk_keys:
                    if bk != 'count':
                        breakdown[sl][bk] += float(m[bk].item())

            if evaluate_m0:
                m0_pr = coarse_warp[i:i + 1, :, :hi, :wi]
                mm0 = evaluate_metrics_dual_regime(m0_pr, gt_i)
                acc_m0['count'] += 1.0
                for k in keys_sum:
                    acc_m0[k] += float(mm0[k].item())

    def _finalize(acc, brk=None):
        if is_ddp:
            vec = torch.tensor([acc['count']] + [acc[k] for k in keys_sum], dtype=torch.float64, device=device)
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            cnt = max(1.0, float(vec[0].item()))
            out = {k: float(vec[idx + 1].item()) / cnt for idx, k in enumerate(keys_sum)}
            out['total_samples'] = int(cnt)
        else:
            cnt = max(1.0, acc['count'])
            out = {k: acc[k] / cnt for k in keys_sum}
            out['total_samples'] = int(cnt)

        if brk is not None:
            brk_out = {}
            for sl in (3, 5, 7):
                if is_ddp:
                    bvec = torch.tensor([brk[sl][k] for k in brk_keys], dtype=torch.float64, device=device)
                    dist.all_reduce(bvec, op=dist.ReduceOp.SUM)
                    bcnt = max(1.0, float(bvec[0].item()))
                    brk_out[sl] = {k: float(bvec[idx].item()) / bcnt for idx, k in enumerate(brk_keys) if k != 'count'}
                    brk_out[sl]['count'] = int(bvec[0].item())
                else:
                    bcnt = max(1.0, brk[sl]['count'])
                    brk_out[sl] = {k: brk[sl][k] / bcnt for k in brk_keys if k != 'count'}
                    brk_out[sl]['count'] = int(brk[sl]['count'])
            out['breakdown'] = brk_out
        return out

    res_main = _finalize(acc_main, breakdown)
    if evaluate_m0:
        return res_main, _finalize(acc_m0)
    return res_main


# ==============================================================================
# 4. CHÆ¯Æ NG TRÃNH HUáº¤N LUYá»N CHÃNH PHA 3 (DDP WORKER)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="DDP Worker Pha 3: Fine-tune nháº¹ trÃªn Coarse TPS Äa lá»p")
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--train_clips_dir', type=str, required=True)
    parser.add_argument('--val_clips_dir', type=str, required=True)
    parser.add_argument('--tps_cache', type=str, required=True, help="ÄÆ°á»ng dáº«n tá»i tps_cache_multilayer.npz")
    parser.add_argument('--m1_checkpoint', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, default="", help="Æ¯u tiÃªn model_tps_blended.pt")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/checkpoints_tps_finetune")
    parser.add_argument('--train_mode', type=str, default="LONG", choices=["LONG"])
    parser.add_argument('--epochs', type=int, default=4, help="Sá» epoch nháº¹ cho Pha 3 v2 (máº·c Äá»nh: 4)")
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--max_chunk_frames', type=int, default=3)
    parser.add_argument('--lr_backbone', type=float, default=6.0e-6)
    parser.add_argument('--lr_unet', type=float, default=1.8e-5)
    parser.add_argument('--ema_decay', type=float, default=0.996)
    parser.add_argument('--synth_count', type=int, default=150)
    parser.add_argument('--static_aug_prob', type=float, default=0.12)
    parser.add_argument('--w_ghost', type=float, default=0.32)
    parser.add_argument('--w_temp', type=float, default=0.55)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ram_limit_gb', type=float, default=23.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = 'nccl' if dist.is_nccl_available() else 'gloo'
        else:
            device = torch.device('cpu')
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method=os.environ.get('DDP_INIT_METHOD', 'env://'))
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank0 = (local_rank == 0)
    random.seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    tps_downscale = 2
    if is_rank0:
        init_tag = os.path.basename(args.init_checkpoint) if (args.init_checkpoint and os.path.exists(args.init_checkpoint)) else os.path.basename(args.m1_checkpoint)
        print("=" * 84)
        print(f"ð KHá»I Äá»NG HUáº¤N LUYá»N PHA 3 DDP MULTI-LAYER TPS (INIT={init_tag})")
        print(f"   World Size = {world_size} GPU(s) | Device = {device} | TPS Downscale = 1/{tps_downscale}")
        print(f"   Epochs = {args.epochs} | Synth_ML = {args.synth_count} | StaticBG_Aug = {args.static_aug_prob:.2f}")
        print(f"   w_ghost={args.w_ghost} | w_temp={args.w_temp} | LR=({args.lr_backbone:.1e}, {args.lr_unet:.1e})")
        print("=" * 84, flush=True)

    ckpt_mgr = TwoFileCheckpointManager(args.output_dir, is_rank0=is_rank0)
    base_model = TPSInbetweenOriginal().to(device)
    load_tps_weights(base_model, args.m1_checkpoint, device=device, prefer_ema=False, is_rank0=is_rank0)

    train_ds = SakugaMultiLayerDataset(
        args.train_csv, args.train_clips_dir, args.tps_cache,
        train_mode="LONG", is_train=True, seed=args.seed,
        synth_count=args.synth_count, static_aug_prob=args.static_aug_prob
    )
    val_ds = SakugaMultiLayerDataset(
        args.val_csv, args.val_clips_dir, args.tps_cache,
        train_mode="VAL", is_train=False, seed=args.seed, synth_count=0, static_aug_prob=0.0
    )

    train_sampler = DDPLengthWidthBatchSampler(
        train_ds.samples, world_size=world_size, rank=local_rank,
        batch_n3=4, batch_n5=2, batch_n7=1, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=pad_collate_fn_multilayer,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, sampler=val_sampler,
        collate_fn=pad_collate_fn_multilayer, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), persistent_workers=(args.num_workers > 0), drop_last=False
    )

    if is_rank0:
        print(f"[â] Dá»¯ liá»u Pha 3 sáºµn sÃ ng: Train={len(train_ds):,} ({len(train_loader):,} batches/GPU) | Val={len(val_ds):,}", flush=True)

    baseline_json_path = os.path.join(os.path.dirname(args.output_dir), "val_baseline_m0_m1.json")
    has_resume = os.path.exists(ckpt_mgr.last_path)
    need_baseline = (not has_resume and not args.dry_run and not os.path.exists(baseline_json_path))
    if is_ddp:
        base_flag = torch.tensor([1 if need_baseline else 0], dtype=torch.int32, device=device)
        dist.broadcast(base_flag, src=0)
        need_baseline = bool(base_flag.item())

    m0_val_cache, m1_val_cache = None, None
    if need_baseline:
        if is_rank0:
            print("\nð¬ ÄANG ÄO BASELINE EPOCH 0 TRÃN Táº¬P VAL Vá»I MULTI-LAYER TPS...", flush=True)
        m1_val, m0_val = run_validation_suite_multilayer(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=True)
        m0_val_cache, m1_val_cache = m0_val, m1_val
        if is_rank0:
            with open(baseline_json_path, 'w', encoding='utf-8') as f:
                json.dump({'M0_MultiLayer': m0_val, 'M1_Original': m1_val}, f, indent=2, ensure_ascii=False)
            print(f"ð M0_Multi (Coarse): PSNR={m0_val['psnr']:.2f}dB | WCD_DoG={m0_val['wcd_dog']:.4f} | Ghost={m0_val['ghost']:.0f}", flush=True)
            print(f"ð M1_Multi (Orig)  : PSNR={m1_val['psnr']:.2f}dB | WCD_DoG={m1_val['wcd_dog']:.4f} | Ghost={m1_val['ghost']:.0f}\n", flush=True)
        if is_ddp:
            dist.barrier()
    elif os.path.exists(baseline_json_path):
        try:
            with open(baseline_json_path, 'r', encoding='utf-8') as f:
                b_all = json.load(f)
                m0_val_cache = b_all.get('M0_MultiLayer', b_all.get('M0_Coarse', {}))
                m1_val_cache = b_all.get('M1_Original', {})
        except Exception:
            m0_val_cache, m1_val_cache = None, None

    # Náº¡p Warm-Start (model_tps_blended.pt) vÃ  Äo má»c Init chuáº©n Äá» chá»ng tá»¥t nÃ©t
    init_val_ghost = 9200.0
    init_val_skel = 0.850
    init_val_wcd_dog = 1.395
    init_val_dens = 0.960
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        if os.path.abspath(args.init_checkpoint) != os.path.abspath(args.m1_checkpoint):
            load_tps_weights(base_model, args.init_checkpoint, device=device, prefer_ema=True, is_rank0=is_rank0)
            if not has_resume and not args.dry_run:
                init_val = run_validation_suite_multilayer(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=False)
                init_val_ghost = float(init_val['ghost'])
                init_val_skel = float(init_val['skel_recall'])
                init_val_wcd_dog = float(init_val['wcd_dog'])
                init_val_dens = float(init_val['density_ratio'])
                if is_rank0:
                    print(f"ð Init Warm-Start (Blended + Multi-Layer): PSNR={init_val['psnr']:.2f}dB | WCD_DoG={init_val_wcd_dog:.4f} | Skel={init_val_skel*100:.1f}% | Dens={init_val_dens:.3f} | Ghost={init_val_ghost:.0f}\n", flush=True)

    ema = AdaptiveEMA(base_model, max_decay=args.ema_decay)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in base_model.encoder.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.IFNet.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.unet.parameters() if p.requires_grad], 'lr': args.lr_unet, 'weight_decay': 1e-4},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    if is_ddp:
        if is_rank0:
            criterion = Phase3BalancedLineartLoss(train_mode="LONG", is_rank0=True, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)
        dist.barrier()
        if not is_rank0:
            criterion = Phase3BalancedLineartLoss(train_mode="LONG", is_rank0=False, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)
        dist.barrier()
    else:
        criterion = Phase3BalancedLineartLoss(train_mode="LONG", is_rank0=True, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)

    start_epoch = 1
    history = []
    if not args.dry_run:
        last_ep, hist = ckpt_mgr.try_resume(base_model, ema, optimizer, scaler, map_location=device)
        if last_ep > 0:
            start_epoch = last_ep + 1
            history = hist
            if is_rank0:
                print(f"[â»ï¸ AUTO-RESUME] KhÃ´i phá»¥c thÃ nh cÃ´ng Epoch {last_ep:02d}. Tiáº¿p tá»¥c Epoch {start_epoch:02d}.", flush=True)

    m1_ref = m1_val_cache or {}
    base_wcd_dog = float(m1_ref.get('wcd_dog', 1.460))
    base_semd = float(m1_ref.get('semd', 0.0131))
    base_skel = float(m1_ref.get('skel_recall', 0.816))
    base_tip = float(m1_ref.get('tip_recall', 0.820))
    base_ghost = float(m1_ref.get('ghost', 1409.0))
    best_ghost_gate = init_val_ghost * 1.02

    if is_ddp:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == 'cuda' else {}
        ddp_model = DDP(base_model, find_unused_parameters=False, **ddp_kwargs)
    else:
        ddp_model = base_model

    history_csv_path = os.path.join(os.path.dirname(args.output_dir), "training_history_finetune.csv")
    watchdog_paused = False
    t_total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep0 = time.time()
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Lá»ch LR pháº«u thuáº­t 4 Epochs: Warmup -> Peak -> Cosine Decay
        if epoch == 1:
            lr_scale = 0.45
        elif epoch == 2:
            lr_scale = 1.00
        else:
            prog = float(epoch - 2) / float(max(1, args.epochs - 2))
            lr_scale = max(0.15, 0.5 * (1.0 + math.cos(math.pi * prog)))

        optimizer.param_groups[0]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[1]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[2]['lr'] = args.lr_unet * lr_scale

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_frames = 0
        num_batches = len(train_loader)
        grad_accum = max(1, args.grad_accum)

        for b_idx, batch in enumerate(train_loader):
            h_pad, w_pad = batch['h_pad'], batch['w_pad']
            is_direct = bool(batch.get('is_direct_t_flows', False))
            x0 = batch['x0'].to(device, non_blocking=True)
            x1 = batch['x1'].to(device, non_blocking=True)
            xk_all = batch['xk_all'].to(device, non_blocking=True)
            t_vals = batch['t_vals'].to(device, non_blocking=True)
            orig_hw = batch['orig_hw'].to(device, non_blocking=True)
            bsz, m_steps = xk_all.shape[0], xk_all.shape[1]

            if is_direct:
                # Tá»ng há»£p hÃ¬nh há»c Äa lá»p (Direct-t per step)
                cf01_steps, cf10_steps = [], []
                for m_i in range(m_steps):
                    cf0_i, cf1_i = reconstruct_multilayer_flow_batch_from_cache(
                        batch['c0_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['th01_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['c1_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['th10_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['valid_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts0_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts1_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts_len_all'][:, m_i].to(device, non_blocking=True),
                        batch['static_pts0_all'][:, m_i].to(device, non_blocking=True),
                        batch['static_pts1_all'][:, m_i].to(device, non_blocking=True),
                        batch['num_static_all'][:, m_i].to(device, non_blocking=True),
                        batch['num_clusters_all'][:, m_i].to(device, non_blocking=True),
                        orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale,
                        use_dyn_anchors=False, transition_width_px=18.0, falloff_radius_px=160.0
                    )
                    cf01_steps.append(cf0_i)
                    cf10_steps.append(cf1_i)
                cf01_f = torch.stack(cf01_steps, dim=1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                cf10_f = torch.stack(cf10_steps, dim=1).reshape(bsz * m_steps, 2, h_pad, w_pad)
            else:
                cf01, cf10 = reconstruct_multilayer_flow_batch_from_cache(
                    batch['c0_dyn'].to(device, non_blocking=True),
                    batch['th01_dyn'].to(device, non_blocking=True),
                    batch['c1_dyn'].to(device, non_blocking=True),
                    batch['th10_dyn'].to(device, non_blocking=True),
                    batch['valid_dyn'].to(device, non_blocking=True),
                    batch['dyn_pts0'].to(device, non_blocking=True),
                    batch['dyn_pts1'].to(device, non_blocking=True),
                    batch['dyn_pts_len'].to(device, non_blocking=True),
                    batch['static_pts0'].to(device, non_blocking=True),
                    batch['static_pts1'].to(device, non_blocking=True),
                    batch['num_static'].to(device, non_blocking=True),
                    batch['num_clusters'].to(device, non_blocking=True),
                    orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale,
                    use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
                )
                _apply_paired_spatial_aug_gpu(x0, x1, xk_all, cf01, cf10, orig_hw, is_sequence=True)
                cf01_f = cf01.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                cf10_f = cf10.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)

            x0_f = x0.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
            x1_f = x1.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
            xk_f = xk_all.reshape(bsz * m_steps, 1, h_pad, w_pad)
            t_f = t_vals.reshape(bsz * m_steps, 1, 1, 1)
            hw_f = orig_hw.unsqueeze(1).expand(-1, m_steps, -1).reshape(bsz * m_steps, 2)
            seq_ids_f = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, m_steps).reshape(-1)

            total_f = x0_f.shape[0]
            total_pairs = int((seq_ids_f[1:] == seq_ids_f[:-1]).sum().item()) if total_f >= 2 else 0
            do_opt_step = ((b_idx + 1) % grad_accum == 0) or ((b_idx + 1) == num_batches)
            chunk_sz = max(1, args.max_chunk_frames)

            batch_loss_sum = 0.0
            prev_pred_last, prev_gt_last, prev_seq_id = None, None, None

            for c_start in range(0, total_f, chunk_sz):
                c_end = min(c_start + chunk_sz, total_f)
                need_sync = (do_opt_step and c_end == total_f)
                sync_ctx = (ddp_model.no_sync() if (is_ddp and not need_sync) else contextlib.nullcontext())

                with sync_ctx:
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        pred_c, *_ = ddp_model(
                            x0_f[c_start:c_end], x1_f[c_start:c_end],
                            t_f[c_start:c_end], cf01_f[c_start:c_end], cf10_f[c_start:c_end],
                            direct_t_flows=is_direct
                        )
                        gt_c = xk_f[c_start:c_end]
                        hw_c = hw_f[c_start:c_end]
                        seq_c = seq_ids_f[c_start:c_end]
                        l_c, d_c = criterion(pred_c, gt_c, valid_hw=hw_c)
                        scaled_l = l_c * (float(c_end - c_start) / float(total_f * grad_accum))

                        if total_pairs > 0 and criterion.w_temp > 0.0:
                            l_temp_sum = pred_c.new_zeros(())
                            if pred_c.shape[0] >= 2:
                                same_m = (seq_c[1:] == seq_c[:-1])
                                if same_m.any():
                                    l_temp_sum = l_temp_sum + criterion.compute_temporal_pair_loss(
                                        pred_c[:-1][same_m], pred_c[1:][same_m],
                                        gt_c[:-1][same_m], gt_c[1:][same_m],
                                        valid_hw=hw_c[1:][same_m]
                                    )
                            if prev_pred_last is not None and int(seq_c[0].item()) == prev_seq_id:
                                l_temp_sum = l_temp_sum + criterion.compute_temporal_pair_loss(
                                    prev_pred_last, pred_c[0:1], prev_gt_last, gt_c[0:1], valid_hw=hw_c[0:1]
                                )
                            scaled_l = scaled_l + (l_temp_sum / float(total_pairs * grad_accum))

                    scaler.scale(scaled_l).backward()
                    batch_loss_sum += d_c['total_loss'] * float(c_end - c_start)
                    prev_pred_last = pred_c[-1:].detach()
                    prev_gt_last = gt_c[-1:].detach()
                    prev_seq_id = int(seq_c[-1].item())

            acc_loss += batch_loss_sum
            acc_frames += total_f

            if do_opt_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(base_model)

            if args.dry_run and b_idx >= 1:
                break

        train_min = (time.time() - t_ep0) / 60.0
        if args.dry_run:
            if is_rank0:
                print(f"[â DRY-RUN PHA 3 THÃNH CÃNG] {acc_frames} frames qua Multi-Layer DDP + StaticBG Aug + EMA! Avg Loss={acc_loss/max(1,acc_frames):.4f}", flush=True)
            break

        t_val0 = time.time()
        ema.apply_shadow(base_model)
        val_m = run_validation_suite_multilayer(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=False)
        ema.restore(base_model)
        val_min = (time.time() - t_val0) / 60.0

        dens = float(val_m['density_ratio'])
        skel = float(val_m['skel_recall'])
        tip = float(val_m['tip_recall'])
        wcd_dog = float(val_m['wcd_dog'])
        semd = float(val_m['semd'])
        ghost = float(val_m['ghost'])

        # Cá»ng kiá»m duyá»t Strict Non-Regression so vá»i Init Warm-Start (Blended):
        # KhÃ´ng cho phÃ©p tá»¥t Density, Skel Recall hay phÃ¬nh WCD_DoG!
        gate_density = (0.955 <= dens <= 1.055)
        gate_skel = (skel >= init_val_skel - 0.002)
        gate_wcd = (wcd_dog <= init_val_wcd_dog * 1.005)
        gate_ghost = (ghost <= best_ghost_gate)
        tier1_passed = bool(gate_density and gate_skel and gate_wcd and gate_ghost)

        term_wcd = 0.25 * (wcd_dog / max(1e-4, base_wcd_dog))
        term_semd = 0.08 * (semd / max(1e-4, base_semd))
        term_skel = 0.22 * ((1.0 - skel) / max(1e-4, 1.0 - base_skel))
        term_tip = 0.09 * ((1.0 - tip) / max(1e-4, 1.0 - base_tip))
        term_dens = 0.16 * (abs(dens - 1.00) / 0.10)
        term_ghost = 0.20 * (ghost / max(1500.0, base_ghost * 2.0))
        s_val = term_wcd + term_semd + term_skel + term_tip + term_dens + term_ghost

        if tier1_passed:
            comp_score = s_val
            best_ghost_gate = min(best_ghost_gate, ghost)
        else:
            over_dens = max(0.0, dens - 1.055)
            under_dens = max(0.0, 0.955 - dens)
            under_skel = max(0.0, (init_val_skel - 0.002 - skel) * 50.0)
            over_wcd = max(0.0, (wcd_dog - init_val_wcd_dog * 1.005) * 15.0)
            over_ghost = max(0.0, (ghost - best_ghost_gate) / 1000.0)
            comp_score = s_val + 10.0 + 5.0 * over_dens + 4.0 * under_dens + under_skel + over_wcd + 1.0 * over_ghost

        is_best = comp_score < ckpt_mgr.best_score

        if is_rank0:
            row = {
                'epoch': epoch, 'lr_scale': round(lr_scale, 4),
                'train_loss': acc_loss / max(1, acc_frames),
                'val_l1': val_m['val_l1'], 'psnr': val_m['psnr'], 'ssim': val_m['ssim'],
                'wcd': val_m['wcd'], 'wcd_legacy': val_m['wcd_legacy'],
                'wcd_dog': val_m['wcd_dog'], 'wcd_dog_legacy': val_m['wcd_dog_legacy'],
                'semd': val_m['semd'], 'skel_recall': val_m['skel_recall'],
                'tip_recall': val_m['tip_recall'], 'recall': val_m['recall'],
                'precision': val_m['precision'], 'f1': val_m['f1'],
                'density_ratio': val_m['density_ratio'], 'pwcr': val_m['pwcr'],
                'rgd': val_m['rgd'], 'ghost': val_m['ghost'],
                's_val_norm': s_val, 'score': comp_score,
                'gate_passed': tier1_passed, 'is_best': is_best,
                'train_time_min': train_min, 'val_time_min': val_min,
            }
            for sl in (3, 5, 7):
                if sl in val_m.get('breakdown', {}):
                    row[f'psnr_N{sl}'] = val_m['breakdown'][sl]['psnr']
                    row[f'wcd_N{sl}'] = val_m['breakdown'][sl]['wcd']
                    row[f'wcd_dog_N{sl}'] = val_m['breakdown'][sl]['wcd_dog']
                    row[f'skel_recall_N{sl}'] = val_m['breakdown'][sl]['skel_recall']
                    row[f'tip_recall_N{sl}'] = val_m['breakdown'][sl]['tip_recall']
                    row[f'ghost_N{sl}'] = val_m['breakdown'][sl]['ghost']
                    row[f'density_N{sl}'] = val_m['breakdown'][sl]['density_ratio']

            history.append(row)
            pd.DataFrame(history).to_csv(history_csv_path, index=False)
            ckpt_mgr.save_epoch(epoch, base_model, ema, optimizer, scaler, comp_score, history, is_best=is_best)

            gate_badge = "â PASS" if tier1_passed else "â ï¸ GATE_BREACH"
            print(
                f"ð EP {epoch:02d}/{args.epochs:02d} [PHA3_ML] | Train={train_min:.2f}m | Val={val_min:.2f}m | "
                f"Loss={row['train_loss']:.4f} | PSNR={val_m['psnr']:.2f}dB | "
                f"WCD_DoG={val_m['wcd_dog']:.4f} | SkelRec={val_m['skel_recall']*100:.1f}% | "
                f"Dens={dens:.3f} | Ghost={ghost:.0f} | Gate={gate_badge} | Score={comp_score:.4f}",
                flush=True
            )

        trim_memory()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
        if is_rank0:
            try:
                import psutil
                mem_gb = psutil.virtual_memory().used / (1024 ** 3)
                if mem_gb >= args.ram_limit_gb:
                    print(f"\nâ ï¸ [RAM WATCHDOG]: Host RAM Äáº¡t {mem_gb:.2f} GB >= {args.ram_limit_gb} GB. ThoÃ¡t an toÃ n mÃ£ 42!", flush=True)
                    stop_flag.fill_(1)
            except Exception:
                pass

        if is_ddp:
            dist.broadcast(stop_flag, src=0)
            dist.barrier()
        if int(stop_flag.item()) == 1:
            watchdog_paused = True
            break

    if is_ddp:
        dist.destroy_process_group()
    if watchdog_paused:
        sys.exit(42)
    if is_rank0 and not args.dry_run:
        print(f"\nð HOÃN Táº¤T HUáº¤N LUYá»N PHA 3 TRONG {(time.time() - t_total_start)/60.0:.1f} PHÃT!", flush=True)


if __name__ == '__main__':
    main()
')

bench_script_path = os.path.join(WORKING_DIR, "run_benchmark_suite.py")
with open(bench_script_path, "w", encoding="utf-8") as f:
    f.write('#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark_suite.py
--------------------------------------------------------------------------------
Bá» CÃNG Cá»¤ BENCHMARK Äá»I CHá»¨NG KHÃCH QUAN KÃP (ZERO-BIAS DUAL-BENCHMARK SUITE)
Kiá»m chá»©ng ÄÃ³ng gÃ³p thá»±c táº¿ cá»§a Dataset Custom Äá»i vá»i TPS-Inbetween (M1 gá»c vs M3 finetune)
vá»i CAM Káº¾T 100% ZERO ARCHITECTURAL MODIFICATION.

Hai bÃ i kiá»m Äá»nh chuáº©n hÃ³a:
  [Benchmark #1] Táº­p Chuá»i Äá»c Láº­p `sakuga_golden_benchmark_sequences_v2`
                 (198 sequences | 149 scenes khÃ´ng trÃ¹ng láº·p | 412 frames N=3,5,7)
    - So sÃ¡nh: M0 (Coarse TPS Warp) | M1 (Original model_latest.pt) | M3_best | M3_last
    - Dual-Regime Metrics:
        + Regime A (Raw Continuous): PSNR, SSIM, IE-MAE, PWCR, RGD, Ghost Pixels
        + Regime B (Thresholded tau=0.15 & DoG): WCD, WCD_DoG, S-EMD,
          Skeleton Recall, Hair-Tip Recall, Stroke Recall/Precision/F1, Density Ratio
    - PhÃ¢n táº§ng theo N={3, 5, 7} & BiÃªn Äá» chuyá»n Äá»ng (mean_disp <6, 6..14, >=14 px)
    - Thá»ng kÃª chá»ng thiÃªn vá»: Head-to-Head Win Rate (%) & Wilcoxon Signed-Rank Test
      gom nhÃ³m theo 149 scenes Äá»c láº­p (trÃ¡nh pseudoreplication).
    - Tá»± Äá»ng xuáº¥t Top-3 Máº«u Cáº£i Thiá»n Máº¡nh Nháº¥t & Top-3 Failure Cases (5 cá»t HUD).

  [Benchmark #2] Táº­p Thá»±c Táº¿ LiÃªn Tá»¥c `data/lineart_experiments/test`
                 (24 frames 540p | 7 Keyframes -> 17 Inbetween frames | delta_f = 3,4,5)
    - DÃ¹ng ÄÃºng Äiá»m neo GlueStick chÃ­nh thá»ng (hoáº·c precomputed cache .npz) + pad 540->544.
    - Há» trá»£ Äá»i chiáº¿u SIFT Reference tÆ°Æ¡ng thÃ­ch báº£ng lá»ch sá»­ evolution_timeline_master.csv.
    - Xuáº¥t báº£ng CSV tá»«ng frame, tá»«ng cháº·ng delta_f, áº£nh Side-by-Side 4 cá»t,
      Error Heatmap vÃ  Video Äá»i chiáº¿u MP4/GIF 8 FPS.
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import json
import glob
import math
import types
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    cv2 = None

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

try:
    import imageio
except ImportError:
    imageio = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir):
    cur = start_dir
    for _ in range(7):
        if os.path.isdir(os.path.join(cur, "checkpoints")) and os.path.isdir(os.path.join(cur, "models")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(start_dir, "..", "..", "..", ".."))


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
DEFAULT_ORIG_CKPT = os.path.join(REPO_ROOT, "checkpoints", "pretrained_baselines", "tps-original", "model_latest.pt")
DEFAULT_LINEART_TEST = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
DEFAULT_GLUESTICK_CACHE = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_final", "lineart_test_gluestick_matches.npz")

# TÃ¡i sá»­ dá»¥ng trá»±c tiáº¿p toÃ¡n tá»­ tá»« ddp_worker náº¿u náº±m trong cÃ¹ng repo Äá» trÃ¡nh trÃ¹ng láº·p,
# kÃ¨m fallback Äá»c láº­p khi script cháº¡y tÃ¡ch rá»i trÃªn Kaggle/server khÃ¡c.
_WORKER_DIR = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_final")
_PIPELINES_DIR = os.path.join(REPO_ROOT, "pipelines")
for _d in (_PIPELINES_DIR, _WORKER_DIR):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    from multi_layer_tps import solve_multi_layer_tps, segment_motion_keypoints
except ImportError:
    solve_multi_layer_tps = None
    segment_motion_keypoints = None

try:
    from ddp_worker_tps_sequence_final import (
        TPSInbetweenOriginal,
        load_tps_weights,
        solve_tps_backward_pair,
        solve_tps_direct,
        tps_flow_from_theta_batch,
        evaluate_metrics_dual_regime,
    )
    _IMPORTED_CORE = True
except ImportError:
    _IMPORTED_CORE = False


if not _IMPORTED_CORE:
    FIXED_TPS_ANCHORS = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5],             [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
    ], dtype=np.float32)
    _GRID_CACHE, _PIXEL_GRID_CACHE, _DOG_KERNEL_CACHE = {}, {}, {}

    def warp_normalized_flow(img, flow):
        _, _, h, w = img.shape
        key = (h, w, flow.device)
        base_grid = _GRID_CACHE.get(key)
        if base_grid is None:
            gy, gx = torch.meshgrid(
                torch.linspace(0.0, 1.0, h, device=flow.device),
                torch.linspace(0.0, 1.0, w, device=flow.device), indexing='ij'
            )
            base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
            _GRID_CACHE[key] = base_grid
        grid = (base_grid + flow.float().permute(0, 2, 3, 1)) * 2.0 - 1.0
        return F.grid_sample(img.float().contiguous(), grid, mode='bilinear', padding_mode='border', align_corners=True).to(img.dtype)

    def warp_pixel_flow(img, flow):
        _, _, h, w = img.shape
        key = (h, w, flow.device)
        base_grid = _PIXEL_GRID_CACHE.get(key)
        if base_grid is None:
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=flow.device),
                torch.linspace(-1.0, 1.0, w, device=flow.device), indexing='ij'
            )
            base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
            _PIXEL_GRID_CACHE[key] = base_grid
        flow_f = flow.float()
        fn = torch.cat([flow_f[:, 0:1] / max(1.0, (w - 1.0) / 2.0), flow_f[:, 1:2] / max(1.0, (h - 1.0) / 2.0)], dim=1)
        grid = (base_grid + fn.permute(0, 2, 3, 1)).contiguous()
        return F.grid_sample(img.float().contiguous(), grid, mode='bilinear', padding_mode='border', align_corners=True).to(img.dtype)

    def solve_tps_direct(c_src, c_dst, max_pts=128, lambd=0.01, use_anchors=False):
        if c_src is None or c_dst is None or len(c_src) < 4:
            return np.zeros((max_pts, 2), np.float32), np.zeros((max_pts + 3, 2), np.float32), 0.0
        n_keep = (max_pts - 8) if use_anchors else max_pts
        if len(c_src) > n_keep:
            idx = np.linspace(0, len(c_src) - 1, n_keep, dtype=int)
            cs, cd = np.asarray(c_src[idx], np.float64), np.asarray(c_dst[idx], np.float64)
        else:
            cs, cd = np.asarray(c_src, np.float64), np.asarray(c_dst, np.float64)
        if use_anchors:
            anc = FIXED_TPS_ANCHORS.astype(np.float64)
            cs_all, cd_all = np.vstack([cs, anc]), np.vstack([cd, anc])
        else:
            cs_all, cd_all = cs, cd
        m = len(cs_all)
        r = np.linalg.norm(cs_all[:, None, :] - cs_all[None, :, :], axis=-1)
        K = (r ** 2) * np.log(np.maximum(r, 1e-6))
        np.fill_diagonal(K, float(lambd))
        P = np.hstack([np.ones((m, 1), np.float64), cs_all])
        A = np.zeros((m + 3, m + 3), np.float64)
        A[:m, :m], A[:m, m:], A[m:, :m] = K, P, P.T
        Y = np.zeros((m + 3, 2), np.float64)
        Y[:m, :] = cd_all
        try:
            th = np.linalg.solve(A, Y)
        except np.linalg.LinAlgError:
            th = np.linalg.lstsq(A, Y, rcond=1e-4)[0]
        valid = 0.0 if (np.isnan(th).any() or np.isinf(th).any() or np.max(np.abs(th)) > 50.0) else 1.0
        cp, tp = np.zeros((max_pts, 2), np.float32), np.zeros((max_pts + 3, 2), np.float32)
        if valid > 0.5:
            cp[:m], tp[:m], tp[max_pts:] = cs_all.astype(np.float32), th[:m].astype(np.float32), th[m:].astype(np.float32)
        return cp, tp, valid

    def solve_tps_backward_pair(kps0, kps1, max_pts=128, lambd=0.01):
        c0, th01, v01 = solve_tps_direct(kps1, kps0, max_pts=max_pts, lambd=lambd, use_anchors=True)
        c1, th10, v10 = solve_tps_direct(kps0, kps1, max_pts=max_pts, lambd=lambd, use_anchors=True)
        return c0, th01, c1, th10, (1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0)

    @torch.no_grad()
    def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, orig_hw, h_pad, w_pad, device, downscale_factor=2):
        th_f, cs_f, v_f = theta_batch.float(), c_src_batch.float(), valid_mask.float()
        b = th_f.shape[0]
        h_sub, w_sub = max(16, h_pad // max(1, downscale_factor)), max(16, w_pad // max(1, downscale_factor))
        gy, gx = torch.meshgrid(torch.linspace(0, 1, h_sub, device=device), torch.linspace(0, 1, w_sub, device=device), indexing='ij')
        pts_pad = torch.stack([gx, gy], dim=-1).view(1, -1, 2).expand(b, -1, -1)
        hw_f = orig_hw.to(device=device, dtype=torch.float32)
        sx = (float(max(1, w_pad - 1)) / torch.clamp(hw_f[:, 1] - 1.0, min=1.0)).view(b, 1, 1)
        sy = (float(max(1, h_pad - 1)) / torch.clamp(hw_f[:, 0] - 1.0, min=1.0)).view(b, 1, 1)
        scale_xy = torch.cat([sx, sy], dim=-1)
        pts_raw = pts_pad * scale_xy
        in_b = ((pts_raw[..., 0:1] <= 1.001) & (pts_raw[..., 1:2] <= 1.001)).float()
        pts_q = torch.clamp(pts_raw, 0.0, 1.0)
        chunks = []
        for i in range(0, pts_q.shape[1], 32768):
            q = pts_q[:, i:i + 32768]
            r = torch.clamp(torch.cdist(q, cs_f, p=2.0), min=1e-6)
            U = (r ** 2) * torch.log(r)
            basis = torch.cat([U, torch.ones((b, q.shape[1], 1), device=device), q], dim=-1)
            chunks.append(((torch.bmm(basis, th_f) - q) / scale_xy) * in_b[:, i:i + 32768])
        flow = torch.cat(chunks, dim=1).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()
        if h_sub != h_pad or w_sub != w_pad:
            flow = F.interpolate(flow, size=(h_pad, w_pad), mode='bilinear', align_corners=True)
        return flow * v_f.view(b, 1, 1, 1)

    def _conv(i, o, k=3, s=1, p=1):
        return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=True), nn.PReLU(o))

    def _deconv(i, o):
        return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True), nn.PReLU(o))

    class _Conv2(nn.Module):
        def __init__(self, i, o, s=2):
            super().__init__()
            self.conv1, self.conv2 = _conv(i, o, 3, s, 1), _conv(o, o, 3, 1, 1)
        def forward(self, x):
            return self.conv2(self.conv1(x))

    class _Contextnet(nn.Module):
        def __init__(self, indim=1, c=16):
            super().__init__()
            self.conv1, self.conv2 = _Conv2(indim, c), _Conv2(c, 2 * c)
            self.conv3, self.conv4 = _Conv2(2 * c, 4 * c), _Conv2(4 * c, 8 * c)
        def forward(self, x, flow):
            feats = []
            for cv_blk in (self.conv1, self.conv2, self.conv3, self.conv4):
                x = cv_blk(x)
                flow = F.interpolate(flow, size=x.shape[2:], mode='bilinear', align_corners=False) * 0.5
                feats.append(warp_normalized_flow(x, flow))
            return feats

    class _IFBlock(nn.Module):
        def __init__(self, in_planes, c=64):
            super().__init__()
            self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 3, 2, 1), _conv(c // 2, c, 3, 2, 1))
            self.convblock = nn.Sequential(*[_conv(c, c) for _ in range(8)])
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        def forward(self, x, flow, scale):
            oh, ow = x.shape[2], x.shape[3]
            if scale != 1:
                x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
            if flow is not None:
                flow = F.interpolate(flow, size=x.shape[2:], mode='bilinear', align_corners=False) * (1.0 / scale)
                x = torch.cat((x, flow), 1)
            x = self.conv0(x.contiguous())
            x = self.convblock(x) + x
            tmp = F.interpolate(self.lastconv(x.float().contiguous()), size=(oh, ow), mode='bilinear', align_corners=False)
            return tmp[:, :4] * scale * 2, tmp[:, 4:5]

    class _IFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block0, self.block1, self.block2 = _IFBlock(6, 240), _IFBlock(17, 150), _IFBlock(17, 90)
        def forward(self, i0, i1, scale=(4, 2, 1)):
            r0, r1 = i0.repeat(1, 3, 1, 1), i1.repeat(1, 3, 1, 1)
            w0, w1, flow, mask = i0, i1, None, None
            for i, blk in enumerate((self.block0, self.block1, self.block2)):
                if flow is not None:
                    fd, md = blk(torch.cat((r0, r1, w0.repeat(1, 3, 1, 1), w1.repeat(1, 3, 1, 1), mask), 1), flow, scale[i])
                    flow, mask = flow + fd, mask + md
                else:
                    flow, mask = blk(torch.cat((r0, r1), 1), None, scale[i])
                if i < 2:
                    w0, w1 = warp_pixel_flow(i0, flow[:, :2]), warp_pixel_flow(i1, flow[:, 2:4])
            return flow, torch.sigmoid(mask)

    class _OriginalUnet(nn.Module):
        def __init__(self, indim=9, outdim=1, c=16):
            super().__init__()
            self.down0, self.down1 = _Conv2(indim, 2 * c), _Conv2(4 * c, 4 * c)
            self.down2, self.down3 = _Conv2(8 * c, 8 * c), _Conv2(16 * c, 16 * c)
            self.up0, self.up1 = _deconv(32 * c, 8 * c), _deconv(16 * c, 4 * c)
            self.up2, self.up3 = _deconv(8 * c, 2 * c), _deconv(4 * c, c)
            self.conv = nn.Conv2d(c, outdim, 3, 1, 1)
        def forward(self, img0, img1, flow0, flow1, w0, w1, mask, c0, c1):
            s0 = self.down0(torch.cat((img0, img1, w0, w1, flow0, flow1, mask), 1))
            s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
            s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
            s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
            x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
            x = self.up1(torch.cat((x, s2), 1))
            x = self.up2(torch.cat((x, s1), 1))
            x = self.up3(torch.cat((x, s0), 1))
            return torch.sigmoid(self.conv(x))

    class TPSInbetweenOriginal(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Contextnet(1, 16)
            self.IFNet = _IFNet()
            self.unet = _OriginalUnet(9, 1, 16)
        def forward(self, x0, x1, t_val, cf01, cf10, direct_t_flows=False):
            t = t_val
            if direct_t_flows:
                cf0t, cf1t = cf01, cf10
            else:
                cf0t = cf01 * (t ** 2) - (1.0 - t) * t * cf10
                cf1t = cf10 * ((1.0 - t) ** 2) - t * (1.0 - t) * cf01
            ci0, ci1 = warp_normalized_flow(x0, cf0t), warp_normalized_flow(x1, cf1t)
            coarse_w = torch.clamp((1.0 - t) * ci0 + t * ci1, 0.0, 1.0)
            f0, f1 = self.encoder(x0, cf0t), self.encoder(x1, cf1t)
            fbc, m = self.IFNet(ci0, ci1)
            fi0, fi1 = warp_pixel_flow(ci0, fbc[:, :2]), warp_pixel_flow(ci1, fbc[:, 2:4])
            res = self.unet(x0, x1, cf0t + fbc[:, :2], cf1t + fbc[:, 2:4], fi0, fi1, m, f0, f1)[:, 0:1] * 2.0 - 1.0
            It = torch.clamp((1.0 - t) * fi0 + t * fi1 + res, 0.0, 1.0)
            return It, coarse_w, fi0, fi1, fbc, m

    def load_tps_weights(model, ckpt_path, device='cpu', prefer_ema=False, is_rank0=True):
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(raw, dict):
            if prefer_ema and raw.get('ema_state') is not None:
                eo = raw['ema_state']
                sd = eo['shadow'] if isinstance(eo, dict) and 'shadow' in eo else eo
            elif 'model_state' in raw:
                sd = raw['model_state']
            elif 'state_dict' in raw:
                sd = raw['state_dict']
            elif 'model' in raw:
                sd = raw['model']
            else:
                sd = raw
        else:
            sd = raw
        msd = model.state_dict()
        matched = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd.items()
                   if k.replace('module.', '').replace('base_model.', '') in msd and msd[k.replace('module.', '').replace('base_model.', '')].shape == v.shape}
        missing, _ = model.load_state_dict(matched, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys: {missing[:5]}")
        if is_rank0:
            print(f"[â] Náº¡p {os.path.basename(ckpt_path)}: {len(matched)}/{len(msd)} active tensors!")
        return model

    @torch.no_grad()
    def evaluate_metrics_dual_regime(pred_ink, gt_ink):
        def _dil(x): return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))
        def _ero(x): return torch.min(-F.max_pool2d(-x, (3, 1), (1, 1), (1, 0)), -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1)))
        def _skel(img):
            im = torch.clamp(img, 0.0, 1.0)
            s = F.relu(im - _dil(_ero(im)))
            im2 = _ero(im)
            return torch.clamp(s + F.relu(im2 - _dil(_ero(im2))), 0.0, 1.0)
        def _dt(b_m, steps=10):
            c = b_m; acc = c
            for _ in range(steps):
                c = F.max_pool2d(c, 3, 1, 1); acc = acc + c
            return torch.clamp(1.0 - acc / float(steps + 1), 0.0, 1.0)
        def _blur(im, sig):
            r = max(1, int(round(3 * sig))); x = torch.arange(-r, r + 1, device=im.device, dtype=torch.float32)
            g = torch.exp(-(x ** 2) / (2 * sig * sig)); g = g / g.sum()
            k = torch.outer(g, g).view(1, 1, 2 * r + 1, 2 * r + 1).to(im.dtype)
            return F.conv2d(F.pad(im, (r, r, r, r), mode='replicate'), k)
        def _wcd(pb, gb):
            dg, dp = _dt(gb), _dt(pb)
            d1 = (pb * F.softplus(dg * 10)).sum((-2, -1)) / pb.sum((-2, -1)).clamp(min=1.0)
            d2 = (gb * F.softplus(dp * 10)).sum((-2, -1)) / gb.sum((-2, -1)).clamp(min=1.0)
            ap, ag = pb.sum((-2, -1)), gb.sum((-2, -1))
            diff_rel = (ap - ag).abs() / torch.minimum(ap, ag).clamp(min=1.0)
            h = torch.sigmoid(diff_rel)
            h_leg = 0.5 * (1.0 + torch.minimum(ap, ag) / torch.maximum(ap, ag).clamp(min=1.0))
            return h * (d1 + d2), h_leg * (d1 + d2)
        p, g = pred_ink.float().clamp(0, 1), gt_ink.float().clamp(0, 1)
        mse = ((p - g) ** 2).mean((-3, -2, -1))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-8))
        mp, mg = F.avg_pool2d(p, 11, 1, 5), F.avg_pool2d(g, 11, 1, 5)
        sp, sg = F.avg_pool2d(p * p, 11, 1, 5) - mp ** 2, F.avg_pool2d(g * g, 11, 1, 5) - mg ** 2
        spg = F.avg_pool2d(p * g, 11, 1, 5) - mp * mg
        ssim = (((2 * mp * mg + 1e-4) * (2 * spg + 9e-4)) / ((mp ** 2 + mg ** 2 + 1e-4) * (sp + sg + 9e-4))).mean((-3, -2, -1))
        l1 = (p - g).abs().mean((-3, -2, -1))
        p255, g255 = (1.0 - p) * 255.0, (1.0 - g) * 255.0
        pwcr = (p255 >= 250.0).float().mean((-3, -2, -1))
        rgd = ((p255 >= 180.0) & (p255 < 250.0)).float().mean((-3, -2, -1))
        ghost = (((p255 >= 180.0) & (p255 <= 250.0)).float() * (g255 > 250.0).float()).sum((-3, -2, -1))
        dark_g = ((p > 0.35).float() * (1.0 - _dil(_dil((g > 0.08).float())))).sum((-3, -2, -1))
        pb, gb = (p > 0.15).float(), (g > 0.15).float()
        pd_m = ((_blur(1 - p, 1.0) - _blur(1 - p, 1.6)) < -0.02).float()
        gd_m = ((_blur(1 - g, 1.0) - _blur(1 - g, 1.6)) < -0.02).float()
        wcd_val, wcd_leg = _wcd(pb, gb)
        wcd_dog_val, wcd_dog_leg = _wcd(pd_m, gd_m)
        pn = p / p.sum((-2, -1), keepdim=True).clamp(min=1e-6); gn = g / g.sum((-2, -1), keepdim=True).clamp(min=1e-6)
        semd = 0.5 * ((pn.sum(-1).cumsum(-1) - gn.sum(-1).cumsum(-1)).abs().mean((-2, -1)) + (pn.sum(-2).cumsum(-1) - gn.sum(-2).cumsum(-1)).abs().mean((-2, -1)))
        gdil, pdil = _dil(_dil(gb)), _dil(_dil(pb))
        rec = (gb * pdil).sum((-3, -2, -1)) / gb.sum((-3, -2, -1)).clamp(min=1.0)
        prec = (pb * gdil).sum((-3, -2, -1)) / pb.sum((-3, -2, -1)).clamp(min=1.0)
        f1 = 2 * rec * prec / (rec + prec).clamp(min=1e-6)
        sk_g = _skel(g); sk_m = (sk_g > 0.20).float(); pd1 = _dil(pb)
        sk_r = (pd1 * sk_m).sum((-3, -2, -1)) / sk_m.sum((-3, -2, -1)).clamp(min=1.0)
        nbr = F.avg_pool2d(sk_g, 3, 1, 1) * 9.0 - sk_g
        ep = ((nbr <= 1.8) & (sk_g > 0.20)).float()
        tip_r = (pd1 * ep).sum((-3, -2, -1)) / ep.sum((-3, -2, -1)).clamp(min=1.0)
        dens = pb.sum((-3, -2, -1)) / gb.sum((-3, -2, -1)).clamp(min=1.0)
        return {k: v.squeeze(-1) if v.ndim > 1 else v for k, v in {
            'psnr': psnr, 'ssim': ssim, 'val_l1': l1, 'ie_mae': l1 * 255.0,
            'pwcr': pwcr, 'rgd': rgd, 'ghost': ghost, 'dark_ghost': dark_g,
            'wcd': wcd_val, 'wcd_legacy': wcd_leg, 'wcd_dog': wcd_dog_val, 'wcd_dog_legacy': wcd_dog_leg, 'semd': semd,
            'recall': rec, 'precision': prec, 'f1': f1, 'density_ratio': dens,
            'skel_recall': sk_r, 'tip_recall': tip_r,
        }.items()}


# ==============================================================================
# CÃC HÃM Äá»C áº¢NH, TRÃCH ÄIá»M NEO GLUESTICK/SIFT & Váº¼ BÃO CÃO TRá»°C QUAN
# ==============================================================================
def load_ink_image(path):
    with Image.open(path) as im:
        gray = np.asarray(im.convert('L'), dtype=np.float32)
    return (255.0 - gray) / 255.0


def ink_to_u8_gray(ink_t):
    arr = (1.0 - ink_t.squeeze().detach().cpu().numpy().clip(0.0, 1.0)) * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def extract_sift_matches(im0_u8, im1_u8, max_pts=120):
    if cv2 is None:
        return np.zeros((2, 0, 2), dtype=np.float32)
    h, w = im0_u8.shape
    sift = cv2.SIFT_create(nfeatures=1000)
    k0, d0 = sift.detectAndCompute(im0_u8, None)
    k1, d1 = sift.detectAndCompute(im1_u8, None)
    if d0 is None or d1 is None or len(k0) < 4 or len(k1) < 4:
        return np.zeros((2, 0, 2), dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(d0, d1)
    if len(matches) < 4:
        return np.zeros((2, 0, 2), dtype=np.float32)
    p0 = np.float32([k0[m.queryIdx].pt for m in matches])
    p1 = np.float32([k1[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 15.0)
    if mask is not None:
        p0, p1 = p0[mask.ravel() == 1], p1[mask.ravel() == 1]
    p0[:, 0] /= float(w); p0[:, 1] /= float(h)
    p1[:, 0] /= float(w); p1[:, 1] /= float(h)
    if len(p0) > max_pts:
        idx = np.linspace(0, len(p0) - 1, max_pts, dtype=int)
        p0, p1 = p0[idx], p1[idx]
    return np.stack([p0, p1], axis=0).astype(np.float32)


def try_init_gluestick(device):
    tps_repo = os.path.join(REPO_ROOT, "models", "tps-inbetween", "tps-inbetween")
    weights_p = os.path.join(tps_repo, "model", "resources", "weights", "checkpoint_GlueStick_MD.tar")
    if not os.path.exists(weights_p):
        return None
    if cv2 is not None and 'pytlsd' not in sys.modules:
        def _cv2_lsd(img, scale=1.0):
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.dtype != np.uint8:
                img = (img * 255.0).clip(0, 255).astype(np.uint8) if img.max() <= 1.01 and img.max() > 0 else np.clip(img, 0, 255).astype(np.uint8)
            img_s = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else img
            det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lines, _, _, _ = det.detect(img_s)
            if lines is None or len(lines) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            lines = lines.reshape(-1, 4).astype(np.float32)
            if scale != 1.0:
                lines /= float(scale)
            return np.hstack([lines, np.ones((len(lines), 1), dtype=np.float32)])
        fake = types.ModuleType('pytlsd')
        fake.lsd = _cv2_lsd
        sys.modules['pytlsd'] = fake
    if tps_repo not in sys.path:
        sys.path.insert(0, tps_repo)
    try:
        from model.gluestick.models.two_view_pipeline import TwoViewPipeline
        conf = {
            'name': 'two_view_pipeline', 'use_lines': True,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
                'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
                'max_n_lines': 300,
            },
            'matcher': {'name': 'gluestick', 'weights': weights_p, 'trainable': False},
            'ground_truth': {'from_pose_depth': False},
        }
        return TwoViewPipeline(conf).to(device).eval()
    except Exception:
        return None


def compute_wilcoxon_p(a_vals, b_vals):
    if wilcoxon is None or len(a_vals) < 10:
        return float('nan')
    diff = np.asarray(a_vals, dtype=np.float64) - np.asarray(b_vals, dtype=np.float64)
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        _, p = wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
        return float(p)
    except Exception:
        return float('nan')


def render_audit_row_panel(im0_u8, im1_u8, m0_u8, m1_u8, m3_u8, gt_u8, title_str, stats_m1, stats_m3):
    """Váº½ dáº£i 5 cá»t chuáº©n: [Overlay I0/I1 | M0 Coarse | M1 Orig | M3 Finetune | Ground Truth]."""
    h, w = gt_u8.shape
    overlay = (0.5 * im0_u8.astype(np.float32) + 0.5 * im1_u8.astype(np.float32)).astype(np.uint8)
    cols = [overlay, m0_u8, m1_u8, m3_u8, gt_u8]
    titles = [
        ("Overlay I0/I1", (210, 210, 210)),
        ("M0 Coarse TPS", (180, 180, 180)),
        (f"M1 Orig ({stats_m1})", (100, 150, 255)),
        (f"M3 Fine ({stats_m3})", (90, 240, 120)),
        ("Ground Truth", (245, 245, 245)),
    ]
    head_h, foot_h = 34, 28
    canvas = np.full((head_h + h + foot_h, w * 5, 3), 26, dtype=np.uint8)
    for i, (arr, (lbl, clr)) in enumerate(zip(cols, titles)):
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.repeat(arr[:, :, None], 3, axis=-1)
        canvas[head_h:head_h + h, i * w:(i + 1) * w] = bgr
        if cv2 is not None:
            cv2.putText(canvas, lbl, (i * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)
            if i > 0:
                cv2.line(canvas, (i * w, 0), (i * w, head_h + h + foot_h), (70, 70, 70), 2)
    if cv2 is not None:
        cv2.putText(canvas, title_str, (12, head_h + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


# ==============================================================================
# BENCHMARK #1: Táº¬P CHUá»I CÃCH LY `sakuga_golden_benchmark_sequences_v2`
# ==============================================================================
def run_benchmark_isolated_sequences(bench_dir, models_dict, device, out_dir, downscale_factor=2, save_visuals=True, use_multilayer=True):
    csv_path = os.path.join(bench_dir, "benchmark_sequences.csv")
    clips_dir = os.path.join(bench_dir, "clips")
    matches_dir = os.path.join(bench_dir, "matches")
    df_bench = pd.read_csv(csv_path)
    use_ml = bool(use_multilayer and (solve_multi_layer_tps is not None))

    print("\n" + "=" * 86)
    print(f"ð¬ BENCHMARK #1: Táº¬P CHUá»I CÃCH LY ({len(df_bench)} sequences / {df_bench['scene_id'].nunique()} scenes) | MultiLayerTPS={use_ml}")
    print(f"   CÃ¡c mÃ´ hÃ¬nh Äá»i chá»©ng: {['M0_Coarse'] + list(models_dict.keys())}")
    print("=" * 86, flush=True)

    per_sample_rows = []
    # LÆ°u thÃ´ng tin nháº¹ Äá» sau chá»n Top-3 cáº£i thiá»n & Top-3 tá» nháº¥t (chá» lÆ°u path + metric)
    sample_index = []

    t_start = time.time()
    for row_idx, row in df_bench.iterrows():
        cid = str(row['clip_id'])
        sid = str(row['scene_id'])
        seq_len = int(row['seq_len'])
        disp = float(row.get('mean_disp', 0.0))

        im0 = load_ink_image(os.path.join(clips_dir, cid, "000001.png"))
        im1 = load_ink_image(os.path.join(clips_dir, cid, f"{seq_len:06d}.png"))
        h, w = im0.shape
        h_pad = ((h + 31) // 32) * 32
        w_pad = ((w + 31) // 32) * 32

        x0 = F.pad(torch.from_numpy(im0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1 = F.pad(torch.from_numpy(im1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        orig_hw = torch.tensor([[h, w]], dtype=torch.int32, device=device)

        # Äá»c matches chuáº©n Backward-Warp
        m_cands = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
        if m_cands:
            kps = np.load(m_cands[0]).astype(np.float32)
            if use_ml:
                cf01, cf10, _ = solve_multi_layer_tps(
                    kps[0], kps[1], (h, w), h_pad, w_pad, device,
                    direct_t_flows=False, max_pts=128, lambd=0.01,
                    downscale_factor=downscale_factor, eps_static_px=1.5,
                    use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
                )
            else:
                c0_np, th01_np, c1_np, th10_np, v_flag = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
                c0_t = torch.from_numpy(c0_np)[None].to(device)
                th01_t = torch.from_numpy(th01_np)[None].to(device)
                c1_t = torch.from_numpy(c1_np)[None].to(device)
                th10_t = torch.from_numpy(th10_np)[None].to(device)
                v_t = torch.tensor([v_flag], dtype=torch.float32, device=device)
                cf01 = tps_flow_from_theta_batch(th01_t, c0_t, v_t, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)
                cf10 = tps_flow_from_theta_batch(th10_t, c1_t, v_t, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)
        else:
            cf01 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)
            cf10 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)

        for k in range(1, seq_len - 1):
            fk = k + 1
            t_val = float(k) / float(seq_len - 1)
            t_tensor = torch.tensor([[[[t_val]]]], dtype=torch.float32, device=device)
            gt_ink = torch.from_numpy(load_ink_image(os.path.join(clips_dir, cid, f"{fk:06d}.png")))[None, None].to(device)

            sample_rec = {
                'clip_id': cid, 'scene_id': sid, 'seq_len': seq_len, 'frame_k': fk,
                't_val': round(t_val, 4), 'mean_disp': disp, 'orig_w': w, 'orig_h': h
            }
            coarse_ref = None
            for m_name, mdl in models_dict.items():
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                ts = time.time()
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    pred_p, cw_p, *_ = mdl(x0.float(), x1.float(), t_tensor.float(), cf01.float(), cf10.float())
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                lat_ms = (time.time() - ts) * 1000.0

                pred_crop = pred_p[:, :, :h, :w]
                if coarse_ref is None:
                    coarse_ref = cw_p[:, :, :h, :w]
                    mm0 = evaluate_metrics_dual_regime(coarse_ref, gt_ink)
                    for mk, mv in mm0.items():
                        sample_rec[f"M0_{mk}"] = float(mv.item())

                mm = evaluate_metrics_dual_regime(pred_crop, gt_ink)
                for mk, mv in mm.items():
                    sample_rec[f"{m_name}_{mk}"] = float(mv.item())
                sample_rec[f"{m_name}_latency_ms"] = float(lat_ms)

            per_sample_rows.append(sample_rec)
            sample_index.append(sample_rec)

        if (row_idx + 1) % 50 == 0 or (row_idx + 1) == len(df_bench):
            print(f"   ... HoÃ n thÃ nh {row_idx + 1}/{len(df_bench)} sequences ({time.time() - t_start:.1f}s)", flush=True)

    df_res = pd.DataFrame(per_sample_rows)
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, "benchmark_per_sample_comparison.csv"), index=False)

    # Tá»ng há»£p báº£ng so sÃ¡nh toÃ n cá»¥c cho táº¥t cáº£ model tags
    all_tags = ['M0'] + list(models_dict.keys())
    metric_cols = ['psnr', 'ssim', 'ie_mae', 'pwcr', 'rgd', 'ghost', 'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy',
                   'semd', 'skel_recall', 'tip_recall', 'recall', 'precision', 'f1', 'density_ratio']
    summary_rows = []
    for tag in all_tags:
        srow = {'model_tag': tag, 'total_frames': len(df_res), 'total_scenes': df_res['scene_id'].nunique()}
        for mc in metric_cols:
            col = f"{tag}_{mc}"
            if col in df_res.columns:
                srow[mc] = float(df_res[col].mean())
        lat_col = f"{tag}_latency_ms"
        srow['latency_ms'] = float(df_res[lat_col].mean()) if lat_col in df_res.columns else 0.0
        summary_rows.append(srow)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "benchmark_overall_summary.csv"), index=False)

    # PhÃ¢n táº§ng theo seq_len N={3,5,7} vÃ  Motion Tiers (<6, 6..14, >=14)
    strat_rows = []
    for sl in (3, 5, 7):
        sub = df_res[df_res['seq_len'] == sl]
        if len(sub) == 0:
            continue
        for tag in all_tags:
            r = {'group_type': 'seq_len', 'group_val': f"N={sl}", 'model_tag': tag, 'frames': len(sub)}
            for mc in ['psnr', 'ssim', 'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy', 'semd', 'skel_recall', 'tip_recall', 'f1', 'ghost', 'density_ratio']:
                r[mc] = float(sub[f"{tag}_{mc}"].mean())
            strat_rows.append(r)

    disp_bins = [('Small_<6px', df_res['mean_disp'] < 6.0),
                 ('Med_6_14px', (df_res['mean_disp'] >= 6.0) & (df_res['mean_disp'] < 14.0)),
                 ('Large_>=14px', df_res['mean_disp'] >= 14.0)]
    for dlabel, mask in disp_bins:
        sub = df_res[mask]
        if len(sub) == 0:
            continue
        for tag in all_tags:
            r = {'group_type': 'motion_disp', 'group_val': dlabel, 'model_tag': tag, 'frames': len(sub)}
            for mc in ['psnr', 'ssim', 'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy', 'semd', 'skel_recall', 'tip_recall', 'f1', 'ghost']:
                r[mc] = float(sub[f"{tag}_{mc}"].mean())
            strat_rows.append(r)
    pd.DataFrame(strat_rows).to_csv(os.path.join(out_dir, "benchmark_stratified_breakdown.csv"), index=False)

    # Kiá»m Äá»nh thá»ng kÃª khÃ´ng thiÃªn vá» M3_best (hoáº·c model thá»© 2) vs M1_Original
    stat_audit = {'overall': summary_rows}
    if 'M1_Original' in models_dict:
        compare_tag = 'M3_Best' if 'M3_Best' in models_dict else next((k for k in models_dict if k != 'M1_Original'), None)
        if compare_tag is not None:
            # Gom nhÃ³m trung bÃ¬nh theo tá»«ng scene Äá»c láº­p (149 scenes) trÆ°á»c Wilcoxon test!
            scene_df = df_res.groupby('scene_id').mean(numeric_only=True)
            h2h = {}
            for mc, better_higher in [('wcd', False), ('wcd_dog', False), ('semd', False),
                                      ('skel_recall', True), ('tip_recall', True), ('f1', True), ('psnr', True), ('ssim', True)]:
                c_m1, c_m3 = f"M1_Original_{mc}", f"{compare_tag}_{mc}"
                if better_higher:
                    win_rate = float(np.mean(df_res[c_m3].values > df_res[c_m1].values))
                else:
                    win_rate = float(np.mean(df_res[c_m3].values < df_res[c_m1].values))
                pval = compute_wilcoxon_p(scene_df[c_m3].values, scene_df[c_m1].values)
                h2h[mc] = {
                    'm1_mean': float(df_res[c_m1].mean()),
                    'm3_mean': float(df_res[c_m3].mean()),
                    'delta': float(df_res[c_m3].mean() - df_res[c_m1].mean()),
                    'frame_win_rate_pct': round(win_rate * 100.0, 2),
                    'scene_wilcoxon_p': pval,
                    'significant_p001': (not math.isnan(pval)) and (pval < 0.01),
                }
            # Káº¿t luáº­n H0 vs H1 tá»± Äá»ng dá»±a trÃªn 3 trá»¥ cá»t: HÃ¬nh há»c + TÃ´-pÃ´ + KhÃ´ng dÃ y nÃ©t (Anti-Thickening)
            geom_pass = (h2h['wcd_dog']['delta'] < 0 and h2h['wcd_dog']['significant_p001']) or (h2h['wcd']['delta'] < 0 and h2h['wcd']['significant_p001'])
            topo_pass = h2h['skel_recall']['delta'] > 0 and h2h['skel_recall']['significant_p001']
            dens_m3 = float(df_res[f"{compare_tag}_density_ratio"].mean())
            anti_thickening_pass = (0.95 <= dens_m3 <= 1.08)
            stat_audit['head_to_head_M3_vs_M1'] = h2h
            if geom_pass and topo_pass and anti_thickening_pass:
                stat_audit['hypothesis_verdict'] = "H1_SUPPORTED (Cáº£i thiá»n Äá»ng thá»i cáº£ HÃ¬nh há»c + TÃ´-pÃ´ nÃ©t váº½ vá»i p < 0.01 & Giá»¯ máº­t Äá» nÃ©t chuáº©n [0.95, 1.08])"
            elif geom_pass and topo_pass and not anti_thickening_pass:
                stat_audit['hypothesis_verdict'] = f"PARTIAL_WITH_THICKENING (WCD/Skel cáº£i thiá»n nhÆ°ng máº­t Äá» nÃ©t Äáº¡t {dens_m3:.3f} vÆ°á»£t ngÆ°á»¡ng kiá»m soÃ¡t)"
            else:
                stat_audit['hypothesis_verdict'] = "MIXED_OR_H0 (Cáº§n xem xÃ©t ká»¹ phÃ¢n tÃ¡ch Raw vs DoG vÃ  chá» sá» Skel/WCD)"

            # Xuáº¥t Top-3 Cáº£i Thiá»n & Top-3 Failure Cases theo delta WCD_DoG
            if save_visuals and len(df_res) >= 6:
                df_res['delta_wcd_dog'] = df_res[f"{compare_tag}_wcd_dog"] - df_res["M1_Original_wcd_dog"]
                best3 = df_res.sort_values('delta_wcd_dog', ascending=True).head(3)
                worst3 = df_res.sort_values('delta_wcd_dog', ascending=False).head(3)
                vis_dir = os.path.join(out_dir, "audit_cases_top3")
                os.makedirs(vis_dir, exist_ok=True)
                for tag_grp, sub_df in [("TOP3_IMPROVED", best3), ("TOP3_FAILURE_OR_HARD", worst3)]:
                    panels = []
                    for _, sr in sub_df.iterrows():
                        cid, fk, sl = sr['clip_id'], int(sr['frame_k']), int(sr['seq_len'])
                        im0_l = load_ink_image(os.path.join(clips_dir, cid, "000001.png"))
                        im1_l = load_ink_image(os.path.join(clips_dir, cid, f"{sl:06d}.png"))
                        gt_l = load_ink_image(os.path.join(clips_dir, cid, f"{fk:06d}.png"))
                        hh, ww = im0_l.shape
                        hp, wp = ((hh + 31) // 32) * 32, ((ww + 31) // 32) * 32
                        mc = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
                        if mc:
                            kps = np.load(mc[0]).astype(np.float32)
                            c0n, t01n, c1n, t10n, vf = solve_tps_backward_pair(kps[0], kps[1], 128, 0.01)
                        else:
                            c0n = c1n = np.zeros((128, 2), np.float32); t01n = t10n = np.zeros((131, 2), np.float32); vf = 0.0
                        x0p = F.pad(torch.from_numpy(im0_l)[None, None], (0, wp - ww, 0, hp - hh)).to(device)
                        x1p = F.pad(torch.from_numpy(im1_l)[None, None], (0, wp - ww, 0, hp - hh)).to(device)
                        hw_t = torch.tensor([[hh, ww]], dtype=torch.int32, device=device)
                        cf01_ = tps_flow_from_theta_batch(torch.from_numpy(t01n)[None].to(device), torch.from_numpy(c0n)[None].to(device), torch.tensor([vf], device=device), hw_t, hp, wp, device, downscale_factor)
                        cf10_ = tps_flow_from_theta_batch(torch.from_numpy(t10n)[None].to(device), torch.from_numpy(c1n)[None].to(device), torch.tensor([vf], device=device), hw_t, hp, wp, device, downscale_factor)
                        tt = torch.tensor([[[[float(sr['t_val'])]]]], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            pr1, cw_, *_ = models_dict['M1_Original'](x0p, x1p, tt, cf01_, cf10_)
                            pr3, *_ = models_dict[compare_tag](x0p, x1p, tt, cf01_, cf10_)
                        panel = render_audit_row_panel(
                            ink_to_u8_gray(x0p[:, :, :hh, :ww]), ink_to_u8_gray(x1p[:, :, :hh, :ww]),
                            ink_to_u8_gray(cw_[:, :, :hh, :ww]), ink_to_u8_gray(pr1[:, :, :hh, :ww]),
                            ink_to_u8_gray(pr3[:, :, :hh, :ww]), (255.0 - gt_l * 255.0).clip(0, 255).astype(np.uint8),
                            f"[{tag_grp}] {cid} (N={sl}, k={fk}, t={sr['t_val']:.2f}, disp={sr['mean_disp']:.1f}px) | dWCD_DoG={sr['delta_wcd_dog']:+.4f}",
                            f"WCD={sr['M1_Original_wcd_dog']:.3f}|Sk={sr['M1_Original_skel_recall']*100:.0f}%",
                            f"WCD={sr[f'{compare_tag}_wcd_dog']:.3f}|Sk={sr[f'{compare_tag}_skel_recall']*100:.0f}%"
                        )
                        panels.append(panel)
                    # GhÃ©p 3 hÃ ng cÃ³ thá» hÆ¡i khÃ¡c chiá»u rá»ng -> Äá»m ngang vá» max_w
                    max_pw = max(p.shape[1] for p in panels)
                    norm_panels = [np.pad(p, ((0, 0), (0, max_pw - p.shape[1]), (0, 0)), constant_values=26) for p in panels]
                    stacked = np.vstack(norm_panels)
                    Image.fromarray(stacked[:, :, ::-1] if cv2 is not None else stacked).save(os.path.join(vis_dir, f"{tag_grp}.png"))

    with open(os.path.join(out_dir, "benchmark_statistical_audit.json"), 'w', encoding='utf-8') as f:
        json.dump(stat_audit, f, indent=2, ensure_ascii=False)

    print("\nð Káº¾T QUáº¢ BENCHMARK #1 (ISOLATED 198 CLIPS / 412 FRAMES):")
    print(df_summary.round(4).to_string(index=False))
    if 'head_to_head_M3_vs_M1' in stat_audit:
        print(f"\nâï¸ KIá»M Äá»NH Äá»I Äáº¦U THEO SCENE (149 SCENES) | Káº¿t luáº­n: {stat_audit['hypothesis_verdict']}")
        for mc, st in stat_audit['head_to_head_M3_vs_M1'].items():
            print(f"   - {mc:12s}: M1={st['m1_mean']:.4f} -> M3={st['m3_mean']:.4f} (Delta={st['delta']:+.4f} | WinRate={st['frame_win_rate_pct']:.1f}% | p={st['scene_wilcoxon_p']:.2e})")
    return df_summary, stat_audit


# ==============================================================================
# BENCHMARK #2: Táº O FRAME THá»°C Táº¾ LIÃN Tá»¤C (`data/lineart_experiments/test`)
# ==============================================================================
def run_benchmark_continuous_lineart_test(test_dir, models_dict, device, out_dir, gluestick_npz=None, downscale_factor=2, use_multilayer=True):
    spec_path = os.path.join(test_dir, "continuous_keyframes_spec.json")
    if not os.path.exists(spec_path):
        print(f"[*] KhÃ´ng tÃ¬m tháº¥y {spec_path}, bá» qua Benchmark #2.")
        return None

    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    frames_spec = spec['frames']
    key_indices = sorted(spec['key_indices'])
    use_ml = bool(use_multilayer and (solve_multi_layer_tps is not None))

    print("\n" + "=" * 86)
    print(f"ð¬ BENCHMARK #2: Táº O FRAME THá»°C Táº¾ LIÃN Tá»¤C ({len(frames_spec)} frames: {len(key_indices)} keyframes, {len(frames_spec)-len(key_indices)} inbetweens) | MultiLayerTPS={use_ml}")
    print("=" * 86, flush=True)

    # Náº¡p áº£nh GT 960x540
    gt_imgs_u8 = [np.asarray(Image.open(os.path.join(test_dir, item['filename'])).convert('L'), dtype=np.uint8) for item in frames_spec]
    h, w = gt_imgs_u8[0].shape
    h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32  # 540 -> 544, 960 -> 960

    # Chuáº©n bá» cáº·p Äiá»m neo GlueStick (Æ°u tiÃªn precomputed .npz -> cháº¡y GlueStick trá»±c tiáº¿p -> SIFT fallback)
    gs_cache = dict(np.load(gluestick_npz)) if (gluestick_npz and os.path.exists(gluestick_npz)) else None
    gs_model = None if gs_cache is not None else try_init_gluestick(device)
    match_source = "GlueStick_MD_Precomputed" if gs_cache is not None else ("GlueStick_MD_Live" if gs_model is not None else "SIFT_Fallback")
    print(f"[*] Bá» trÃ­ch xuáº¥t Äiá»m neo hÃ¬nh há»c: {match_source} | Pad: {w}x{h} -> {w_pad}x{h_pad}", flush=True)

    seg_data = {}
    for i in range(len(key_indices) - 1):
        ka, kb = key_indices[i], key_indices[i + 1]
        seg_key = f"seg_{ka}_{kb}"
        ima_u8, imb_u8 = gt_imgs_u8[ka], gt_imgs_u8[kb]
        if gs_cache is not None and seg_key in gs_cache:
            kps = gs_cache[seg_key].astype(np.float32)
        elif gs_model is not None:
            t0_g = torch.from_numpy(ima_u8.astype(np.float32) / 255.0)[None, None].to(device)
            t1_g = torch.from_numpy(imb_u8.astype(np.float32) / 255.0)[None, None].to(device)
            with torch.no_grad():
                pred_g = gs_model({'image0': t0_g, 'image1': t1_g})
            m0 = pred_g['matches0'][0].cpu().numpy()
            valid_g = m0 != -1
            kp0 = pred_g['keypoints0'][0].cpu().numpy()[valid_g].astype(np.float32)
            kp1 = pred_g['keypoints1'][0].cpu().numpy()[m0[valid_g]].astype(np.float32)
            kp0[:, 0] /= float(w); kp0[:, 1] /= float(h)
            kp1[:, 0] /= float(w); kp1[:, 1] /= float(h)
            kps = np.stack([kp0, kp1], axis=0)
        else:
            kps = extract_sift_matches(ima_u8, imb_u8, max_pts=120)

        x0_p = F.pad(torch.from_numpy((255.0 - ima_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1_p = F.pad(torch.from_numpy((255.0 - imb_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        hw_t = torch.tensor([[h, w]], dtype=torch.int32, device=device)
        if use_ml:
            cf01, cf10, _ = solve_multi_layer_tps(
                kps[0], kps[1], (h, w), h_pad, w_pad, device,
                direct_t_flows=False, max_pts=128, lambd=0.01,
                downscale_factor=downscale_factor, eps_static_px=1.5,
                use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
            )
        else:
            c0_np, t01_np, c1_np, t10_np, vf = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
            cf01 = tps_flow_from_theta_batch(torch.from_numpy(t01_np)[None].to(device), torch.from_numpy(c0_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, downscale_factor)
            cf10 = tps_flow_from_theta_batch(torch.from_numpy(t10_np)[None].to(device), torch.from_numpy(c1_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, downscale_factor)
        seg_data[(ka, kb)] = {'x0': x0_p, 'x1': x1_p, 'cf01': cf01, 'cf10': cf10, 'num_matches': kps.shape[1]}

    os.makedirs(out_dir, exist_ok=True)
    reconstructed = {tag: [img.copy() for img in gt_imgs_u8] for tag in ['M0'] + list(models_dict.keys())}
    per_frame_rows = []

    for item in frames_spec:
        f_idx = int(item['frame_idx'])
        if item['type'] != 'INBETWEEN':
            continue
        ka, kb, tv = int(item['ref_key_a']), int(item['ref_key_b']), float(item['t_val'])
        delta_f = kb - ka
        seg = seg_data[(ka, kb)]
        gt_ink = torch.from_numpy((255.0 - gt_imgs_u8[f_idx].astype(np.float32)) / 255.0)[None, None].to(device)
        tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

        row_base = {'frame_idx': f_idx, 'time_sec': item['time_sec'], 'filename': item['filename'],
                    'segment': f"{ka}->{kb}", 'delta_f': delta_f, 't_val': round(tv, 4), 'matches': seg['num_matches']}
        cw_done = False
        for m_name, mdl in models_dict.items():
            ts = time.time()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                pr_p, cw_p, *_ = mdl(seg['x0'].float(), seg['x1'].float(), tt.float(), seg['cf01'].float(), seg['cf10'].float())
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            lat = (time.time() - ts) * 1000.0
            pr_crop = pr_p[:, :, :h, :w]
            reconstructed[m_name][f_idx] = ink_to_u8_gray(pr_crop)
            mm = evaluate_metrics_dual_regime(pr_crop, gt_ink)

            if not cw_done:
                cw_crop = cw_p[:, :, :h, :w]
                reconstructed['M0'][f_idx] = ink_to_u8_gray(cw_crop)
                mm0 = evaluate_metrics_dual_regime(cw_crop, gt_ink)
                for mk, mv in mm0.items():
                    row_base[f"M0_{mk}"] = float(mv.item())
                cw_done = True

            for mk, mv in mm.items():
                row_base[f"{m_name}_{mk}"] = float(mv.item())
            row_base[f"{m_name}_latency_ms"] = float(lat)

        per_frame_rows.append(row_base)

    # TÃ­nh chá» sá» nháº¥t quÃ¡n thá»i gian (2C: Temporal Difference MAE & Background Flicker) trÃªn toÃ n chuá»i liÃªn tiáº¿p
    tags = ['M0'] + list(models_dict.keys())
    for row in per_frame_rows:
        fi = int(row['frame_idx'])
        g_cur = (255.0 - gt_imgs_u8[fi].astype(np.float32)) / 255.0
        g_prev = (255.0 - gt_imgs_u8[fi - 1].astype(np.float32)) / 255.0
        both_bg = (g_cur < 0.04) & (g_prev < 0.04)
        dg = g_cur - g_prev
        for t in tags:
            p_cur = (255.0 - reconstructed[t][fi].astype(np.float32)) / 255.0
            p_prev = (255.0 - reconstructed[t][fi - 1].astype(np.float32)) / 255.0
            dp = p_cur - p_prev
            row[f"{t}_temp_diff_mae"] = float(np.mean(np.abs(dp - dg)) * 255.0)
            row[f"{t}_temp_bg_flicker"] = float(np.sum(np.abs(dp) * both_bg))

    df_frames = pd.DataFrame(per_frame_rows)
    df_frames.to_csv(os.path.join(out_dir, "lineart_test_per_frame_metrics.csv"), index=False)

    # Tá»ng há»£p theo toÃ n chuá»i & theo tá»«ng delta_f (3, 4, 5)
    summary_list = []
    for grp_name, sub_df in [("ALL_17_INBETWEENS", df_frames)] + [(f"delta_f={dfv}", df_frames[df_frames['delta_f'] == dfv]) for dfv in sorted(df_frames['delta_f'].unique())]:
        for t in tags:
            sr = {'group': grp_name, 'model_tag': t, 'frames': len(sub_df)}
            for mc in ['psnr', 'ssim', 'ie_mae', 'wcd', 'wcd_dog', 'semd', 'skel_recall', 'tip_recall', 'f1',
                       'density_ratio', 'ghost', 'pwcr', 'rgd', 'temp_diff_mae', 'temp_bg_flicker']:
                sr[mc] = float(sub_df[f"{t}_{mc}"].mean())
            summary_list.append(sr)
    df_sum2 = pd.DataFrame(summary_list)
    df_sum2.to_csv(os.path.join(out_dir, "lineart_test_summary.csv"), index=False)

    print("\nð Káº¾T QUáº¢ BENCHMARK #2 (ALL 17 INBETWEEN FRAMES - GLUESTICK - FP32):")
    print(df_sum2[df_sum2['group'] == 'ALL_17_INBETWEENS'].round(4).to_string(index=False))

    # Xuáº¥t Video / GIF 24 frames, Error Heatmaps & Side-by-Side (náº¿u cÃ³ cv2 hoáº·c imageio)
    m1_tag = 'M1_Original' if 'M1_Original' in models_dict else None
    m3_tag = 'M3_Best' if 'M3_Best' in models_dict else next((k for k in models_dict if k != 'M1_Original'), None)
    if m1_tag and m3_tag and cv2 is not None:
        sbs_dir = os.path.join(out_dir, "frames_side_by_side")
        heat_dir = os.path.join(out_dir, "error_heatmaps")
        os.makedirs(sbs_dir, exist_ok=True)
        os.makedirs(heat_dir, exist_ok=True)
        gif_frames = []
        montage_rows = []
        # Váº½ thanh Side-by-Side 4 cá»t [Overlay Keyframes | Original M1 | Finetuned M3 | Ground Truth]
        for item in frames_spec:
            idx = int(item['frame_idx'])
            is_k = (item['type'] == 'KEYFRAME')
            gt_u8 = gt_imgs_u8[idx]
            m1_u8 = reconstructed[m1_tag][idx]
            m3_u8 = reconstructed[m3_tag][idx]
            if is_k:
                ov_u8 = gt_u8
                s1, s3 = "KEYFRAME", "KEYFRAME"
            else:
                ka, kb = int(item['ref_key_a']), int(item['ref_key_b'])
                ov_u8 = (0.5 * gt_imgs_u8[ka].astype(np.float32) + 0.5 * gt_imgs_u8[kb].astype(np.float32)).astype(np.uint8)
                fr_row = df_frames[df_frames['frame_idx'] == idx].iloc[0]
                s1 = f"PSNR:{fr_row[f'{m1_tag}_psnr']:.1f}|Sk:{fr_row[f'{m1_tag}_skel_recall']*100:.0f}%"
                s3 = f"PSNR:{fr_row[f'{m3_tag}_psnr']:.1f}|Sk:{fr_row[f'{m3_tag}_skel_recall']*100:.0f}%"

                # Báº£n Äá» nhiá»t sai sá»: [M1 Error Heatmap | M3 Error Heatmap]
                d1 = np.clip(np.abs(m1_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
                d3 = np.clip(np.abs(m3_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
                hm_pair = np.hstack([cv2.applyColorMap(d1, cv2.COLORMAP_JET), cv2.applyColorMap(d3, cv2.COLORMAP_JET)])
                hm_head = np.full((32, w * 2, 3), 24, dtype=np.uint8)
                cv2.putText(hm_head, f"M1 ORIG ERROR [MAE={fr_row[f'{m1_tag}_ie_mae']:.2f}]", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 150, 255), 2, cv2.LINE_AA)
                cv2.putText(hm_head, f"M3 FINE ERROR [MAE={fr_row[f'{m3_tag}_ie_mae']:.2f}]", (w + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(heat_dir, f"heatmap_{idx:02d}_{item['filename']}"), np.vstack([hm_head, hm_pair]))

            head_h = 34
            canvas = np.full((h + head_h, w * 4, 3), 24, dtype=np.uint8)
            cols = [(ov_u8, f"INPUT [{item['filename']}]", (210, 210, 210)),
                    (m1_u8, f"M1 ORIG ({s1})", (100, 150, 255)),
                    (m3_u8, f"M3 FINE ({s3})", (80, 240, 120)),
                    (gt_u8, "GROUND TRUTH", (245, 245, 245))]
            for ci, (arr, lbl, colr) in enumerate(cols):
                canvas[head_h:, ci * w:(ci + 1) * w] = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                cv2.putText(canvas, lbl, (ci * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colr, 2, cv2.LINE_AA)
                if ci > 0:
                    cv2.line(canvas, (ci * w, 0), (ci * w, h + head_h), (70, 70, 70), 2)
            out_png = os.path.join(sbs_dir, f"frame_{idx:02d}_{item['filename']}")
            cv2.imwrite(out_png, canvas)
            half_frame = cv2.resize(canvas, (w * 2, (h + head_h) // 2))
            gif_frames.append(half_frame[:, :, ::-1])
            if (not is_k) and (idx in {2, 5, 9, 12, 17, 21}):
                montage_rows.append(half_frame)

        if montage_rows:
            cv2.imwrite(os.path.join(out_dir, "montage_6segments_M1_vs_M3.png"), np.vstack(montage_rows))

        if imageio is not None:
            gif_path = os.path.join(out_dir, "lineart_test_comparison_M1_vs_M3.gif")
            imageio.mimsave(gif_path, gif_frames, duration=0.14, loop=0)
            print(f"[â] ÄÃ£ xuáº¥t áº£nh Äá»ng Äá»i Äáº§u GIF: {gif_path}")

        mp4_path = os.path.join(out_dir, "lineart_test_comparison_M1_vs_M3.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(mp4_path, fourcc, 8.0, (gif_frames[0].shape[1], gif_frames[0].shape[0]))
        for gf in gif_frames:
            vw.write(gf[:, :, ::-1])
        vw.release()
        print(f"[â] ÄÃ£ xuáº¥t video Äá»i Äáº§u MP4: {mp4_path}")

    return df_sum2


# ==============================================================================
# BENCHMARK #3: DEFENSE GEOMETRY SUITE (HÃNH TRÃN THU/PHÃNG, Dá»CH XA, Ná»N TÄ¨NH)
# ==============================================================================
def extract_hybrid_contour_matches(ink0, ink1, n_pts_per_comp=32):
    """GhÃ©p cáº·p ÄÆ°á»ng bao ngoÃ i theo cá»¥m liÃªn thÃ´ng + láº¥y máº«u Äá»u theo Äá» dÃ i cung."""
    if cv2 is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    h, w = ink0.shape

    def _extract(ink):
        mask = (ink > 0.20).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        feats = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            if peri < 15.0:
                continue
            pts_raw = c.reshape(-1, 2).astype(np.float64)
            diffs = np.diff(np.vstack([pts_raw, pts_raw[0]]), axis=0)
            seglen = np.linalg.norm(diffs, axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seglen)])
            if cum[-1] < 1e-3:
                continue
            targets = np.linspace(0.0, cum[-1], n_pts_per_comp, endpoint=False)
            pts = np.zeros((n_pts_per_comp, 2), dtype=np.float64)
            for k, d in enumerate(targets):
                idx = min(np.searchsorted(cum, d, side="right") - 1, len(pts_raw) - 1)
                alpha = (d - cum[idx]) / max(seglen[idx], 1e-6)
                pts[k] = (1.0 - alpha) * pts_raw[idx] + alpha * pts_raw[(idx + 1) % len(pts_raw)]
            cx_m, cy_m = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - cy_m, pts[:, 0] - cx_m)
            pts = np.roll(pts, -int(np.argmin(np.abs(ang))), axis=0)
            area_s = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
            if area_s < 0:
                pts = np.vstack([pts[0:1], pts[1:][::-1]])
            pts[:, 0] /= max(1.0, w - 1.0)
            pts[:, 1] /= max(1.0, h - 1.0)
            _, _, bw, bh = cv2.boundingRect(c)
            feats.append({'pts': pts.astype(np.float32), 'cx': cx_m / w, 'cy': cy_m / h, 'aspect': bw / max(1.0, bh)})
        return feats

    f0, f1 = _extract(ink0), _extract(ink1)
    if not f0 or not f1:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    cost = np.zeros((len(f0), len(f1)), dtype=np.float64)
    for i, a in enumerate(f0):
        for j, b in enumerate(f1):
            cost[i, j] = 2.0 * abs(a['cy'] - b['cy']) + 0.3 * abs(a['cx'] - b['cx']) + 1.5 * abs(math.log(max(a['aspect'], 0.1) / max(b['aspect'], 0.1)))
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
    except Exception:
        ri = np.arange(min(len(f0), len(f1)))
        ci = np.arange(min(len(f0), len(f1)))
    return np.vstack([f0[i]['pts'] for i in ri]), np.vstack([f1[j]['pts'] for j in ci])


def run_benchmark_defense_geometry_suite(models_dict, device, out_dir, downscale_factor=2):
    """Kiá»m Äá»nh 4 ká»ch báº£n hÃ¬nh há»c dá»ch chuyá»n lá»n + co giÃ£n kÃ­ch thÆ°á»c báº±ng Hybrid Contour + Lagrangian TPS."""
    if cv2 is None:
        print("[*] Thiáº¿u OpenCV, bá» qua Benchmark #3 Defense Geometry Suite.")
        return None
    os.makedirs(out_dir, exist_ok=True)
    H, W = 384, 640
    t_list = [0.25, 0.50, 0.75]

    def _draw_ellipse_case(cx0, cy0, rx0, ry0, th0, cx1, cy1, rx1, ry1, th1, static_line=False):
        def _fn(t):
            im = np.zeros((H, W), dtype=np.float32)
            cx = int(round((1.0 - t) * cx0 + t * cx1))
            cy = int(round((1.0 - t) * cy0 + t * cy1))
            rx = int(round((1.0 - t) * rx0 + t * rx1))
            ry = int(round((1.0 - t) * ry0 + t * ry1))
            th = max(2, int(round((1.0 - t) * th0 + t * th1)))
            cv2.ellipse(im, (cx, cy), (rx, ry), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
            if static_line:
                cv2.line(im, (80, 330), (560, 330), 1.0, 4, lineType=cv2.LINE_AA)
            return im
        return _fn

    cases = [
        ("ShrinkingCircle_75to32", _draw_ellipse_case(120, 192, 75, 75, 5, 520, 192, 32, 32, 4, False)),
        ("ExpandingCircle_30to70", _draw_ellipse_case(120, 192, 30, 30, 4, 520, 192, 70, 70, 5, False)),
        ("ShrinkingCircle_StaticGround", _draw_ellipse_case(120, 180, 75, 75, 5, 520, 180, 32, 32, 4, True)),
        ("SquashOval90x40_to_Circle45", _draw_ellipse_case(140, 192, 90, 40, 4, 490, 192, 45, 45, 4, False)),
    ]

    hw_t = torch.tensor([[H, W]], dtype=torch.int32, device=device)
    rows_per_case = []
    tags = ['M0'] + list(models_dict.keys())

    for cname, gt_fn in cases:
        im0, im1 = gt_fn(0.0), gt_fn(1.0)
        k0, k1 = extract_hybrid_contour_matches(im0, im1, n_pts_per_comp=32)
        x0 = torch.from_numpy(im0)[None, None].to(device)
        x1 = torch.from_numpy(im1)[None, None].to(device)

        vis_rows = {t: [] for t in tags}
        gt_imgs = []
        for tv in t_list:
            gt_im = gt_fn(tv)
            gt_imgs.append(gt_im)
            gt_t = torch.from_numpy(gt_im)[None, None].to(device)
            if solve_multi_layer_tps is not None:
                cf0t, cf1t, _ = solve_multi_layer_tps(
                    k0, k1, (H, W), H, W, device,
                    t_val=tv, direct_t_flows=True, max_pts=128, lambd=0.01,
                    downscale_factor=downscale_factor, eps_static_px=1.5,
                    use_dyn_anchors=False, transition_width_px=18.0, falloff_radius_px=160.0
                )
            else:
                kt = (1.0 - tv) * k0 + tv * k1
                c_t0, th_t0, v0 = solve_tps_direct(kt, k0, max_pts=128, lambd=0.01, use_anchors=False)
                c_t1, th_t1, v1 = solve_tps_direct(kt, k1, max_pts=128, lambd=0.01, use_anchors=False)
                cf0t = tps_flow_from_theta_batch(torch.from_numpy(th_t0)[None].to(device), torch.from_numpy(c_t0)[None].to(device), torch.tensor([v0], device=device), hw_t, H, W, device, downscale_factor)
                cf1t = tps_flow_from_theta_batch(torch.from_numpy(th_t1)[None].to(device), torch.from_numpy(c_t1)[None].to(device), torch.tensor([v1], device=device), hw_t, H, W, device, downscale_factor)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

            rec = {'case': cname, 't_val': tv}
            cw_done = False
            for m_name, mdl in models_dict.items():
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    pr, cw, *_ = mdl(x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)
                if not cw_done:
                    vis_rows['M0'].append(ink_to_u8_gray(cw))
                    mm0 = evaluate_metrics_dual_regime(cw, gt_t)
                    for mk, mv in mm0.items():
                        rec[f"M0_{mk}"] = float(mv.item())
                    cw_done = True
                vis_rows[m_name].append(ink_to_u8_gray(pr))
                mm = evaluate_metrics_dual_regime(pr, gt_t)
                for mk, mv in mm.items():
                    rec[f"{m_name}_{mk}"] = float(mv.item())
            rows_per_case.append(rec)

        # LÆ°u áº£nh lÆ°á»i tá»«ng case
        cols_all = [tags + ['GT']]
        panel_rows = []
        for tg in tags:
            row_u8 = [(255.0 - im0 * 255.0).clip(0, 255).astype(np.uint8)] + vis_rows[tg] + [(255.0 - im1 * 255.0).clip(0, 255).astype(np.uint8)]
            strip = np.hstack([cv2.cvtColor(u, cv2.COLOR_GRAY2BGR) for u in row_u8])
            cv2.putText(strip, tg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 110, 220), 2, cv2.LINE_AA)
            panel_rows.append(strip)
        gt_u8_row = [(255.0 - im0 * 255.0).clip(0, 255).astype(np.uint8)] + [(255.0 - g * 255.0).clip(0, 255).astype(np.uint8) for g in gt_imgs] + [(255.0 - im1 * 255.0).clip(0, 255).astype(np.uint8)]
        gt_strip = np.hstack([cv2.cvtColor(u, cv2.COLOR_GRAY2BGR) for u in gt_u8_row])
        cv2.putText(gt_strip, "GroundTruth", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 150, 40), 2, cv2.LINE_AA)
        panel_rows.append(gt_strip)
        cv2.imwrite(os.path.join(out_dir, f"{cname}_grid.png"), np.vstack(panel_rows))

    df_cases = pd.DataFrame(rows_per_case)
    df_cases.to_csv(os.path.join(out_dir, "defense_geometry_per_frame.csv"), index=False)

    sum_rows = []
    for tg in tags:
        r = {'model_tag': tg, 'frames': len(df_cases)}
        for mc in ['psnr', 'wcd_dog', 'f1', 'density_ratio', 'ghost']:
            r[mc] = float(df_cases[f"{tg}_{mc}"].mean())
        sum_rows.append(r)
    df_sum3 = pd.DataFrame(sum_rows)
    df_sum3.to_csv(os.path.join(out_dir, "defense_geometry_summary.csv"), index=False)

    print("\nð Káº¾T QUáº¢ BENCHMARK #3 (DEFENSE GEOMETRY SUITE - LAGRANGIAN TPS):")
    print(df_sum3.round(4).to_string(index=False))
    return df_sum3


def main():
    parser = argparse.ArgumentParser(description="Cháº¡y Triple-Benchmark Äá»i chá»©ng khÃ¡ch quan (M0 vs M1 vs M3)")
    parser.add_argument('--bench_dir', type=str, default=SCRIPT_DIR)
    parser.add_argument('--orig_ckpt', type=str, default=DEFAULT_ORIG_CKPT)
    parser.add_argument('--finetuned_ckpt', type=str, default=None, help="ÄÆ°á»ng dáº«n model_tps_finetuned_best.pt")
    parser.add_argument('--last_ckpt', type=str, default=None, help="ÄÆ°á»ng dáº«n model_tps_finetuned_last.pt (tÃ¹y chá»n)")
    parser.add_argument('--lineart_test_dir', type=str, default=DEFAULT_LINEART_TEST)
    parser.add_argument('--gluestick_npz', type=str, default=DEFAULT_GLUESTICK_CACHE)
    parser.add_argument('--out_dir', type=str, default=os.path.join(SCRIPT_DIR, "benchmark_results"))
    parser.add_argument('--downscale', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models_dict = {}

    if os.path.exists(args.orig_ckpt):
        m1 = TPSInbetweenOriginal().to(device)
        load_tps_weights(m1, args.orig_ckpt, device=device, prefer_ema=False)
        models_dict['M1_Original'] = m1.eval()
    else:
        print(f"[!] KhÃ´ng tÃ¬m tháº¥y Original Checkpoint: {args.orig_ckpt}")

    if args.finetuned_ckpt and os.path.exists(args.finetuned_ckpt):
        m3_b = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_b, args.finetuned_ckpt, device=device, prefer_ema=True)
        models_dict['M3_Best'] = m3_b.eval()

    if args.last_ckpt and os.path.exists(args.last_ckpt):
        m3_l = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_l, args.last_ckpt, device=device, prefer_ema=True)
        models_dict['M3_Last'] = m3_l.eval()

    if not models_dict:
        print("[!] ChÆ°a náº¡p ÄÆ°á»£c mÃ´ hÃ¬nh nÃ o Äá» Äá»i chuáº©n!")
        sys.exit(1)

    run_benchmark_isolated_sequences(
        args.bench_dir, models_dict, device,
        os.path.join(args.out_dir, "bench1_isolated_sequences"),
        downscale_factor=args.downscale, save_visuals=True
    )
    run_benchmark_continuous_lineart_test(
        args.lineart_test_dir, models_dict, device,
        os.path.join(args.out_dir, "bench2_continuous_lineart_test"),
        gluestick_npz=args.gluestick_npz, downscale_factor=args.downscale
    )
    run_benchmark_defense_geometry_suite(
        models_dict, device,
        os.path.join(args.out_dir, "bench3_defense_geometry"),
        downscale_factor=args.downscale
    )
    print(f"\nð HOÃN Táº¤T BÃO CÃO Äá»I CHUáº¨N Táº I: {args.out_dir}")


if __name__ == '__main__':
    main()
