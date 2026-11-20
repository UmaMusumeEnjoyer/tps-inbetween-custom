#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_p3_inference_pipeline.py
--------------------------------------------------------------------------------
PIPELINE SUY LUẬN & XUẤT FRAME CHÍNH THỨC CỦA PHA 3 (ZERO-COST INFERENCE POLISH)
Đề tài: Nội suy khung hình hoạt họa Anime Lineart (PBL6)

Kết hợp 3 tầng xử lý đồng bộ:
  - Tầng 1: Coarse TPS Đa lớp (Multi-Layer Clustered TPS - `seq_p3_multi_layer_tps.py`)
            Tách cụm nền tĩnh và tiền cảnh động, giải TPS độc lập + mặt nạ C^2.
            Tích hợp cơ chế trích xuất điểm kép: Contour Matching (Lineart) + SIFT Fallback.
  - Tầng 2: Mạng Tinh Chỉnh UNet (`TPSInbetweenOriginal` - `M3_Best` Pha 3 Remake)
            Bảo toàn Skeleton Recall, Tip Recall, mật độ nét 1.002 và giảm WCD_DoG.
  - Tầng 3: Bộ lọc Tương phản Thích ứng Cục bộ (`seq_p3_adaptive_sharpener.py`, mode='balanced')
            Tích hợp tại tầng xuất frame: kéo dốc sườn chuyển tiếp nét, xóa quầng
            mờ halo & vệt ghost trên nền mà vẫn giữ nguyên 100% Skeleton Recall.
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    cv2 = None

try:
    import imageio
except ImportError:
    imageio = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PHASE3_WORKER_DIR = os.path.join(REPO_ROOT, "training", "03_sequence", "p3_multilayer_tps")

for _d in (SCRIPT_DIR, PHASE3_WORKER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from seq_p3_multi_layer_tps import solve_multi_layer_tps
from seq_p3_v2_multilayer_tps_worker import TPSInbetweenOriginal, load_tps_weights
from seq_p3_adaptive_sharpener import apply_adaptive_sharpener

LOCAL_CKPT = os.path.join(SCRIPT_DIR, "model_tps_finetuned_best.pt")
DEFAULT_PHASE3_CKPT = LOCAL_CKPT if os.path.exists(LOCAL_CKPT) else os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3 remake",
    "checkpoints", "model_tps_finetuned_best.pt"
)


def load_ink_from_input(img_input):
    """Chuẩn hóa đầu vào (path, np.ndarray, PIL.Image, torch.Tensor) về numpy float32 ink [0..1] (1=mực đen, 0=nền trắng)."""
    if isinstance(img_input, str):
        with Image.open(img_input) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32)
        return np.clip((255.0 - gray) / 255.0, 0.0, 1.0)
    if isinstance(img_input, Image.Image):
        gray = np.asarray(img_input.convert("L"), dtype=np.float32)
        return np.clip((255.0 - gray) / 255.0, 0.0, 1.0)
    if isinstance(img_input, np.ndarray):
        arr = img_input
        if arr.ndim == 3:
            if cv2 is not None:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            else:
                arr = arr.mean(axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            return np.clip((255.0 - arr) / 255.0, 0.0, 1.0)
        return np.clip(1.0 - arr, 0.0, 1.0)
    if isinstance(img_input, torch.Tensor):
        t = img_input.squeeze().detach().cpu().numpy().astype(np.float32)
        if t.max() > 1.5:
            return np.clip((255.0 - t) / 255.0, 0.0, 1.0)
        return np.clip(t, 0.0, 1.0)
    raise TypeError(f"Không hỗ trợ kiểu dữ liệu đầu vào: {type(img_input)}")


def ink_tensor_to_u8(ink_t):
    """Chuyển tensor ink [0..1] (1=mực đen) về ảnh uint8 [0..255] (0=mực đen, 255=giấy trắng)."""
    arr = (1.0 - ink_t.squeeze().detach().cpu().numpy().clip(0.0, 1.0)) * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def _compute_turning_angles(pts):
    """Tính góc chuyển hướng (turning angle in degrees) tại mỗi đỉnh của đa giác."""
    v = pts.reshape(-1, 2)
    n = len(v)
    if n < 3:
        return np.zeros(n)
    angles = []
    for i in range(n):
        p_prev = v[(i - 1) % n]
        p_curr = v[i]
        p_next = v[(i + 1) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-4 or l2 < 1e-4:
            angles.append(0.0)
            continue
        cos_th = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos_th))))
    return np.array(angles)


def extract_hybrid_contour_matches(ink0, ink1, n_pts_per_comp=32):
    """
    Trích xuất điểm neo dọc theo đường bao nét vẽ Line Art (Polygon-Aware Contour Matching).
    Tự động nhận diện đỉnh góc vuông / đa giác và khóa chặt (pin) các góc để triệt tiêu
    hoàn toàn hiện tượng trượt chu vi (Arc-length Drift) khi tỷ lệ khung hình thay đổi.
    """
    if cv2 is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    h, w = ink0.shape

    def _extract(ink):
        mask = (ink > 0.15).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        feats = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            if peri < 15.0:
                continue
            pts_raw = c.reshape(-1, 2).astype(np.float64)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            verts = approx.reshape(-1, 2).astype(np.float64)
            turning = _compute_turning_angles(verts)
            _, _, bw, bh = cv2.boundingRect(c)
            cx_m, cy_m = pts_raw.mean(axis=0)
            feats.append({
                'cnt': c,
                'pts_raw': pts_raw,
                'verts': verts,
                'turning': turning,
                'peri': peri,
                'cx': cx_m / max(1.0, w),
                'cy': cy_m / max(1.0, h),
                'aspect': bw / max(1.0, float(bh)),
                'bw': bw,
                'bh': bh,
            })
        return feats

    f0, f1 = _extract(ink0), _extract(ink1)
    if not f0 or not f1:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    k0_list, k1_list = [], []
    used1 = set()
    for item0 in f0:
        best_j, best_cost = -1, 1e9
        for j, item1 in enumerate(f1):
            if j in used1:
                continue
            d_pos = np.hypot(item0['cx'] - item1['cx'], item0['cy'] - item1['cy'])
            d_asp = abs(item0['aspect'] - item1['aspect']) / max(0.1, item0['aspect'] + item1['aspect'])
            cost = d_pos * 2.0 + d_asp * 0.5
            if cost < best_cost:
                best_cost, best_j = cost, j
        if best_j >= 0 and best_cost < 1.8:
            used1.add(best_j)
            item1 = f1[best_j]
            v0, v1 = item0['verts'].copy(), item1['verts'].copy()
            n0, n1 = len(v0), len(v1)
            turn0, turn1 = item0['turning'], item1['turning']
            c0, c1 = item0['pts_raw'], item1['pts_raw']

            # Điều kiện đối sánh đa giác có đỉnh góc:
            # Cùng số lượng đỉnh (từ 3 đến 16) và có chứa góc quặt rõ ràng (> 35 độ)
            is_poly = (3 <= n0 <= 16 and n0 == n1 and (np.max(turn0) > 35.0 or np.max(turn1) > 35.0))

            if is_poly:
                # Đồng bộ chiều quay ngược chiều kim đồng hồ
                if cv2.contourArea(v0.astype(np.float32), True) < 0:
                    v0 = v0[::-1]
                if cv2.contourArea(v1.astype(np.float32), True) < 0:
                    v1 = v1[::-1]

                c0_xy = v0.mean(axis=0)
                c1_xy = v1.mean(axis=0)
                best_shift, min_v_dist = 0, 1e12
                for s in range(n0):
                    v1_shifted = np.roll(v1, -s, axis=0)
                    dist = np.sum(np.linalg.norm((v0 - c0_xy) - (v1_shifted - c1_xy), axis=1))
                    if dist < min_v_dist:
                        min_v_dist = dist
                        best_shift = s
                v1_aligned = np.roll(v1, -best_shift, axis=0)

                pts_per_edge = max(2, n_pts_per_comp // n0)
                p0_edge_list, p1_edge_list = [], []
                for e in range(n0):
                    e_next = (e + 1) % n0
                    edge_pts0 = np.linspace(v0[e], v0[e_next], pts_per_edge, endpoint=False)
                    edge_pts1 = np.linspace(v1_aligned[e], v1_aligned[e_next], pts_per_edge, endpoint=False)
                    p0_edge_list.append(edge_pts0)
                    p1_edge_list.append(edge_pts1)
                p0_match = np.vstack(p0_edge_list)
                p1_match = np.vstack(p1_edge_list)
            else:
                # Đồ họa nét cong trơn / hữu cơ: lấy mẫu đều theo chu vi
                diffs0 = np.diff(np.vstack([c0, c0[0]]), axis=0)
                cum0 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs0, axis=1))])
                targets0 = np.linspace(0.0, cum0[-1], n_pts_per_comp, endpoint=False)
                p0_match = np.zeros((n_pts_per_comp, 2), dtype=np.float64)
                for k, d in enumerate(targets0):
                    idx = min(np.searchsorted(cum0, d, side="right") - 1, len(c0) - 1)
                    denom = max(np.linalg.norm(diffs0[idx]), 1e-6)
                    alpha = (d - cum0[idx]) / denom
                    p0_match[k] = (1.0 - alpha) * c0[idx] + alpha * c0[(idx + 1) % len(c0)]

                diffs1 = np.diff(np.vstack([c1, c1[0]]), axis=0)
                cum1 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs1, axis=1))])
                targets1 = np.linspace(0.0, cum1[-1], n_pts_per_comp, endpoint=False)
                p1_match = np.zeros((n_pts_per_comp, 2), dtype=np.float64)
                for k, d in enumerate(targets1):
                    idx = min(np.searchsorted(cum1, d, side="right") - 1, len(c1) - 1)
                    denom = max(np.linalg.norm(diffs1[idx]), 1e-6)
                    alpha = (d - cum1[idx]) / denom
                    p1_match[k] = (1.0 - alpha) * c1[idx] + alpha * c1[(idx + 1) % len(c1)]

                cx0_m, cy0_m = p0_match.mean(axis=0)
                cx1_m, cy1_m = p1_match.mean(axis=0)
                ang0 = np.arctan2(p0_match[:, 1] - cy0_m, p0_match[:, 0] - cx0_m)
                ang1 = np.arctan2(p1_match[:, 1] - cy1_m, p1_match[:, 0] - cx1_m)
                p0_match = np.roll(p0_match, -int(np.argmin(np.abs(ang0))), axis=0)
                p1_match = np.roll(p1_match, -int(np.argmin(np.abs(ang1))), axis=0)

                area_s0 = 0.5 * np.sum(p0_match[:, 0] * np.roll(p0_match[:, 1], -1) - np.roll(p0_match[:, 0], -1) * p0_match[:, 1])
                if area_s0 < 0:
                    p0_match = np.vstack([p0_match[0:1], p0_match[1:][::-1]])
                area_s1 = 0.5 * np.sum(p1_match[:, 0] * np.roll(p1_match[:, 1], -1) - np.roll(p1_match[:, 0], -1) * p1_match[:, 1])
                if area_s1 < 0:
                    p1_match = np.vstack([p1_match[0:1], p1_match[1:][::-1]])

            p0_norm = p0_match.copy()
            p1_norm = p1_match.copy()
            p0_norm[:, 0] /= max(1.0, w - 1.0)
            p0_norm[:, 1] /= max(1.0, h - 1.0)
            p1_norm[:, 0] /= max(1.0, w - 1.0)
            p1_norm[:, 1] /= max(1.0, h - 1.0)
            k0_list.append(p0_norm)
            k1_list.append(p1_norm)

    if not k0_list:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    k0_all = np.vstack(k0_list)
    k1_all = np.vstack(k1_list)
    return k0_all.astype(np.float32), k1_all.astype(np.float32)


def extract_sift_keypoints_pair(ink0, ink1, max_pts=120):
    """Trích xuất điểm neo SIFT + lọc RANSAC trên không gian chuẩn hóa [0, 1]."""
    if cv2 is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    h, w = ink0.shape
    u0 = np.clip((1.0 - ink0) * 255.0, 0, 255).astype(np.uint8)
    u1 = np.clip((1.0 - ink1) * 255.0, 0, 255).astype(np.uint8)

    sift = cv2.SIFT_create(nfeatures=1000)
    k0, d0 = sift.detectAndCompute(u0, None)
    k1, d1 = sift.detectAndCompute(u1, None)
    if d0 is None or d1 is None or len(k0) < 4 or len(k1) < 4:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(d0, d1)
    if len(matches) < 4:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    p0 = np.float32([k0[m.queryIdx].pt for m in matches])
    p1 = np.float32([k1[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 15.0)
    if mask is not None and int(mask.sum()) >= 4:
        p0, p1 = p0[mask.ravel() == 1], p1[mask.ravel() == 1]

    p0[:, 0] /= max(1.0, float(w))
    p0[:, 1] /= max(1.0, float(h))
    p1[:, 0] /= max(1.0, float(w))
    p1[:, 1] /= max(1.0, float(h))
    if len(p0) > max_pts:
        idx = np.linspace(0, len(p0) - 1, max_pts, dtype=int)
        p0, p1 = p0[idx], p1[idx]
    return p0.astype(np.float32), p1.astype(np.float32)


def extract_auto_keypoints_pair(ink0, ink1):
    """
    Cơ chế trích xuất điểm neo thông minh:
    1. Ưu tiên Contour Matching cho Line Art / vẽ tay.
    2. Fallback sang SIFT nếu là ảnh tả thực có chất liệu phong phú.
    """
    k0, k1 = extract_hybrid_contour_matches(ink0, ink1, n_pts_per_comp=32)
    if len(k0) >= 8:
        return k0, k1
    s0, s1 = extract_sift_keypoints_pair(ink0, ink1)
    if len(s0) > len(k0):
        return s0, s1
    return k0, k1


class Phase3InferencePipeline:
    """
    Pipeline suy luận & xuất frame chính thức của Pha 3 Remake,
    tích hợp sẵn bộ lọc Adaptive Local Edge Sharpener (mode='balanced').
    """

    def __init__(
        self,
        checkpoint_path=DEFAULT_PHASE3_CKPT,
        device=None,
        prefer_ema=True,
        use_sharpener=True,
        sharpener_mode="balanced",
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_sharpener = bool(use_sharpener)
        self.sharpener_mode = str(sharpener_mode)
        self.model = TPSInbetweenOriginal().to(self.device).eval()
        self.checkpoint_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint Pha 3 tại: {checkpoint_path}")
        load_tps_weights(self.model, checkpoint_path, device=self.device, prefer_ema=prefer_ema)

    def compute_multilayer_flows(self, ink0, ink1, kps0=None, kps1=None, t_val=None, direct_t_flows=False):
        """Giải trường luồng Coarse TPS Đa lớp cho cặp keyframe."""
        h0, w0 = ink0.shape
        h1, w1 = ink1.shape
        h, w = max(h0, h1), max(w0, w1)
        if (h0, w0) != (h, w):
            ink0 = np.pad(ink0, ((0, h - h0), (0, w - w0)), mode="constant", constant_values=0.0)
        if (h1, w1) != (h, w):
            ink1 = np.pad(ink1, ((0, h - h1), (0, w - w1)), mode="constant", constant_values=0.0)

        h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32

        if kps0 is None or kps1 is None:
            kps0, kps1 = extract_auto_keypoints_pair(ink0, ink1)

        if len(kps0) > 0:
            cf01, cf10, seg_info = solve_multi_layer_tps(
                kps0, kps1, (h, w), h_pad, w_pad, self.device,
                t_val=t_val, direct_t_flows=direct_t_flows,
                max_pts=128, lambd=0.01, downscale_factor=2, eps_static_px=1.5,
                use_dyn_anchors=False if direct_t_flows else True,
                transition_width_px=18.0 if direct_t_flows else 22.0,
                falloff_radius_px=160.0 if direct_t_flows else 105.0,
            )
        else:
            cf01 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=self.device)
            cf10 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=self.device)
            seg_info = {"n_matches": 0}
        return cf01, cf10, (h, w), (h_pad, w_pad), seg_info, kps0, kps1, ink0, ink1

    def interpolate_with_precomputed_flows(
        self,
        x0_pad,
        x1_pad,
        cf01,
        cf10,
        orig_hw,
        t,
        use_sharpener=None,
        sharpener_mode=None,
        direct_t_flows=False,
    ):
        """Chạy mạng UNet + bộ lọc xuất frame khi đã có Coarse TPS flows."""
        h, w = orig_hw
        do_sharp = self.use_sharpener if use_sharpener is None else bool(use_sharpener)
        s_mode = self.sharpener_mode if sharpener_mode is None else str(sharpener_mode)

        tt = torch.tensor([[[[float(t)]]]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    pr_pad, cw_pad, *_ = self.model(
                        x0_pad.float(), x1_pad.float(), tt, cf01.float(), cf10.float(),
                        direct_t_flows=direct_t_flows
                    )
            else:
                pr_pad, cw_pad, *_ = self.model(
                    x0_pad.float(), x1_pad.float(), tt, cf01.float(), cf10.float(),
                    direct_t_flows=direct_t_flows
                )

        raw_ink = pr_pad[:, :, :h, :w].clamp(0.0, 1.0)
        coarse_ink = cw_pad[:, :, :h, :w].clamp(0.0, 1.0)

        if do_sharp:
            final_ink = apply_adaptive_sharpener(raw_ink, mode=s_mode, input_is_ink=True)
        else:
            final_ink = raw_ink

        return {
            "inbetween_u8": ink_tensor_to_u8(final_ink),
            "raw_u8": ink_tensor_to_u8(raw_ink),
            "coarse_u8": ink_tensor_to_u8(coarse_ink),
            "ink_tensor": final_ink,
            "raw_ink_tensor": raw_ink,
            "coarse_ink_tensor": coarse_ink,
        }

    def interpolate_pair(
        self,
        img0,
        img1,
        t=0.5,
        kps0=None,
        kps1=None,
        use_sharpener=None,
        sharpener_mode=None,
        direct_t_flows=True,
    ):
        """Suy luận đầu cuối một khung hình trung gian tại thời điểm t in (0, 1)."""
        ink0 = load_ink_from_input(img0)
        ink1 = load_ink_from_input(img1)

        cf0t, cf1t, (h, w), (h_pad, w_pad), seg_info, _, _, ink0_pad, ink1_pad = self.compute_multilayer_flows(
            ink0, ink1, kps0=kps0, kps1=kps1, t_val=t, direct_t_flows=direct_t_flows
        )

        x0 = F.pad(torch.from_numpy(ink0_pad)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)
        x1 = F.pad(torch.from_numpy(ink1_pad)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)

        res = self.interpolate_with_precomputed_flows(
            x0, x1, cf0t, cf1t, (h, w), t,
            use_sharpener=use_sharpener, sharpener_mode=sharpener_mode,
            direct_t_flows=direct_t_flows,
        )
        res["seg_info"] = seg_info
        return res

    def export_sequence(
        self,
        frame_paths,
        keyframe_indices,
        out_dir,
        gluestick_npz=None,
        use_sharpener=None,
        sharpener_mode=None,
        save_video=True,
        fps=8,
    ):
        """
        Nội suy và xuất trọn bộ chuỗi frame sạch (và video MP4/GIF) từ danh sách keyframes.
        """
        os.makedirs(out_dir, exist_ok=True)
        frames_dir = os.path.join(out_dir, "exported_frames")
        os.makedirs(frames_dir, exist_ok=True)

        gs_cache = np.load(gluestick_npz) if (gluestick_npz and os.path.exists(gluestick_npz)) else None
        total_frames = len(frame_paths)
        exported_u8 = [None] * total_frames

        # Lưu các keyframe gốc
        for idx in keyframe_indices:
            ink_kf = load_ink_from_input(frame_paths[idx])
            kf_u8 = np.clip((1.0 - ink_kf) * 255.0, 0, 255).astype(np.uint8)
            exported_u8[idx] = kf_u8
            out_name = os.path.basename(frame_paths[idx])
            Image.fromarray(kf_u8).save(os.path.join(frames_dir, out_name))

        # Nội suy các đoạn giữa các keyframe
        t0 = time.time()
        inbet_count = 0
        for s_idx in range(len(keyframe_indices) - 1):
            i0 = keyframe_indices[s_idx]
            i1 = keyframe_indices[s_idx + 1]
            ink0 = load_ink_from_input(frame_paths[i0])
            ink1 = load_ink_from_input(frame_paths[i1])
            h, w = ink0.shape
            h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32

            seg_key = f"seg_{i0}_{i1}"
            if gs_cache is not None and seg_key in gs_cache:
                kps = gs_cache[seg_key].astype(np.float32)
                kps0, kps1 = kps[0], kps[1]
            else:
                kps0, kps1 = extract_auto_keypoints_pair(ink0, ink1)

            x0 = F.pad(torch.from_numpy(ink0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)
            x1 = F.pad(torch.from_numpy(ink1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)

            for f_idx in range(i0 + 1, i1):
                tv = float(f_idx - i0) / float(i1 - i0)
                cf0t, cf1t, (h, w), (h_pad, w_pad), _, _, _, _, _ = self.compute_multilayer_flows(
                    ink0, ink1, kps0=kps0, kps1=kps1, t_val=tv, direct_t_flows=True
                )
                res = self.interpolate_with_precomputed_flows(
                    x0, x1, cf0t, cf1t, (h, w), tv,
                    use_sharpener=use_sharpener, sharpener_mode=sharpener_mode,
                    direct_t_flows=True,
                )
                out_u8 = res["inbetween_u8"]
                exported_u8[f_idx] = out_u8
                out_name = os.path.basename(frame_paths[f_idx])
                Image.fromarray(out_u8).save(os.path.join(frames_dir, out_name))
                inbet_count += 1

        elapsed = time.time() - t0
        print(f"[✓] Đã xuất {total_frames} frames ({inbet_count} in-betweens) trong {elapsed:.2f}s -> {frames_dir}")

        if save_video and imageio is not None:
            rgb_frames = [np.stack([f] * 3, axis=-1) for f in exported_u8 if f is not None]
            gif_path = os.path.join(out_dir, "phase3_sharp_sequence.gif")
            imageio.mimsave(gif_path, rgb_frames, fps=fps, loop=0)
            print(f"[✓] Đã xuất GIF: {gif_path}")
            try:
                mp4_path = os.path.join(out_dir, "phase3_sharp_sequence.mp4")
                imageio.mimsave(mp4_path, rgb_frames, fps=fps, macro_block_size=1)
                print(f"[✓] Đã xuất MP4: {mp4_path}")
            except Exception:
                pass

        return exported_u8


def main():
    parser = argparse.ArgumentParser(description="Pipeline suy luận & xuất frame Pha 3 Remake + Balanced Sharpener")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_PHASE3_CKPT, help="Đường dẫn checkpoint M3_Best")
    parser.add_argument("--img0", type=str, default=None, help="Ảnh keyframe đầu (chạy đơn lẻ)")
    parser.add_argument("--img1", type=str, default=None, help="Ảnh keyframe cuối (chạy đơn lẻ)")
    parser.add_argument("--t", type=float, default=0.5, help="Thời điểm nội suy t in (0, 1)")
    parser.add_argument("--out", type=str, default="inbetween_sharp.png", help="File ảnh đầu ra khi chạy đơn lẻ")
    parser.add_argument("--seq_dir", type=str, default=None, help="Thư mục chuỗi frame test (ví dụ data/lineart_experiments/test)")
    parser.add_argument("--out_dir", type=str, default="phase3_export_out", help="Thư mục xuất chuỗi")
    parser.add_argument("--mode", type=str, default="balanced", choices=["gentle", "balanced", "crisp"], help="Chế độ Adaptive Sharpener")
    parser.add_argument("--no_sharpener", action="store_true", help="Tắt bộ lọc sharpener (xuất raw)")
    args = parser.parse_args()

    pipe = Phase3InferencePipeline(
        checkpoint_path=args.ckpt,
        use_sharpener=not args.no_sharpener,
        sharpener_mode=args.mode,
    )

    if args.img0 and args.img1:
        res = pipe.interpolate_pair(args.img0, args.img1, t=args.t)
        Image.fromarray(res["inbetween_u8"]).save(args.out)
        print(f"[✓] Đã lưu ảnh nội suy: {args.out}")
    elif args.seq_dir:
        paths = sorted(glob.glob(os.path.join(args.seq_dir, "*.png")))
        kf_idx = [0, 3, 7, 11, 14, 19, 23] if len(paths) == 24 else list(range(0, len(paths), 4))
        if kf_idx[-1] != len(paths) - 1:
            kf_idx.append(len(paths) - 1)
        gs_npz = os.path.join(REPO_ROOT, "data_samples", "caches", "lineart_test_gluestick_matches.npz")
        pipe.export_sequence(paths, kf_idx, args.out_dir, gluestick_npz=gs_npz)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
