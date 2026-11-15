#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_p4_occlusion_pipeline.py
--------------------------------------------------------------------------------
PIPELINE SUY LUẬN & XUẤT FRAME PHA 4 (OCCLUSION-AWARE TPS INBETWEEN)
Đề tài: Nội suy khung hình hoạt họa Anime Lineart (PBL6)
Không gian làm việc cách ly: Phase 4 Occlusion Head (M4_Occ)

Các tính năng nổi bật:
  - Tầng 1: Coarse TPS Đa lớp (Multi-Layer Clustered TPS - `seq_p3_multi_layer_tps.py`)
  - Tầng 2: Mạng TPSInbetweenOcclusion với Occlusion Head tại Block 2
            Xuất trực tiếp luồng quang học và 2 mặt nạ che khuất [M0->t, M1->t].
  - Tầng 3: Bộ Trộn Trọng số Chuẩn hóa (Normalized Visibility Blending):
            W0 = (1 - t) * M0->t,  W1 = t * M1->t
            I_fused = (W0 * fine_I0t + W1 * fine_I1t) / (W0 + W1 + eps)
  - Tầng 4: Xuất trực quan bản đồ hiển thị / che khuất (M0, M1, Blend Weight)
            giúp trực quan hóa chính xác vị trí các nét xuất hiện / biến mất (occlusion / disocclusion).
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
PHASE4_DIR = os.path.join(REPO_ROOT, "training", "03_sequence", "p4_occlusion_head")

for _d in (SCRIPT_DIR, PHASE4_DIR, os.path.join(REPO_ROOT, "src", "sequence_p3_p4")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from seq_p3_multi_layer_tps import solve_multi_layer_tps
from seq_p4_occlusion_model import (
    TPSInbetweenOcclusion,
    load_tps_occlusion_weights,
    warp_normalized_flow,
    warp_pixel_flow,
)

DEFAULT_PHASE4_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 4 Occlusion Head",
    "model_tps_occlusion_best.pt"
)
FALLBACK_PHASE3_CKPT = os.path.join(
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


def mask_tensor_to_u8(mask_t):
    """Chuyển tensor mặt nạ [0..1] (1=visible/trắng, 0=occluded/đen) về ảnh uint8 [0..255]."""
    arr = mask_t.squeeze().detach().cpu().numpy().clip(0.0, 1.0) * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


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


class Phase4OcclusionPipeline:
    """
    Pipeline suy luận & phân tích che khuất (Occlusion) chính thức của Phase 4.
    """

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        prefer_ema=True,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            if os.path.exists(DEFAULT_PHASE4_CKPT):
                checkpoint_path = DEFAULT_PHASE4_CKPT
            else:
                checkpoint_path = FALLBACK_PHASE3_CKPT
                print(f"[*] Checkpoint Phase 4 chưa hoàn tất trên Kaggle -> Warm-Start từ Phase 3: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.model = TPSInbetweenOcclusion().to(self.device).eval()

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {checkpoint_path}")

        load_tps_occlusion_weights(self.model, checkpoint_path, device=self.device, prefer_ema=prefer_ema)

    def compute_multilayer_flows(self, ink0, ink1, kps0=None, kps1=None):
        """Giải trường luồng Coarse TPS Đa lớp cho cặp keyframe."""
        h, w = ink0.shape
        h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32

        if kps0 is None or kps1 is None:
            kps0, kps1 = extract_sift_keypoints_pair(ink0, ink1)

        if len(kps0) > 0:
            cf01, cf10, seg_info = solve_multi_layer_tps(
                kps0, kps1, (h, w), h_pad, w_pad, self.device,
                direct_t_flows=False, max_pts=128, lambd=0.01,
                downscale_factor=2, eps_static_px=1.5,
                use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0,
            )
        else:
            cf01 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=self.device)
            cf10 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=self.device)
            seg_info = {"n_matches": 0}
        return cf01, cf10, (h, w), (h_pad, w_pad), seg_info

    def interpolate_with_precomputed_flows(
        self,
        x0_pad,
        x1_pad,
        cf01,
        cf10,
        orig_hw,
        t,
    ):
        """Chạy mạng TPSInbetweenOcclusion với coarse flows đã tính sẵn."""
        h, w = orig_hw
        tt = torch.tensor([[[[float(t)]]]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    pr_pad, cw_pad, f0_pad, f1_pad, flow_pad, M0_pad, M1_pad = self.model(
                        x0_pad.float(), x1_pad.float(), tt, cf01.float(), cf10.float()
                    )
            else:
                pr_pad, cw_pad, f0_pad, f1_pad, flow_pad, M0_pad, M1_pad = self.model(
                    x0_pad.float(), x1_pad.float(), tt, cf01.float(), cf10.float()
                )

        raw_ink = pr_pad[:, :, :h, :w].clamp(0.0, 1.0)
        coarse_ink = cw_pad[:, :, :h, :w].clamp(0.0, 1.0)
        M0_crop = M0_pad[:, :, :h, :w].clamp(0.0, 1.0)
        M1_crop = M1_pad[:, :, :h, :w].clamp(0.0, 1.0)

        # Tính tỷ lệ trộn thích ứng (Normalized Visibility Fraction W1)
        w0 = (1.0 - float(t)) * M0_crop
        w1 = float(t) * M1_crop
        blend_fraction = w1 / (w0 + w1 + 1e-6)

        return {
            "inbetween_u8": ink_tensor_to_u8(raw_ink),
            "coarse_u8": ink_tensor_to_u8(coarse_ink),
            "M0_u8": mask_tensor_to_u8(M0_crop),
            "M1_u8": mask_tensor_to_u8(M1_crop),
            "blend_weight_u8": mask_tensor_to_u8(blend_fraction),
            "ink_tensor": raw_ink,
            "coarse_ink_tensor": coarse_ink,
            "M0_tensor": M0_crop,
            "M1_tensor": M1_crop,
            "blend_fraction_tensor": blend_fraction,
        }

    def interpolate_pair(
        self,
        img0,
        img1,
        t=0.5,
        kps0=None,
        kps1=None,
    ):
        """Suy luận đầu cuối một khung hình trung gian tại thời điểm t in (0, 1)."""
        ink0 = load_ink_from_input(img0)
        ink1 = load_ink_from_input(img1)
        cf01, cf10, (h, w), (h_pad, w_pad), seg_info = self.compute_multilayer_flows(ink0, ink1, kps0=kps0, kps1=kps1)

        x0 = F.pad(torch.from_numpy(ink0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)
        x1 = F.pad(torch.from_numpy(ink1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)

        res = self.interpolate_with_precomputed_flows(x0, x1, cf01, cf10, (h, w), t)
        res["seg_info"] = seg_info
        return res

    def export_sequence(
        self,
        frame_paths,
        keyframe_indices,
        out_dir,
        gluestick_npz=None,
        save_video=True,
        save_masks=True,
        fps=8,
    ):
        """Nội suy và xuất trọn bộ chuỗi frame kèm các mặt nạ che khuất M0, M1."""
        os.makedirs(out_dir, exist_ok=True)
        frames_dir = os.path.join(out_dir, "exported_frames")
        masks_dir = os.path.join(out_dir, "exported_occlusion_masks")
        os.makedirs(frames_dir, exist_ok=True)
        if save_masks:
            os.makedirs(masks_dir, exist_ok=True)

        gs_cache = np.load(gluestick_npz) if (gluestick_npz and os.path.exists(gluestick_npz)) else None
        total_frames = len(frame_paths)
        exported_u8 = [None] * total_frames

        # Lưu keyframes gốc
        for idx in keyframe_indices:
            ink_kf = load_ink_from_input(frame_paths[idx])
            kf_u8 = np.clip((1.0 - ink_kf) * 255.0, 0, 255).astype(np.uint8)
            exported_u8[idx] = kf_u8
            out_name = os.path.basename(frame_paths[idx])
            Image.fromarray(kf_u8).save(os.path.join(frames_dir, out_name))

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
                kps0, kps1 = extract_sift_keypoints_pair(ink0, ink1)

            cf01, cf10, _, _, _ = self.compute_multilayer_flows(ink0, ink1, kps0=kps0, kps1=kps1)
            x0 = F.pad(torch.from_numpy(ink0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)
            x1 = F.pad(torch.from_numpy(ink1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(self.device)

            for f_idx in range(i0 + 1, i1):
                tv = float(f_idx - i0) / float(i1 - i0)
                res = self.interpolate_with_precomputed_flows(x0, x1, cf01, cf10, (h, w), tv)
                out_u8 = res["inbetween_u8"]
                exported_u8[f_idx] = out_u8
                out_name = os.path.basename(frame_paths[f_idx])
                Image.fromarray(out_u8).save(os.path.join(frames_dir, out_name))

                if save_masks:
                    stem = os.path.splitext(out_name)[0]
                    Image.fromarray(res["M0_u8"]).save(os.path.join(masks_dir, f"{stem}_M0.png"))
                    Image.fromarray(res["M1_u8"]).save(os.path.join(masks_dir, f"{stem}_M1.png"))
                    Image.fromarray(res["blend_weight_u8"]).save(os.path.join(masks_dir, f"{stem}_BlendWeight.png"))

                inbet_count += 1

        elapsed = time.time() - t0
        print(f"[✓] Đã xuất {total_frames} frames ({inbet_count} in-betweens) trong {elapsed:.2f}s -> {frames_dir}")

        if save_video and imageio is not None:
            rgb_frames = [np.stack([f] * 3, axis=-1) for f in exported_u8 if f is not None]
            gif_path = os.path.join(out_dir, "phase4_occlusion_sequence.gif")
            imageio.mimsave(gif_path, rgb_frames, fps=fps, loop=0)
            print(f"[✓] Đã xuất GIF: {gif_path}")
            try:
                mp4_path = os.path.join(out_dir, "phase4_occlusion_sequence.mp4")
                imageio.mimsave(mp4_path, rgb_frames, fps=fps, macro_block_size=1)
                print(f"[✓] Đã xuất MP4: {mp4_path}")
            except Exception:
                pass

        return exported_u8


def main():
    parser = argparse.ArgumentParser(description="Pipeline suy luận Phase 4 Occlusion Head")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_PHASE4_CKPT)
    parser.add_argument("--img0", type=str, default=None)
    parser.add_argument("--img1", type=str, default=None)
    parser.add_argument("--t", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="inbetween_occlusion.png")
    parser.add_argument("--seq_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="phase4_export_out")
    args = parser.parse_args()

    pipe = Phase4OcclusionPipeline(checkpoint_path=args.ckpt)

    if args.img0 and args.img1:
        res = pipe.interpolate_pair(args.img0, args.img1, t=args.t)
        Image.fromarray(res["inbetween_u8"]).save(args.out)
        print(f"[✓] Đã lưu ảnh nội suy: {args.out}")
        out_stem = os.path.splitext(args.out)[0]
        Image.fromarray(res["M0_u8"]).save(f"{out_stem}_M0.png")
        Image.fromarray(res["M1_u8"]).save(f"{out_stem}_M1.png")
        print(f"[✓] Đã lưu mặt nạ che khuất: {out_stem}_M0.png & {out_stem}_M1.png")
    elif args.seq_dir:
        paths = sorted(glob.glob(os.path.join(args.seq_dir, "*.png")))
        kf_idx = [0, 3, 7, 11, 14, 19, 23] if len(paths) == 24 else list(range(0, len(paths), 4))
        if kf_idx[-1] != len(paths) - 1:
            kf_idx.append(len(paths) - 1)
        pipe.export_sequence(paths, kf_idx, args.out_dir)


if __name__ == "__main__":
    main()
