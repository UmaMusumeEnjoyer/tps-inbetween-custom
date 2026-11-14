#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_sharpener.py
--------------------------------------------------------------------------------
Kiểm chứng trực quan bộ lọc Adaptive Local Line Sharpener trên các frame
thực tế của Pha 3 Remake (M3_Best) vs Pha 2 Blended vs Ground Truth.
--------------------------------------------------------------------------------
"""

import os
import cv2
import numpy as np

ARTIFACT_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\2c2f8fce-6bc8-4068-bbc5-8671efcb7aa4"
BENCH2_DIR = r"E:\CODE\PBL\PBL6\checkpoints\Re-train\Sequence 2 pha 3 remake\benchmark_outputs\bench2_continuous_lineart_test\frames_side_by_side"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def apply_adaptive_sharpener(gray_u8, mode='balanced'):
    f = gray_u8.astype(np.float32) / 255.0
    
    if mode == 'gentle':
        alpha, beta = 0.05, 0.88
        t = np.clip((f - alpha) / (beta - alpha), 0.0, 1.0)
        out = t * t * (3.0 - 2.0 * t)
        out[f > 0.88] = 1.0
    elif mode == 'balanced':
        # Hàm sigmoid dốc tại x0=0.55
        k = 9.5
        x0 = 0.55
        sig = 1.0 / (1.0 + np.exp(-k * (f - x0)))
        sig_0 = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
        sig_1 = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
        out = (sig - sig_0) / (sig_1 - sig_0)
        # Nền mờ xa nét (> 0.82) kéo trắng hoàn toàn
        out[f > 0.82] = 1.0
    elif mode == 'crisp':
        alpha, beta = 0.10, 0.76
        t = np.clip((f - alpha) / (beta - alpha), 0.0, 1.0)
        out = t * t * (3.0 - 2.0 * t)
        out[f > 0.76] = 1.0
        
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def compute_quick_metrics(pred_u8, gt_u8):
    # ink mask: đen = mực (pixel < 128)
    p_ink = pred_u8 < 128
    g_ink = gt_u8 < 128
    
    # Skeleton of GT
    from skimage.morphology import skeletonize
    g_skel = skeletonize(g_ink)
    
    # Skel recall
    skel_total = g_skel.sum()
    if skel_total > 0:
        skel_rec = (p_ink & g_skel).sum() / float(skel_total) * 100.0
    else:
        skel_rec = 100.0
        
    # Density ratio
    g_cnt = g_ink.sum()
    p_cnt = p_ink.sum()
    dens = (p_cnt / float(g_cnt)) if g_cnt > 0 else 1.0
    
    # Ghost pixels: p_ink mà g_ink mở rộng 5px vẫn là false
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    g_dil = cv2.dilate(g_ink.astype(np.uint8), kernel) > 0
    ghost_px = int((p_ink & (~g_dil)).sum())
    
    return skel_rec, dens, ghost_px


def main():
    target_frames = [
        ("frame_03_0.3s.png", "Frame 03 (0.3s - Chuyen dong tay & toc)", (120, 320, 260, 460)),
        ("frame_07_0.8s.png", "Frame 07 (0.8s - Bien dang xoay nhanh)", (160, 380, 300, 520)),
        ("frame_14_1.7s.png", "Frame 14 (1.7s - Net mat & bo vai)", (140, 350, 280, 490)),
    ]
    
    output_montages = []
    
    for fname, title, (y0, x0, y1, x1) in target_frames:
        fpath = os.path.join(BENCH2_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[!] Khong thay {fpath}")
            continue
            
        full_img = cv2.imread(fpath)
        h, w, _ = full_img.shape
        pw = w // 4  # 960
        
        # Bỏ header 34px trên cùng
        gt_u8 = cv2.cvtColor(full_img[34:, 0:pw], cv2.COLOR_BGR2GRAY)
        m1_u8 = cv2.cvtColor(full_img[34:, pw:2*pw], cv2.COLOR_BGR2GRAY)
        p2_u8 = cv2.cvtColor(full_img[34:, 2*pw:3*pw], cv2.COLOR_BGR2GRAY)
        m3_raw_u8 = cv2.cvtColor(full_img[34:, 3*pw:4*pw], cv2.COLOR_BGR2GRAY)
        
        # Áp dụng bộ lọc
        m3_sharp_bal = apply_adaptive_sharpener(m3_raw_u8, mode='balanced')
        m3_sharp_crisp = apply_adaptive_sharpener(m3_raw_u8, mode='crisp')
        
        # Đo chỉ số
        rec_p2, dens_p2, ghost_p2 = compute_quick_metrics(p2_u8, gt_u8)
        rec_raw, dens_raw, ghost_raw = compute_quick_metrics(m3_raw_u8, gt_u8)
        rec_bal, dens_bal, ghost_bal = compute_quick_metrics(m3_sharp_bal, gt_u8)
        rec_crp, dens_crp, ghost_crp = compute_quick_metrics(m3_sharp_crisp, gt_u8)
        
        print(f"=== {title} ===")
        print(f"  Pha 2 Blended   : SkelRec={rec_p2:5.1f}%, Density={dens_p2:5.3f}, Ghost={ghost_p2:5d} px")
        print(f"  Pha 3 M3 (Raw)  : SkelRec={rec_raw:5.1f}%, Density={dens_raw:5.3f}, Ghost={ghost_raw:5d} px (Bi mo/halo)")
        print(f"  Pha 3 + Balanced: SkelRec={rec_bal:5.1f}%, Density={dens_bal:5.3f}, Ghost={ghost_bal:5d} px (Sac net, giu recall)")
        print(f"  Pha 3 + Crisp   : SkelRec={rec_crp:5.1f}%, Density={dens_crp:5.3f}, Ghost={ghost_crp:5d} px (Manga style)")
        
        # Tạo panel so sánh
        # 4 Cột: [Pha2_Blended] | [Pha 3 M3 Raw] | [Pha 3 + Balanced Sharpener] | [Ground Truth]
        panels = [
            (p2_u8, f"Pha 2 Blended (Ghost={ghost_p2}px)", (40, 120, 220)),
            (m3_raw_u8, f"Pha 3 M3 Raw (Mo/Halo - Ghost={ghost_raw}px)", (0, 0, 200)),
            (m3_sharp_bal, f"Pha 3 + Balanced Sharp (Sac net - Ghost={ghost_bal}px)", (0, 160, 40)),
            (gt_u8, f"Ground Truth (Chuan)", (60, 60, 60))
        ]
        
        # Row 1: Toàn cảnh thu nhỏ
        row1_imgs = []
        for p_img, p_lbl, col in panels:
            bgr = cv2.cvtColor(p_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(bgr, p_lbl, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
            row1_imgs.append(bgr)
        row1 = np.hstack(row1_imgs)
        
        # Row 2: Zoom-in 300% vào chi tiết viền nét
        crop_h = y1 - y0
        crop_w = x1 - x0
        row2_imgs = []
        for p_img, p_lbl, col in panels:
            sub = p_img[y0:y1, x0:x1]
            sub_zoom = cv2.resize(sub, (pw, int(pw * (crop_h / float(crop_w)))), interpolation=cv2.INTER_NEAREST)
            bgr_zoom = cv2.cvtColor(sub_zoom, cv2.COLOR_GRAY2BGR)
            cv2.putText(bgr_zoom, f"ZOOM 3.5x: {p_lbl.split('(')[0].strip()}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2, cv2.LINE_AA)
            # Khung viền
            cv2.rectangle(bgr_zoom, (0, 0), (bgr_zoom.shape[1]-1, bgr_zoom.shape[0]-1), col, 3)
            row2_imgs.append(bgr_zoom)
        row2 = np.hstack(row2_imgs)
        
        # Vẽ ô vuông đỏ chỉ vị trí crop lên panel Ground Truth ở row 1
        cv2.rectangle(row1_imgs[3], (x0, y0), (x1, y1), (0, 0, 255), 2)
        row1 = np.hstack(row1_imgs)
        
        # Ghép 2 hàng
        case_montage = np.vstack([row1, row2])
        out_case_path = os.path.join(ARTIFACT_DIR, f"diagnostic_sharpener_{fname.split('.')[0]}.png")
        cv2.imwrite(out_case_path, case_montage)
        print(f"  -> Da ghi: {out_case_path}")
        output_montages.append(out_case_path)
        
    print("\n[✓] Hoan tat toan bo kiem chung truc quan.")

if __name__ == '__main__':
    main()
