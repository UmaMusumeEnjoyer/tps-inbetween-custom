import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import warnings
from torch.nn import functional as F

warnings.filterwarnings("ignore")

# Đảm bảo thư mục gốc của SAIN nằm trong sys.path
SAIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SAIN_ROOT not in sys.path:
    sys.path.insert(0, SAIN_ROOT)

# Hỗ trợ hiển thị ký tự Unicode tiếng Việt trên Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import config
from model.SAIN import SAIN
from model.warplayer import warp


def parse_time_steps(times_args, num_frames):
    """
    Phân tích danh sách các mốc thời gian t từ tham số truyền vào CLI.
    Hỗ trợ cả danh sách cách nhau bởi dấu cách hoặc dấu phẩy.
    """
    t_list = []
    if times_args is not None and len(times_args) > 0:
        for item in times_args:
            parts = str(item).replace(',', ' ').split()
            for p in parts:
                try:
                    val = float(p)
                    if 0.0 <= val <= 1.0:
                        t_list.append(val)
                    else:
                        print(f"[Cảnh báo] Mốc thời gian t={val} nằm ngoài khoảng [0.0, 1.0], bỏ qua.", flush=True)
                except ValueError:
                    print(f"[Cảnh báo] Không thể đọc mốc thời gian '{p}', bỏ qua.", flush=True)
    elif num_frames is not None and num_frames > 0:
        step = 1.0 / (num_frames + 1)
        t_list = [(i + 1) * step for i in range(num_frames)]
    else:
        t_list = [0.5]

    t_list = sorted(list(set(t_list)))
    return t_list


def format_inbetween_filename(prefix_1, t, digits=3):
    """
    Định dạng tên file theo quy chuẩn: {prefix_1}.{xxx}.png
    Ví dụ: prefix_1 = 1, t = 0.50  -> 1.500.png
           prefix_1 = 1, t = 0.20  -> 1.200.png
    """
    multiplier = 10 ** digits
    t_part = int(round(t * multiplier))
    return f"{prefix_1}.{t_part:0{digits}d}.png"


def extract_matched_keypoints(img0_bgr, img1_bgr):
    """
    Trích xuất các cặp điểm tương ứng (keypoint matches) giữa 2 frame
    sử dụng SIFT + Ratio Test + RANSAC.
    """
    gray0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m_tuple in matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    if mask is not None:
        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]

    return pts0, pts1


def load_sain_model(checkpoint_path, device):
    """
    Khởi tạo mô hình SAIN và nạp checkpoint.
    """
    args, _ = config.get_args()
    args.device = device.type
    args.resume_flownet = False
    args.crop_size = 192

    model = SAIN(args).to(device)
    model.eval()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint SAIN tại: {checkpoint_path}")

    print(f"[*] Nạp checkpoint từ: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)
    return model


def run_interpolation(args):
    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
    print(f"[*] Thiết bị sử dụng: {device}", flush=True)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(args.frame0):
        raise FileNotFoundError(f"Không tìm thấy ảnh frame0 tại: {args.frame0}")
    if not os.path.exists(args.frame1):
        raise FileNotFoundError(f"Không tìm thấy ảnh frame1 tại: {args.frame1}")

    ori_img0 = cv2.imread(args.frame0)
    ori_img1 = cv2.imread(args.frame1)

    if ori_img0 is None:
        raise ValueError(f"Không thể đọc file ảnh Frame 1 tại: {args.frame0}")
    if ori_img1 is None:
        raise ValueError(f"Không thể đọc file ảnh Frame 2 tại: {args.frame1}")

    if ori_img0.shape != ori_img1.shape:
        raise ValueError(f"Kích thước 2 ảnh không khớp! Frame0: {ori_img0.shape}, Frame1: {ori_img1.shape}")

    h, w, _ = ori_img0.shape
    print(f"[*] Kích thước ảnh gốc: {w}x{h} (WxH)", flush=True)

    # Phân tích danh sách mốc thời gian t
    t_list = parse_time_steps(args.times, args.num_frames)
    print(f"[*] Danh sách mốc thời gian t cần nội suy ({len(t_list)} frames): {t_list}", flush=True)

    # Lưu Frame 1 và Frame 2 gốc
    frame1_name = f"{args.prefix_1}.{'0' * args.digits}.png"
    frame2_name = f"{args.prefix_2}.{'0' * args.digits}.png"
    frame1_path = os.path.join(save_dir, frame1_name)
    frame2_path = os.path.join(save_dir, frame2_name)

    cv2.imwrite(frame1_path, ori_img0)
    cv2.imwrite(frame2_path, ori_img1)
    print(f"[+] Đã lưu Frame 1 gốc -> {frame1_path}", flush=True)
    print(f"[+] Đã lưu Frame 2 gốc -> {frame2_path}", flush=True)

    # Nạp mô hình
    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(SAIN_ROOT, 'ckp', 'checkpoints', 'model_best.pth')
    model = load_sain_model(ckpt_path, device)

    # Trích xuất keypoints tương ứng giữa 2 frame (Sketch-Aware Guidance)
    pts0, pts1 = extract_matched_keypoints(ori_img0, ori_img1)
    print(f"[*] Số điểm nét vẽ (keypoint matches) tìm được: {len(pts0)}", flush=True)

    # Chuẩn bị Tensor (padding bội số 64 cho Swin Transformer)
    divisor = 64
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    img0_t = torch.tensor(cv2.cvtColor(ori_img0, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
    img1_t = torch.tensor(cv2.cvtColor(ori_img1, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0

    if pad_h > 0 or pad_w > 0:
        img0_t = F.pad(img0_t, (0, pad_w, 0, pad_h), mode='replicate')
        img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h), mode='replicate')

    img0_t = img0_t.to(device)
    img1_t = img1_t.to(device)
    _, _, pH, pW = img0_t.shape

    # Tính toán optical flow cơ sở qua FlowNet + RefineNet của SAIN
    with torch.no_grad():
        imgs_pair = torch.cat((img0_t, img1_t), 1)
        base_flow, _ = model.flownet(imgs_pair)
        base_flow, _, _ = model.refinenet(img0_t, img1_t, base_flow)

    print(f"[*] Bắt đầu quá trình nội suy các khung hình trung gian (In-between frames)...", flush=True)
    for idx, t in enumerate(t_list):
        if t == 0.0 or t == 1.0:
            continue

        # 1. Tạo bản đồ điểm nét vẽ nội suy (Point map tại thời điểm t)
        point_map = np.ones((pH, pW, 1), dtype=np.uint8) * 255
        if len(pts0) > 0:
            for p0, p1 in zip(pts0, pts1):
                xt = int(round((1.0 - t) * p0[0] + t * p1[0]))
                yt = int(round((1.0 - t) * p0[1] + t * p1[1]))
                if 0 <= xt < pW and 0 <= yt < pH:
                    cv2.circle(point_map, (xt, yt), radius=0, color=0, thickness=1)

        point_t = (torch.tensor(point_map.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0).to(device)

        # 2. Điều hướng trường chuyển động (Scaled Motion Field) theo mốc thời gian t
        # SAIN ước tính dòng chuyển động đến mốc giữa; ta co giãn theo t tương đối
        flow0t = base_flow[:, :2] * (2.0 * t)
        flow1t = base_flow[:, 2:] * (2.0 * (1.0 - t))

        with torch.no_grad():
            warped_img0 = warp(img0_t, flow0t)
            warped_img1 = warp(img1_t, flow1t)

            points_feat = model.points_fuse(point_t)

            warped_img2 = warp(img0_t, flow0t)
            warped_img3 = warp(img1_t, flow1t)

            fused_img0 = model.fuse_block1(torch.cat([warped_img0, warped_img2], dim=1))
            fused_img1 = model.fuse_block2(torch.cat([warped_img1, warped_img3], dim=1))

            i0_output = model.cross_tran(points_feat, fused_img0)
            res0 = torch.sigmoid(i0_output)

            i1_output = model.cross_tran(points_feat, fused_img1)
            res1 = torch.sigmoid(i1_output)

            x = model.fuse_block(torch.cat([warped_img0, warped_img1, points_feat, warped_img2, warped_img3], dim=1))
            refine_output = model.transformer(x)
            res = torch.sigmoid(refine_output)

            pred = model.final_fuse_block(torch.cat([res0, res1, res], dim=1))
            pred = torch.sigmoid(pred)

        # Cắt về kích thước gốc ban đầu (crop padding)
        pred_crop = pred[:, :, :h, :w]
        pred_np = (pred_crop[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

        out_filename = format_inbetween_filename(args.prefix_1, t, digits=args.digits)
        out_filepath = os.path.join(save_dir, out_filename)
        cv2.imwrite(out_filepath, pred_bgr)
        print(f"    [{idx+1}/{len(t_list)}] Hoàn thành t={t:.4f} -> {out_filename}", flush=True)

    print(f"\n[✓] Hoàn tất toàn bộ quá trình! Tất cả kết quả đã được lưu tại: {save_dir}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAIN (Sketch-Aware Interpolation Network) Custom Arbitrary Time Demo")
    parser.add_argument("--frame0", type=str, required=True, help="Đường dẫn đến file ảnh Frame 1")
    parser.add_argument("--frame1", type=str, required=True, help="Đường dẫn đến file ảnh Frame 2")
    parser.add_argument("--times", type=str, nargs="*", default=None, help="Danh sách mốc thời gian t bất kỳ trong khoảng (0.0, 1.0)")
    parser.add_argument("--num_frames", type=int, default=None, help="Số lượng frame in-between muốn chia đều")
    parser.add_argument("--save_dir", type=str, default="./demo/output", help="Thư mục lưu ảnh kết quả")
    parser.add_argument("--prefix_1", type=str, default="1", help="Tiền tố tên file Frame 1 (mặc định '1')")
    parser.add_argument("--prefix_2", type=str, default="2", help="Tiền tố tên file Frame 2 (mặc định '2')")
    parser.add_argument("--digits", type=int, default=3, help="Số chữ số thập phân cho tên file (mặc định 3)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn file checkpoint model_best.pth")
    parser.add_argument("--cpu", action="store_true", help="Bắt buộc chạy trên CPU")
    args = parser.parse_args()

    run_interpolation(args)
