import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import json
import yaml
from easydict import EasyDict
from torch.nn import functional as F

# Đảm bảo thư mục gốc của AnimeInbet nằm trong sys.path
ANIMEINBET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ANIMEINBET_ROOT not in sys.path:
    sys.path.insert(0, ANIMEINBET_ROOT)

# Hỗ trợ hiển thị ký tự Unicode tiếng Việt trên Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

import models
from sknetwork.embedding import Spectral


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
                        print(f"[Cảnh báo] Mốc thời gian t={val} nằm ngoài khoảng [0.0, 1.0], bỏ qua.")
                except ValueError:
                    print(f"[Cảnh báo] Không thể đọc mốc thời gian '{p}', bỏ qua.")
    elif num_frames is not None and num_frames > 0:
        # Tự động chia đều các mốc t trong khoảng (0, 1)
        step = 1.0 / (num_frames + 1)
        t_list = [(i + 1) * step for i in range(num_frames)]
    else:
        # Mặc định lấy mốc chính giữa t = 0.5
        t_list = [0.5]

    # Loại bỏ trùng lặp và sắp xếp tăng dần
    t_list = sorted(list(set(t_list)))
    return t_list


def format_inbetween_filename(prefix_1, t, digits=3):
    """
    Định dạng tên file theo quy chuẩn: {prefix_1}.{xxx}.png
    Ví dụ: prefix_1 = 1, t = 0.50  -> 1.500.png
           prefix_1 = 1, t = 0.125 -> 1.125.png
           prefix_1 = 1, t = 0.05  -> 1.050.png
    """
    multiplier = 10 ** digits
    t_part = int(round(t * multiplier))
    return f"{prefix_1}.{t_part:0{digits}d}.png"


def read_json_graph(file_path):
    """
    Đọc dữ liệu đỉnh và đồ thị từ tệp nhãn JSON theo chuẩn MixamoLine240 (ML240).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        vertex2d = np.array(data['vertex location'], dtype=np.float32)
        topology = data['connection']
        index = np.array(data.get('original index', list(range(len(vertex2d)))))
    return vertex2d, topology, index


def extract_lineart_graph(img_input, max_kpts=2000, approx_epsilon=1.0, thresh=220, min_len=12):
    """
    Tự động trích xuất vector graph (đỉnh 2D và cấu trúc liên kết topo) từ ảnh nét vẽ raster
    bằng thuật toán binarization + contour polygonal approximation.
    - Hỗ trợ ảnh màu Douga (dùng min channel across RGB để phát hiện nét màu đỏ, cam, xanh).
    - Lọc các contour quá nhỏ (nhiễu scan/bụi giấy).
    - Sắp xếp contour theo độ dài giảm dần để ưu tiên giữ lại các nét cấu trúc chính của nhân vật.
    """
    if len(img_input.shape) == 3 and img_input.shape[2] == 3:
        # Ảnh BGR: lấy giá trị min của 3 kênh để các nét màu (đỏ, xanh, cam) trở nên đậm
        detect_ch = np.min(img_input, axis=2)
        h, w = img_input.shape[:2]
    else:
        detect_ch = img_input
        h, w = img_input.shape[:2]

    # Xác định nền trắng nét đen hay nền đen nét trắng
    if detect_ch.mean() > 127:
        _, binary = cv2.threshold(detect_ch, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(detect_ch, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc bỏ nhiễu hạt bụi scan
    valid_cnts = [c for c in contours if cv2.arcLength(c, True) >= min_len]
    # Sắp xếp các đường contour dài nhất trước (nét cơ thể, khuôn mặt, trang phục)
    valid_cnts = sorted(valid_cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)

    vertices = []
    topo = []
    current_idx = 0

    for cnt in valid_cnts:
        approx = cv2.approxPolyDP(cnt, epsilon=approx_epsilon, closed=True)
        if len(approx) < 2:
            continue
        pts = approx.reshape(-1, 2)
        n_pts = len(pts)
        if current_idx + n_pts > max_kpts:
            # Dừng lại khi đạt ngưỡng max_kpts để tránh cắt đứt giữa chừng một contour
            break
        for i in range(n_pts):
            vertices.append(pts[i].tolist())
            prev_i = (i - 1) % n_pts
            next_i = (i + 1) % n_pts
            topo.append([current_idx + prev_i, current_idx + next_i])
        current_idx += n_pts

    # Nếu không tìm thấy đường nét nào, tạo một số điểm bao quanh khung
    if len(vertices) < 2:
        vertices = [[10.0, 10.0], [w - 10.0, 10.0], [w - 10.0, h - 10.0], [10.0, h - 10.0]]
        topo = [[1, 3], [0, 2], [1, 3], [0, 2]]

    vertices = np.array(vertices, dtype=np.float32)
    vertices[:, 0] = np.clip(vertices[:, 0], 0, w - 1)
    vertices[:, 1] = np.clip(vertices[:, 1], 0, h - 1)

    return vertices, topo


def compute_spectral_and_adj(vertices, topo, spec_dim=64):
    """
    Tính ma trận kề có self-loop (theo chuẩn datasets/ml_seq.py của AnimeInbet)
    và sinh spectral embedding 64 chiều từ cấu trúc đồ thị.
    """
    n_v = len(vertices)
    # Bổ sung self-loop cho từng đỉnh trước khi tạo ma trận kề
    topo_loops = [list(nbs) + [i] for i, nbs in enumerate(topo)]
    try:
        adj = sknetwork.data.from_adjacency_list(topo_loops, matrix_only=True, reindex=False).toarray()
    except Exception:
        adj = np.eye(n_v, dtype=np.float32)

    spectral = Spectral(spec_dim, normalized=False)
    try:
        spec = np.abs(spectral.fit_transform(adj))
        if spec.shape[1] < spec_dim:
            pad_cols = spec_dim - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_cols)), mode='constant')
        elif spec.shape[1] > spec_dim:
            spec = spec[:, :spec_dim]
    except Exception:
        spec = np.zeros((n_v, spec_dim), dtype=np.float32)

    return spec, adj


def load_animeinbet_model(checkpoint_path=None, model_name='InbetweenerTM', config_path=None, device=torch.device('cuda')):
    """
    Khởi tạo mô hình AnimeInbet và nạp trọng số từ checkpoint (nếu có).
    """
    # 1. Cấu hình mặc định cho mô hình
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
            model_cfg = EasyDict(cfg_dict['model'])
    else:
        # Thử tìm config mặc định trong thư mục configs
        default_cfg_file = os.path.join(ANIMEINBET_ROOT, 'configs', 'cr_inbetweener_full.yaml')
        if os.path.exists(default_cfg_file):
            with open(default_cfg_file, 'r', encoding='utf-8') as f:
                cfg_dict = yaml.safe_load(f)
                model_cfg = EasyDict(cfg_dict['model'])
        else:
            model_cfg = EasyDict({
                'name': model_name,
                'corr_model': {
                    'descriptor_dim': 128,
                    'keypoint_encoder': [32, 64, 128],
                    'GNN_layer_num': 12,
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2
                },
                'pos_weight': 0.2
            })

    if hasattr(model_cfg, 'name') and model_name is None:
        model_name = model_cfg.name

    # 2. Khởi tạo đối tượng Model
    if hasattr(models, model_name):
        model_class = getattr(models, model_name)
    elif hasattr(models, 'InbetweenerTM'):
        model_class = getattr(models, 'InbetweenerTM')
    else:
        raise ValueError(f"Không tìm thấy lớp kiến trúc mô hình: {model_name}")

    print(f"[*] Khởi tạo kiến trúc mô hình: {model_class.__name__}")
    model = model_class(model_cfg)
    model = model.to(device).eval()

    # 3. Nạp trọng số checkpoint nếu được cung cấp
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Hỗ trợ tương thích PyTorch 2.6+ (tránh lỗi UnpicklingError do easydict)
        try:
            import easydict
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([easydict.EasyDict])
        except Exception:
            pass

        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Loại bỏ tiền tố 'module.' nếu checkpoint từ DataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict, strict=False)
        print("[+] Đã nạp checkpoint thành công.")
    else:
        if checkpoint_path:
            print(f"[!] Cảnh báo: Không tìm thấy file checkpoint tại: {checkpoint_path}. Sử dụng mô hình khởi tạo mặc định.")
        else:
            print("[*] Không chỉ định checkpoint, chạy với cấu hình trọng số hiện tại.")

    return model


def run_interpolation(args):
    # 1. Xác định thiết bị tính toán (GPU / CPU)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # 2. Tạo thư mục lưu kết quả đầu ra
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 3. Đọc và kiểm tra tính hợp lệ của 2 ảnh đầu vào
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
        raise ValueError(
            f"Kích thước 2 ảnh không khớp! Frame0: {ori_img0.shape}, Frame1: {ori_img1.shape}. "
            f"Vui lòng đảm bảo 2 ảnh có cùng chiều cao, chiều rộng và số kênh màu."
        )

    h, w = ori_img0.shape[:2]
    print(f"[*] Đã tải 2 ảnh đầu vào thành công. Kích thước gốc: {w}x{h} (WxH)")
    print(f"[*] Thiết bị sử dụng: {device}")

    # 4. Phân tích chuỗi mốc thời gian t
    t_list = parse_time_steps(args.times, args.num_frames)
    inbetween_t_list = [t for t in t_list if 0.0 < t < 1.0]
    print(f"[*] Danh sách mốc thời gian t cần nội suy ({len(inbetween_t_list)} frames): {inbetween_t_list}")

    # 5. Copy và lưu lại Frame 1 & Frame 2 vào thư mục đầu ra
    frame1_name = f"{args.prefix_1}.{'0' * args.digits}.png"
    frame2_name = f"{args.prefix_2}.{'0' * args.digits}.png"

    frame1_path = os.path.join(save_dir, frame1_name)
    frame2_path = os.path.join(save_dir, frame2_name)

    cv2.imwrite(frame1_path, ori_img0)
    cv2.imwrite(frame2_path, ori_img1)
    print(f"[+] Đã lưu Frame 1 gốc -> {frame1_path}")
    print(f"[+] Đã lưu Frame 2 gốc -> {frame2_path}")

    if len(inbetween_t_list) == 0:
        print("[!] Không có mốc thời gian in-between nào nằm trong khoảng (0.0, 1.0). Hoàn tất.")
        return

    # 6. Chuẩn bị ảnh xám
    if len(ori_img0.shape) == 3 and ori_img0.shape[2] == 3:
        gray0 = cv2.cvtColor(ori_img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)
    else:
        gray0 = ori_img0
        gray1 = ori_img1

    # 7. Đọc hoặc trích xuất đồ thị hình học (Keypoints, Topo, Spectral Embedding, Adjacency Matrix)
    print("====== [1/2] Chuẩn bị dữ liệu đồ thị đường nét (Graph Representation) ======")
    label0_path = args.label0
    label1_path = args.label1

    # Tự động tìm kiếm file .json tương ứng nếu không chỉ định rõ
    if label0_path is None:
        auto_json0 = os.path.splitext(args.frame0)[0] + '.json'
        if os.path.exists(auto_json0):
            label0_path = auto_json0
    if label1_path is None:
        auto_json1 = os.path.splitext(args.frame1)[0] + '.json'
        if os.path.exists(auto_json1):
            label1_path = auto_json1

    if label0_path and os.path.exists(label0_path):
        print(f"[*] Nạp nhãn vector graph Frame 0 từ: {label0_path}")
        v0, t0, _ = read_json_graph(label0_path)
    else:
        print(f"[*] Tự động trích xuất vector graph từ ảnh Frame 0...")
        v0, t0 = extract_lineart_graph(
            ori_img0,
            max_kpts=args.max_kpts,
            approx_epsilon=args.approx_epsilon,
            thresh=args.thresh,
            min_len=args.min_contour_len
        )

    if label1_path and os.path.exists(label1_path):
        print(f"[*] Nạp nhãn vector graph Frame 1 từ: {label1_path}")
        v1, t1, _ = read_json_graph(label1_path)
    else:
        print(f"[*] Tự động trích xuất vector graph từ ảnh Frame 1...")
        v1, t1 = extract_lineart_graph(
            ori_img1,
            max_kpts=args.max_kpts,
            approx_epsilon=args.approx_epsilon,
            thresh=args.thresh,
            min_len=args.min_contour_len
        )

    # Đảm bảo số lượng đỉnh không vượt quá max_kpts
    if len(v0) > args.max_kpts:
        v0 = v0[:args.max_kpts]
        t0 = t0[:args.max_kpts]
        for i in range(len(t0)):
            t0[i] = [nb for nb in t0[i] if nb < args.max_kpts]
    if len(v1) > args.max_kpts:
        v1 = v1[:args.max_kpts]
        t1 = t1[:args.max_kpts]
        for i in range(len(t1)):
            t1[i] = [nb for nb in t1[i] if nb < args.max_kpts]

    # Tính toán spectral embedding và ma trận kề chuẩn cho 2 đồ thị
    s0, adj0 = compute_spectral_and_adj(v0, t0, spec_dim=64)
    s1, adj1 = compute_spectral_and_adj(v1, t1, spec_dim=64)

    print(f"[*] Số đỉnh đồ thị Frame 0: {len(v0)} đỉnh | Frame 1: {len(v1)} đỉnh")

    # 8. Chuẩn bị tensor và xử lý padding nếu cần
    # Chuẩn hóa ảnh về khoảng [-1.0, 1.0] theo đúng pipeline của AnimeInbet
    if len(ori_img0.shape) == 2:
        img0_rgb = cv2.cvtColor(ori_img0, cv2.COLOR_GRAY2BGR)
        img1_rgb = cv2.cvtColor(ori_img1, cv2.COLOR_GRAY2BGR)
    else:
        img0_rgb = ori_img0
        img1_rgb = ori_img1

    tensor_img0 = torch.from_numpy(img0_rgb).permute(2, 0, 1).float().unsqueeze(0) * 2.0 / 255.0 - 1.0
    tensor_img1 = torch.from_numpy(img1_rgb).permute(2, 0, 1).float().unsqueeze(0) * 2.0 / 255.0 - 1.0

    # Padding kích thước ảnh lên bội số của 32 nếu cần
    divisor = 32
    pad_h = ((h - 1) // divisor + 1) * divisor
    pad_w = ((w - 1) // divisor + 1) * divisor
    pad_bottom = pad_h - h
    pad_right = pad_w - w

    if pad_bottom > 0 or pad_right > 0:
        tensor_img0 = F.pad(tensor_img0, (0, pad_right, 0, pad_bottom), mode="replicate")
        tensor_img1 = F.pad(tensor_img1, (0, pad_right, 0, pad_bottom), mode="replicate")

    tensor_img0 = tensor_img0.to(device)
    tensor_img1 = tensor_img1.to(device)

    # Đóng gói dữ liệu đầu vào cho AnimeInbet
    data = {
        'image0': tensor_img0,
        'image1': tensor_img1,
        'keypoints0': torch.from_numpy(v0).float().unsqueeze(0).to(device),
        'keypoints1': torch.from_numpy(v1).float().unsqueeze(0).to(device),
        'adj_mat0': torch.from_numpy(adj0).float().unsqueeze(0).to(device),
        'adj_mat1': torch.from_numpy(adj1).float().unsqueeze(0).to(device),
        'spec0': torch.from_numpy(s0).float().unsqueeze(0).to(device),
        'spec1': torch.from_numpy(s1).float().unsqueeze(0).to(device),
        'ms': torch.tensor([len(v0)], device=device),
        'ns': torch.tensor([len(v1)], device=device),
        'mask0': torch.ones(1, len(v0), device=device),
        'mask1': torch.ones(1, len(v1), device=device),
        'gen_vid': True
    }
    # Chỉ truyền topo vào model nếu bật cờ --topo_prune (mặc định tắt vì vòng lặp
    # co giãn topo trong InbetweenerTM.forward làm đứt gãy 85% nét vẽ của ảnh raster)
    if args.topo_prune:
        data['topo0'] = [t0]
        data['topo1'] = [t1]

    # 9. Tải và chạy mô hình AnimeInbet
    print("====== [2/2] Bắt đầu nội suy khung hình (AnimeInbet Inference) ======")
    checkpoint_path = getattr(args, 'checkpoint', None) or getattr(args, 'model_path', None)
    if checkpoint_path is None:
        # Thử tìm checkpoint mặc định tại experiments/inbetweener_full/ckpt/epoch_20.pt
        def_ckpt = os.path.join(ANIMEINBET_ROOT, 'experiments', 'inbetweener_full', 'ckpt', 'epoch_20.pt')
        if os.path.exists(def_ckpt):
            checkpoint_path = def_ckpt

    model = load_animeinbet_model(
        checkpoint_path=checkpoint_path,
        model_name=args.model_size,
        config_path=args.config,
        device=device
    )

    with torch.no_grad():
        out = model(data)

    r0 = out['r0'][0].cpu().numpy()
    r1 = out['r1'][0].cpu().numpy()
    vb0 = (out['vb0'][0].cpu().numpy() > args.vis_thresh)
    vb1 = (out['vb1'][0].cpu().numpy() > args.vis_thresh)

    # 10. Tổng hợp và kết xuất từng khung hình in-between tại mốc thời gian t
    # Xác định màu nền và màu nét vẽ theo ảnh gốc
    if gray0.mean() > 127:
        bg_val = 255
        line_color = (0, 0, 0)
    else:
        bg_val = 0
        line_color = (255, 255, 255)

    print(f"[*] Bắt đầu tổng hợp {len(inbetween_t_list)} khung hình trung gian...")
    for idx, t in enumerate(inbetween_t_list):
        # Biến dạng tọa độ đỉnh theo vector chuyển động nội suy với clip giới hạn khung hình
        v0_warp = np.clip(v0 + t * r0, 0, [w - 1, h - 1]).astype(int)
        v1_warp = np.clip(v1 + (1.0 - t) * r1, 0, [w - 1, h - 1]).astype(int)

        # Khởi tạo canvas vẽ ảnh kích thước gốc (h, w)
        canvas = np.full((h, w, 3), bg_val, dtype=np.uint8)

        # Vẽ các đoạn thẳng từ đồ thị Frame 0
        for node, nbs in enumerate(t0):
            for nb in nbs:
                if vb0[node] and vb0[nb] and nb < len(v0_warp) and node < len(v0_warp):
                    p1 = (int(v0_warp[node][0]), int(v0_warp[node][1]))
                    p2 = (int(v0_warp[nb][0]), int(v0_warp[nb][1]))
                    cv2.line(canvas, p1, p2, line_color, args.line_width, cv2.LINE_AA)

        # Vẽ các đoạn thẳng từ đồ thị Frame 1
        for node, nbs in enumerate(t1):
            for nb in nbs:
                if vb1[node] and vb1[nb] and nb < len(v1_warp) and node < len(v1_warp):
                    p1 = (int(v1_warp[node][0]), int(v1_warp[node][1]))
                    p2 = (int(v1_warp[nb][0]), int(v1_warp[nb][1]))
                    cv2.line(canvas, p1, p2, line_color, args.line_width, cv2.LINE_AA)

        # Định dạng và lưu file
        out_filename = format_inbetween_filename(args.prefix_1, t, digits=args.digits)
        out_filepath = os.path.join(save_dir, out_filename)
        cv2.imwrite(out_filepath, canvas)
        print(f"    [{idx+1}/{len(inbetween_t_list)}] Hoàn thành t={t:.4f} -> {out_filename}")

    print(f"\n[✓] Hoàn tất toàn bộ quá trình! Tất cả kết quả đã được lưu tại: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AnimeInbet Custom Arbitrary Time-step Animation Line Inbetweening Demo"
    )
    parser.add_argument(
        "--frame0", type=str, required=True,
        help="Đường dẫn đến file ảnh Frame 1 (I_0)"
    )
    parser.add_argument(
        "--frame1", type=str, required=True,
        help="Đường dẫn đến file ảnh Frame 2 (I_1)"
    )
    parser.add_argument(
        "--times", type=str, nargs="*", default=None,
        help="Danh sách mốc thời gian t bất kỳ trong khoảng (0.0, 1.0). Ví dụ: --times 0.25 0.5 0.75 hoặc --times 0.2,0.4,0.6"
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Số lượng khung hình chia đều nếu không chỉ định --times (Ví dụ: --num_frames 3)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./demo/output",
        help="Thư mục lưu các file ảnh kết quả (Mặc định: ./demo/output)"
    )
    parser.add_argument(
        "--prefix_1", type=int, default=1,
        help="Tiền tố đánh số cho Frame 1 (Mặc định: 1)"
    )
    parser.add_argument(
        "--prefix_2", type=int, default=2,
        help="Tiền tố đánh số cho Frame 2 (Mặc định: 2)"
    )
    parser.add_argument(
        "--digits", type=int, default=3,
        help="Số chữ số thập phân cho tên file (Mặc định: 3)"
    )
    parser.add_argument(
        "--model_size", type=str, default="InbetweenerTM",
        choices=["InbetweenerTM", "InbetweenerM2"],
        help="Kiến trúc / Kích thước mô hình (Mặc định: InbetweenerTM)"
    )
    parser.add_argument(
        "--checkpoint", "--model_path", dest="checkpoint", type=str, default=None,
        help="Đường dẫn đến file trọng số checkpoint (.pt) của AnimeInbet"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Đường dẫn đến file cấu hình yaml (Mặc định: configs/cr_inbetweener_full.yaml)"
    )
    parser.add_argument(
        "--label0", type=str, default=None,
        help="Đường dẫn đến file nhãn .json cho Frame 1 (tùy chọn, tự động trích xuất nếu không có)"
    )
    parser.add_argument(
        "--label1", type=str, default=None,
        help="Đường dẫn đến file nhãn .json cho Frame 2 (tùy chọn, tự động trích xuất nếu không có)"
    )
    parser.add_argument(
        "--line_width", type=int, default=2,
        help="Độ dày nét vẽ khi kết xuất in-between frames (Mặc định: 2)"
    )
    parser.add_argument(
        "--max_kpts", type=int, default=2000,
        help="Số lượng điểm đỉnh tối đa khi trích xuất đồ thị (Mặc định: 2000)"
    )
    parser.add_argument(
        "--thresh", type=int, default=220,
        help="Ngưỡng binarization (0-255) khi trích xuất nét vẽ (Mặc định: 220)"
    )
    parser.add_argument(
        "--approx_epsilon", type=float, default=1.0,
        help="Tham số epsilon xấp xỉ đa giác contour (Mặc định: 1.0, nhỏ hơn cho nét mịn hơn)"
    )
    parser.add_argument(
        "--min_contour_len", type=int, default=12,
        help="Chiều dài tối thiểu của contour để loại bỏ nhiễu hạt bụi scan (Mặc định: 12)"
    )
    parser.add_argument(
        "--vis_thresh", type=float, default=-0.2,
        help="Ngưỡng visibility logit để giữ nét vẽ trong khung hình kết xuất (Mặc định: -0.2)"
    )
    parser.add_argument(
        "--topo_prune", action="store_true",
        help="Bật kiểm tra độ co giãn topo trong forward model (Mặc định: Tắt để tránh triệt tiêu 85%% nét vẽ)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Bắt buộc chạy trên CPU (Mặc định: tự động dùng GPU CUDA nếu khả dụng)"
    )

    args = parser.parse_args()
    run_interpolation(args)
