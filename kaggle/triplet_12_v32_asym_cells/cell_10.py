# [CELL 10]: Vòng Lặp Huấn Luyện & Thẩm Định Hoàn Chỉnh V3.2 (25 Epochs, Multi-GPU DDP Engine, Zero Leak, Target < 10 Min/Epoch)
import os
import sys
import time
import subprocess
import torch
from tqdm.auto import tqdm

WORKING_DIR = globals().get('WORKING_DIR', '/kaggle/working' if os.path.exists('/kaggle') else './workspace')
CKPT_DIR = globals().get('CKPT_DIR', os.path.join(WORKING_DIR, 'checkpoints_v3.2'))
CACHE_DIR = globals().get('CACHE_DIR', os.path.join(WORKING_DIR, 'dataset_cache_v3.2'))
STAGE_DIR = globals().get('DATASET_ROOT', globals().get('LOCAL_SSD_STAGING_DIR', '/tmp/sakuga_ssd_v32'))
NUM_GPUS = globals().get('NUM_GPUS', torch.cuda.device_count())
device = globals().get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

print("=" * 80)
print(f"🚀 KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN V3.2 (PROGRESSIVE STROKE PRESERVATION)")
print(f"[*] Số lượng GPU phát hiện: {NUM_GPUS} | Nền tảng: {sys.platform}")
print(f"[*] Working Dir: {WORKING_DIR} | Stage Dir: {STAGE_DIR}")
print("=" * 80)
sys.stdout.flush()

DDP_WORKER_PATH = os.path.join(WORKING_DIR, 'ddp_worker_v32.py')

# __DDP_WORKER_INJECTION_POINT__
if not os.path.exists(DDP_WORKER_PATH):
    cand_paths = [
        'ddp_worker_v32.py',
        'kaggle/v32_cells/ddp_worker_v32.py',
        os.path.join(os.path.dirname(__file__) if '__file__' in globals() else '.', 'ddp_worker_v32.py')
    ]
    for cp in cand_paths:
        if os.path.exists(cp):
            import shutil
            shutil.copyfile(cp, DDP_WORKER_PATH)
            break

# Lựa chọn chế độ thực thi: DDP (Multi-GPU Linux) hoặc Single-GPU
if NUM_GPUS >= 2 and sys.platform != 'win32':
    print(f"[*] Khởi chạy DistributedDataParallel (DDP) trên {NUM_GPUS} GPUs qua torchrun...")
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--standalone',
        '--nnodes=1',
        f'--nproc_per_node={NUM_GPUS}',
        DDP_WORKER_PATH,
        f'--working-dir={WORKING_DIR}',
        f'--ckpt-dir={CKPT_DIR}',
        f'--stage-dir={STAGE_DIR}',
        f'--total-epochs=25',
        f'--per-gpu-batch=4',
        f'--grad-accum-steps=2',
        '--pbar-pipe'
    ]
else:
    print(f"[*] Khởi chạy chế độ Single-GPU tối ưu hóa qua subprocess unbuffered...")
    cmd = [
        sys.executable,
        DDP_WORKER_PATH,
        '--single-gpu',
        f'--working-dir={WORKING_DIR}',
        f'--ckpt-dir={CKPT_DIR}',
        f'--stage-dir={STAGE_DIR}',
        f'--total-epochs=25',
        f'--per-gpu-batch=8',
        f'--grad-accum-steps=2',
        '--pbar-pipe'
    ]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

t_train_start = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

train_pbar = None
val_pbar = None
curr_train_ep = None
curr_val_ep = None

def close_train_pbar():
    global train_pbar, curr_train_ep
    if train_pbar is not None:
        try:
            train_pbar.close()
        except Exception:
            pass
        train_pbar = None
        curr_train_ep = None

def close_val_pbar():
    global val_pbar, curr_val_ep
    if val_pbar is not None:
        try:
            val_pbar.close()
        except Exception:
            pass
        val_pbar = None
        curr_val_ep = None

for raw_line in proc.stdout:
    line = raw_line.rstrip('\r\n')
    if not line:
        continue

    if line.startswith('__PBAR_TRAIN__:'):
        close_val_pbar()
        payload = line[len('__PBAR_TRAIN__:'):]
        parts = dict(kv.split('=', 1) for kv in payload.split('|') if '=' in kv)
        try:
            ep = int(parts.get('ep', 1))
            total_ep = int(parts.get('total_ep', 25))
            step = int(parts.get('step', 0))
            total_steps = int(parts.get('total', 1))
            loss = float(parts.get('loss', 0.0))
            dens = float(parts.get('dens', 0.0))
            lr = parts.get('lr', '')

            if train_pbar is None or curr_train_ep != ep:
                close_train_pbar()
                curr_train_ep = ep
                desc = f"Train Ep {ep:02d}/{total_ep:02d}" + (f" [LR={lr}]" if lr else "")
                train_pbar = tqdm(total=total_steps, desc=desc, unit="batch", leave=False, dynamic_ncols=True)

            diff = step - train_pbar.n
            if diff > 0:
                train_pbar.update(diff)
            train_pbar.set_postfix({'loss': f"{loss:.4f}", 'dens': f"{dens:.3f}"})
        except Exception:
            pass

    elif line.startswith('__PBAR_VAL__:'):
        close_train_pbar()
        payload = line[len('__PBAR_VAL__:'):]
        parts = dict(kv.split('=', 1) for kv in payload.split('|') if '=' in kv)
        try:
            ep = int(parts.get('ep', 1))
            total_ep = int(parts.get('total_ep', 25))
            step = int(parts.get('step', 0))
            total_steps = int(parts.get('total', 1))

            if val_pbar is None or curr_val_ep != ep:
                close_val_pbar()
                curr_val_ep = ep
                desc = f"Val Ep {ep:02d}/{total_ep:02d}"
                val_pbar = tqdm(total=total_steps, desc=desc, unit="batch", leave=False, dynamic_ncols=True)

            diff = step - val_pbar.n
            if diff > 0:
                val_pbar.update(diff)
        except Exception:
            pass

    elif line.startswith('__PBAR_TRAIN_DONE__:'):
        close_train_pbar()

    elif line.startswith('__PBAR_VAL_DONE__:'):
        close_val_pbar()

    else:
        if '📌 HOÀN TẤT EPOCH' in line or 'BẮT ĐẦU VÒNG LẶP' in line:
            close_train_pbar()
            close_val_pbar()
        if train_pbar is not None or val_pbar is not None:
            tqdm.write(line)
        else:
            print(line, flush=True)

close_train_pbar()
close_val_pbar()
proc.wait()

if proc.returncode == 42:
    print("\n⏸️ HUẤN LUYỆN TẠM DỪNG DO WATCHDOG RAM. Kernel có thể restart để train tiếp.")
elif proc.returncode != 0:
    raise RuntimeError(f"Tiến trình huấn luyện kết thúc với lỗi (Mã trả về: {proc.returncode})!")
else:
    print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN TOÀN BỘ 25 EPOCHS TRONG {(time.time() - t_train_start)/60:.1f} PHÚT!")

# Đồng bộ lại trọng số tốt nhất vào mô hình base_model của kernel để các cell 11-13 chạy liền mạch
best_ckpt_p = os.path.join(CKPT_DIR, 'model_tps_v3.2_best.pt')
if not os.path.exists(best_ckpt_p):
    best_ckpt_p = os.path.join(CKPT_DIR, 'model_tps_v3.2_best_progress.pt')
if os.path.exists(best_ckpt_p) and 'base_model' in globals():
    sd = torch.load(best_ckpt_p, map_location=device, weights_only=False)
    base_model.load_state_dict(sd, strict=False)
    print(f"[✓] Đã nạp thành công checkpoint kỷ lục {best_ckpt_p} vào base_model của notebook.")
sys.stdout.flush()
