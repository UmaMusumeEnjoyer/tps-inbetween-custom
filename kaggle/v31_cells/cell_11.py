# [CELL 11]: Trực Quan Hóa Tiến Trình Hội Tụ & Chỉ Số Huấn Luyện V3.1 (6 Biểu Đồ Phân Tích)
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_training_metrics_convergence_v31(history_csv=None, save_dir=None):
    if history_csv is None: history_csv = os.path.join(WORKING_DIR, 'training_history_v3.1.csv')
    if save_dir is None: save_dir = WORKING_DIR
    if not os.path.exists(history_csv):
        print(f"[!] Không tìm thấy file lịch sử {history_csv} để vẽ biểu đồ.")
        return

    df = pd.read_csv(history_csv)
    if len(df) == 0: return

    epochs = df['epoch'].values
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=150)
    fig.suptitle('TPS-INBETWEEN V3.1 TRAINING CONVERGENCE & ANALYTICS (NATIVE 540P & FLAT MINIMA RETENTION)', fontsize=16, fontweight='bold')

    # 1. Loss Convergence
    ax = axes[0, 0]
    ax.plot(epochs, df['train_loss'], 'b-o', label='Train Total Loss', linewidth=2)
    ax.plot(epochs, df['val_l1'], 'r--s', label='Val L1 Loss', linewidth=2)
    ax.set_title('Loss Convergence (Train Total & Val L1)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # 2. SSIM & PSNR
    ax = axes[0, 1]
    ax.plot(epochs, df['val_ssim'], 'g-^', label='Val SSIM', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(epochs, df['val_psnr'], 'm--d', label='Val PSNR (dB)', linewidth=2)
    ax.set_title('Structural Similarity (SSIM) & PSNR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM', color='g')
    ax2.set_ylabel('PSNR (dB)', color='m')
    ax.grid(True)

    # 3. Stroke Recall & F1
    ax = axes[0, 2]
    ax.plot(epochs, df['stroke_recall'] * 100, 'c-o', label='Stroke Recall (%)', linewidth=2)
    ax.plot(epochs, df['stroke_f1'] * 100, 'y--v', label='Stroke F1 (%)', linewidth=2)
    ax.axhline(y=90.5, color='red', linestyle=':', label='Hard Gate Recall (90.5%)')
    ax.set_title('Stroke Detail Retention (Recall & F1)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage (%)')
    ax.legend()
    ax.grid(True)

    # 4. Stroke Density Ratio (Gold Standard 1.10x)
    ax = axes[1, 0]
    ax.plot(epochs, df['density_ratio'], 'k-o', label='Density Ratio', linewidth=2)
    ax.axhspan(1.06, 1.14, color='green', alpha=0.20, label='Gold Visual Band [1.06, 1.14]')
    ax.axhline(y=1.10, color='green', linestyle='--', label='Target Gold 1.10x')
    ax.axhline(y=1.02, color='red', linestyle=':', label='Gate Min 1.02x')
    ax.axhline(y=1.18, color='red', linestyle=':', label='Gate Max 1.18x')
    ax.set_title('Stroke Density Ratio (Gold Standard 1.10x)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Density Ratio')
    ax.legend(loc='upper right')
    ax.grid(True)

    # 5. Artifact & Ghost Stroke Suppression
    ax = axes[1, 1]
    ax.plot(epochs, df['ghost_pixels'], 'orange', marker='o', label='Effective Ghost Pixels', linewidth=2)
    ax.plot(epochs, df['dark_ghost_pixels'], 'darkred', linestyle='--', marker='s', label='Dark Ghost Pixels', linewidth=2)
    ax.set_title('Artifact & Ghost Stroke Suppression')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pixel Count')
    ax.legend()
    ax.grid(True)

    # 6. Training Time Per Epoch (< 10 min Target) & RAM Stability
    ax = axes[1, 2]
    if 'train_time_minutes' in df.columns:
        ax.plot(epochs, df['train_time_minutes'], 'purple', marker='s', linewidth=2.5, label='Train Time/Epoch (Min)')
        ax.axhline(y=10.0, color='red', linestyle='--', label='Max Allowed (10 Min/Epoch)')
        ax.set_ylabel('Minutes', color='purple')
    elif 'time_minutes' in df.columns:
        ax.plot(epochs, df['time_minutes'], 'purple', marker='s', linewidth=2.5, label='Total Epoch Time (Min)')
        ax.axhline(y=10.0, color='red', linestyle='--', label='Target Limit (10 Min)')
        ax.set_ylabel('Minutes', color='purple')
    
    if 'host_ram_used_gb' in df.columns:
        ax3 = ax.twinx()
        ax3.plot(epochs, df['host_ram_used_gb'], 'gray', linestyle=':', label='RAM Working Set (GB)')
        ax3.set_ylabel('RAM Used (GB)', color='gray')
        ax3.set_ylim(0, 32)

    ax.set_title('Training Speed (< 10 Min/Epoch) & RAM Stability')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    plot_p = os.path.join(save_dir, 'training_metrics_convergence_v31.png')
    plt.savefig(plot_p, bbox_inches='tight')
    plt.show()
    print(f"[✓] Đã xuất biểu đồ hội tụ V3.1 tại: {plot_p}")

plot_training_metrics_convergence_v31()
