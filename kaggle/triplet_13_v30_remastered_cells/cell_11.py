# [CELL 11]: Bảng Điều Khiển Hội Tụ Đa Chỉ Số Chuẩn Nhận Thức V3.0-Remastered R2
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

WORKING_DIR = globals().get('WORKING_DIR', '/kaggle/working' if os.path.exists('/kaggle') else './workspace')
history_path = os.path.join(WORKING_DIR, 'training_history_v3.0_remastered.csv')

if not os.path.exists(history_path):
    cands = glob.glob(os.path.join(WORKING_DIR, '**', 'training_history_v3.0_remastered.csv'), recursive=True)
    if cands:
        history_path = cands[0]

if not os.path.exists(history_path):
    print(f"[*] Chưa tìm thấy file lịch sử {history_path}. Bỏ qua vẽ dashboard.")
else:
    df_h = pd.read_csv(history_path)
    print(f"[✓] Đọc thành công {len(df_h)} epochs từ {history_path}!")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("TPS-Inbetween V3.0-Remastered R2 Perceptual & Topological Convergence Dashboard", fontsize=16, fontweight='bold')

    epochs = df_h['epoch'].values

    # Subplot 1: Train Loss & Val L1
    ax1 = axes[0, 0]
    ax1.plot(epochs, df_h['train_loss'], 'b-o', label='Train Composite Loss')
    ax1.plot(epochs, df_h['val_l1'], 'r--s', label='Val Charbonnier L1')
    ax1.set_title("Hội tụ Hàm Mất Mát (Loss Convergence)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: Stroke Recall, Skeleton Recall & Hair-Tip Recall
    ax2 = axes[0, 1]
    ax2.plot(epochs, df_h['stroke_recall'] * 100, 'g-^', label='Stroke Recall (%)')
    if 'skel_recall' in df_h.columns:
        ax2.plot(epochs, df_h['skel_recall'] * 100, 'm-d', label='Skeleton Recall (%)')
    if 'tip_recall' in df_h.columns:
        ax2.plot(epochs, df_h['tip_recall'] * 100, 'c--x', label='Hair-Tip Recall (%)')
    ax2.axhline(92.0, color='darkgreen', linestyle=':', label='Target Stroke Rec 92%')
    ax2.axhline(88.0, color='purple', linestyle=':', label='Target Skel Rec 88%')
    ax2.set_title("Bảo Toàn Hình Thái & Ngọn Tóc (Topology & Tip Recall)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Tỷ lệ (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Subplot 3: Stroke Density Ratio
    ax3 = axes[0, 2]
    ax3.plot(epochs, df_h['density_ratio'], 'teal', marker='o', label='Stroke Density Ratio')
    ax3.axhline(1.10, color='r', linestyle='-', label='Tâm Chuẩn 1.10x')
    ax3.axhspan(1.05, 1.18, alpha=0.15, color='green', label='Vùng Nghiệm Thu [1.05x, 1.18x]')
    ax3.set_title("Mật Độ Mực Nét Vẽ (Stroke Density Ratio)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Tỷ số Density")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Subplot 4: Weighted Chamfer Distance & S-EMD
    ax4 = axes[1, 0]
    if 'wcd' in df_h.columns:
        ax4.plot(epochs, df_h['wcd'], 'indigo', marker='o', label='Weighted Chamfer Dist (WCD)')
    if 'sliced_emd' in df_h.columns:
        ax4.plot(epochs, df_h['sliced_emd'], 'coral', marker='^', label='1D-Sliced EMD')
    ax4.axhline(0.00120, color='indigo', linestyle=':', label='Ref WCD 0.00120')
    ax4.set_title("Khoảng Cách Vật Lý & Phân Bố Tọa Độ (WCD & S-EMD)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Khoảng cách tọa độ")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Subplot 5: Ghost Pixels & Dark Ghosts
    ax5 = axes[1, 1]
    ax5.plot(epochs, df_h['ghost_pixels'], 'orange', marker='o', label='Far Ghosts (Threshold > 0.15)')
    if 'dark_ghost' in df_h.columns:
        ax5.plot(epochs, df_h['dark_ghost'], 'black', marker='x', label='Dark Ghosts (Threshold > 0.35)')
    ax5.axhline(6000.0, color='red', linestyle=':', label='Ngưỡng chuẩn <= 6,000 px')
    ax5.set_title("Kiểm Soát Bóng Ma (Ghost Artifacts)")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Pixels")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Subplot 6: Background Cleanliness & SSIM Reference
    ax6 = axes[1, 2]
    if 'bg_cleanliness' in df_h.columns:
        ax6.plot(epochs, df_h['bg_cleanliness'] * 100, 'navy', marker='o', label='Background Cleanliness (%)')
    if 'val_ssim' in df_h.columns:
        ax6_2 = ax6.twinx()
        ax6_2.plot(epochs, df_h['val_ssim'], 'purple', linestyle='--', marker='s', label='Val SSIM (Ref Only)')
        ax6_2.set_ylabel("SSIM", color='purple')
        ax6_2.legend(loc='lower right')
    ax6.axhline(99.5, color='navy', linestyle=':', label='Ngưỡng sạch >= 99.5%')
    ax6.set_title("Độ Sạch Nền & SSIM Tham Chiếu (Cleanliness & SSIM)")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Tỷ lệ Nền Sạch (%)", color='navy')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper left')

    plt.tight_layout()
    dashboard_png = os.path.join(WORKING_DIR, 'tps_v3.0_remastered_convergence_dashboard.png')
    plt.savefig(dashboard_png, dpi=150)
    plt.show()
    print(f"[✓] Đã xuất dashboard hội tụ V3.0-Remastered R2 sang {dashboard_png}!")
sys.stdout.flush()
