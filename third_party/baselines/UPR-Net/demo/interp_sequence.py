import os
import sys
import math
import glob
import re
import argparse
import cv2
import torch
from torch.nn import functional as F

# Ensure root directory of UPR-Net is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pipeline import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def interp_pair(ppl, ori_img0, ori_img1, time_period=0.5, pyr_level=None):
    img0 = (torch.tensor(ori_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    img1 = (torch.tensor(ori_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    if pyr_level is None:
        pyr_level = math.ceil(math.log2(w / 448) + 3)

    divisor = 2 ** (pyr_level - 1 + 2)
    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    with torch.no_grad():
        interp_img, _ = ppl.inference(img0, img1, time_period=time_period, pyr_level=pyr_level)
    interp_img = interp_img[:, :, :h, :w]
    interp_img = (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return interp_img


def main():
    parser = argparse.ArgumentParser(description="Interpolate a sequence of frames using UPR-Net")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing sequence of images")
    parser.add_argument("--save_dir", type=str, default="./demo/sequence_output", help="Directory to save output")
    parser.add_argument("--interp_factor", type=int, default=2, choices=[2, 4, 8], help="Interpolation factor (e.g. 2 means 2x FPS)")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large", "LARGE"], help="Model size")
    parser.add_argument("--model_file", type=str, default="./checkpoints/upr-base.pkl", help="Checkpoint path")
    parser.add_argument("--create_video", action="store_true", help="Also export output frames as mp4 video")
    parser.add_argument("--fps", type=int, default=12, help="FPS for exported video")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Gather images
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    img_files = []
    for ext in valid_exts:
        img_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    img_files = sorted(img_files, key=natural_sort_key)

    if len(img_files) < 2:
        print(f"Error: Found only {len(img_files)} images in {args.input_dir}. At least 2 images required.")
        return

    print(f"Found {len(img_files)} images in {args.input_dir}.")
    print(f"Loading UPR-Net ({args.model_size}) from {args.model_file} on {DEVICE}...")

    model_cfg_dict = dict(
        load_pretrain=True,
        model_size=args.model_size,
        model_file=args.model_file
    )
    ppl = Pipeline(model_cfg_dict)
    ppl.eval()

    output_frames = []
    frame_counter = 0

    print("Starting interpolation...")
    for i in range(len(img_files) - 1):
        img0_path = img_files[i]
        img1_path = img_files[i + 1]

        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)

        if img0 is None or img1 is None:
            print(f"Skipping unreadable frame pair: {img0_path}, {img1_path}")
            continue

        # Save first frame of pair
        out_name = f"frame_{frame_counter:05d}.png"
        cv2.imwrite(os.path.join(args.save_dir, out_name), img0)
        output_frames.append(img0)
        frame_counter += 1

        # Determine intermediate timestamps
        # For 2x: [0.5]
        # For 4x: [0.25, 0.5, 0.75]
        # For 8x: [1/8, 2/8, ..., 7/8]
        steps = args.interp_factor
        for s in range(1, steps):
            t = s / float(steps)
            interp_res = interp_pair(ppl, img0, img1, time_period=t)
            out_name = f"frame_{frame_counter:05d}.png"
            cv2.imwrite(os.path.join(args.save_dir, out_name), interp_res)
            output_frames.append(interp_res)
            frame_counter += 1

        print(f"Processed pair [{i+1}/{len(img_files)-1}]: {os.path.basename(img0_path)} -> {os.path.basename(img1_path)}")

    # Save the last frame of the entire sequence
    last_img = cv2.imread(img_files[-1])
    out_name = f"frame_{frame_counter:05d}.png"
    cv2.imwrite(os.path.join(args.save_dir, out_name), last_img)
    output_frames.append(last_img)

    print(f"\nAll frames saved to {args.save_dir} (Total frames: {len(output_frames)})")

    if args.create_video:
        video_path = os.path.join(args.save_dir, "interpolated_video.mp4")
        h, w, _ = output_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
        for f in output_frames:
            out_video.write(f)
        out_video.release()
        print(f"Video saved to: {video_path} (FPS: {args.fps})")


if __name__ == "__main__":
    main()
