#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_op_ed_detector.py
Bộ công cụ nhận diện và lọc bỏ Opening (OP), Ending (ED), Preview, Eyecatch qua các tập phim Anime.

Phương pháp:
1. Engine 1: MKV Chapter Metadata (Ưu tiên số 1 - Chính xác 100% từng mili-giây cho các bản BD / WebRip).
2. Engine 2: Audio Cross-Correlation (Tự động so khớp đoạn nhạc OP/ED 90s lặp lại giữa các tập khi không có chapter).
3. Engine 3: Smart Japanese Anime Rule Fallback (Cắt đúng cửa sổ 90s chuẩn thay vì cắt mù 5 phút).
"""

import os
import sys
import glob
import json
import subprocess
import tempfile
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Các từ khóa thường dùng trong Chapter của Anime Nhật Bản để nhận diện phần cần LOẠI BỎ
SKIP_KEYWORDS = [
    'op', 'opening', 'ed', 'ending', 'preview', 'outro',
    'credit', 'eyecatch', 'staff', 'cast', 'ncop', 'nced'
]

# Các từ khóa của nội dung CẦN GIỮ LẠI
KEEP_KEYWORDS = [
    'intro', 'prologue', 'episode', 'part a', 'part b',
    'part 1', 'part 2', 'main', 'epilogue', 'prelude'
]


def get_video_duration(video_path: str) -> float:
    """Lấy thời lượng video (giây) bằng ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def detect_mkv_chapters(video_path: str):
    """
    Trích xuất và phân tích chapter metadata từ file MKV.
    Trả về: (valid_ranges, skipped_ranges) hoặc (None, None) nếu không có chapter.
    """
    cmd = ['ffprobe', '-v', 'error', '-show_chapters', '-print_format', 'json', video_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
    except Exception:
        return None, None

    chapters = data.get('chapters', [])
    if not chapters:
        return None, None

    valid_ranges = []
    skipped_ranges = []

    for ch in chapters:
        title = ch.get('tags', {}).get('title', '').strip()
        start = float(ch.get('start_time', 0))
        end = float(ch.get('end_time', 0))
        dur = end - start

        # Kiểm tra từ khóa loại bỏ
        is_skip = any(kw in title.lower() for kw in SKIP_KEYWORDS)
        
        # Nếu chapter quá ngắn (< 3 giây) như eyecatch hoặc flash
        if dur < 3.0:
            is_skip = True

        if is_skip:
            skipped_ranges.append({'title': title, 'start': start, 'end': end, 'duration': dur})
        else:
            valid_ranges.append({'title': title, 'start': start, 'end': end, 'duration': dur})

    return valid_ranges, skipped_ranges


def detect_op_ed_audio_correlation(ep_path_a: str, ep_path_b: str, window_sec: int = 360):
    """
    So khớp tương quan âm thanh (Audio Cross-Correlation) giữa 2 tập phim cùng bộ
    để tìm đoạn Opening nhạc 90 giây lặp lại.
    """
    from scipy import signal
    from scipy.io import wavfile

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, 'a.wav')
        wav_b = os.path.join(tmpdir, 'b.wav')

        # Trích xuất 6 phút đầu tiên dưới dạng 8kHz mono 16-bit
        cmd_a = ['ffmpeg', '-y', '-ss', '0', '-t', str(window_sec), '-i', ep_path_a,
                 '-vn', '-ac', '1', '-ar', '8000', '-c:a', 'pcm_s16le', wav_a]
        cmd_b = ['ffmpeg', '-y', '-ss', '0', '-t', str(window_sec), '-i', ep_path_b,
                 '-vn', '-ac', '1', '-ar', '8000', '-c:a', 'pcm_s16le', wav_b]

        subprocess.run(cmd_a, capture_output=True, check=True)
        subprocess.run(cmd_b, capture_output=True, check=True)

        sr_a, data_a = wavfile.read(wav_a)
        sr_b, data_b = wavfile.read(wav_b)

        # Tính bao năng lượng âm thanh (RMS envelope)
        hop = 80  # 100 mẫu/giây (độ phân giải 10ms)
        env_a = np.array([np.sqrt(np.mean(data_a[i:i+hop].astype(float)**2))
                          for i in range(0, len(data_a) - hop, hop)])
        env_b = np.array([np.sqrt(np.mean(data_b[i:i+hop].astype(float)**2))
                          for i in range(0, len(data_b) - hop, hop)])

        # Trượt một đoạn mẫu 80s để dò tìm OP
        sample_len = 80 * 100  # 80s
        best_corr = -1
        best_pos_a = 0
        best_pos_b = 0

        # Quét qua ep_a
        for start_a in range(0, len(env_a) - sample_len, 50):
            patch = env_a[start_a:start_a + sample_len]
            norm_patch = patch - np.mean(patch)
            patch_std = np.std(patch)
            if patch_std < 1e-4:
                continue

            corr = signal.correlate(env_b - np.mean(env_b), norm_patch, mode='valid')
            max_idx = np.argmax(corr)
            max_val = corr[max_idx] / (len(norm_patch) * patch_std * (np.std(env_b[max_idx:max_idx+sample_len]) + 1e-6))

            if max_val > best_corr:
                best_corr = max_val
                best_pos_a = start_a / 100.0
                best_pos_b = max_idx / 100.0

        if best_corr > 0.65:
            # Phát hiện thành công OP lặp lại ~90s
            op_start_a = max(0.0, best_pos_a - 5.0)
            op_end_a = op_start_a + 90.0
            return {'op_start': op_start_a, 'op_end': op_end_a, 'confidence': float(best_corr)}

    return None


def get_clean_segments_for_video(video_path: str, sibling_video_path: str = None) -> list:
    """
    Xác định toàn bộ các phân đoạn thời gian sạch (loại bỏ OP, ED, Preview) của 1 video.
    Trả về: danh sách các tuple `[(start_sec, end_sec), ...]`
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []

    # 1. Thử đọc Chapter metadata (Cách nhanh và chuẩn nhất)
    valid_chs, skipped_chs = detect_mkv_chapters(video_path)
    if valid_chs is not None and len(valid_chs) > 0:
        clean_ranges = []
        for ch in valid_chs:
            clean_ranges.append((round(ch['start'], 2), round(ch['end'], 2)))
        return clean_ranges

    # 2. Nếu không có chapter và có video tập khác cùng bộ -> Thử Audio correlation
    if sibling_video_path and os.path.exists(sibling_video_path):
        try:
            detected = detect_op_ed_audio_correlation(video_path, sibling_video_path)
            if detected:
                op_s = detected['op_start']
                op_e = detected['op_end']
                ed_s = max(0.0, duration - 180.0)
                ed_e = ed_s + 90.0

                ranges = []
                if op_s > 10.0:  # Có Prologue trước OP
                    ranges.append((0.0, op_s))
                ranges.append((op_e, ed_s))
                if duration - ed_e > 10.0:  # Có Epilogue sau ED
                    ranges.append((ed_e, duration - 15.0))  # Bỏ 15s preview cuối
                return ranges
        except Exception as e:
            pass

    # 3. Fallback Heuristic chuẩn Anime Nhật Bản (Chính xác hơn nhiều so với việc cắt mù 5 phút)
    # Cấu trúc phổ biến:
    # 0s - 90s: Prologue (Giữ lại!)
    # 90s - 180s: Opening 90s chuẩn (Bỏ)
    # 180s - (duration - 180s): Phần phim chính (Giữ lại!)
    # (duration - 180s) - (duration - 90s): Ending 90s chuẩn (Bỏ)
    # (duration - 90s) - (duration - 20s): Epilogue (Giữ lại nếu có)
    # (duration - 20s) - duration: Preview (Bỏ)
    op_start = 90.0
    op_end = 180.0
    ed_start = max(180.0, duration - 180.0)
    ed_end = ed_start + 90.0

    ranges = [
        (0.0, op_start),
        (op_end, ed_start)
    ]
    if duration - ed_end > 25.0:
        ranges.append((ed_end, duration - 20.0))

    return ranges


def scan_anime_directory(anime_root: str, output_json: str = "clean_segments_manifest.json"):
    """Quét toàn bộ video trong thư mục AnimeGoc và xuất file manifest JSON."""
    video_files = sorted(glob.glob(os.path.join(anime_root, "**", "*.mkv"), recursive=True))
    print(f"[*] Tìm thấy tổng cộng {len(video_files)} file MKV trong: {anime_root}")

    results = {}
    
    # Gom nhóm theo thư mục (bộ phim)
    groups = {}
    for f in video_files:
        pdir = os.path.dirname(f)
        groups.setdefault(pdir, []).append(f)

    for pdir, files in groups.items():
        dir_name = os.path.basename(pdir)
        print(f"\n📂 Đang phân tích bộ phim: {dir_name} ({len(files)} tập)...")

        for idx, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            sibling = files[1] if idx == 0 and len(files) > 1 else files[0] if len(files) > 1 else None
            
            # Kiểm tra chapter
            valid_chs, skipped_chs = detect_mkv_chapters(fpath)
            method = "MKV_CHAPTERS" if (valid_chs is not None and len(valid_chs) > 0) else "HEURISTIC_RULE"
            
            clean_ranges = get_clean_segments_for_video(fpath, sibling)
            total_clean_time = sum([e - s for s, e in clean_ranges])
            total_dur = get_video_duration(fpath)

            results[fpath] = {
                'relative_path': os.path.relpath(fpath, anime_root),
                'filename': fname,
                'total_duration_sec': round(total_dur, 2),
                'clean_duration_sec': round(total_clean_time, 2),
                'detection_method': method,
                'clean_ranges': clean_ranges,
                'skipped_chapters': skipped_chs if skipped_chs else []
            }
            print(f"  [{method}] {fname} -> Giữ lại {total_clean_time:.1f}s / {total_dur:.1f}s ({len(clean_ranges)} đoạn)")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Hoàn thành! Đã lưu bản đồ phân đoạn sạch vào: {output_json}")
    return results


if __name__ == '__main__':
    default_anime_dir = r"E:\CODE\PBL\PBL6\AnimeGoc"
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_segments_manifest.json")
    scan_anime_directory(default_anime_dir, out_file)
