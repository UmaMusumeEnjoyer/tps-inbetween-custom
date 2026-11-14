#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trình khởi chạy Ứng dụng Web Score Weight Calibrator
Tự động kích hoạt Local Web Server và mở trình duyệt mặc định.
"""

import os
import sys
import webbrowser
import socket
from http.server import SimpleHTTPRequestHandler, HTTPServer

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CALIBRATOR_DIR = os.path.dirname(os.path.abspath(__file__))

def find_free_port(start_port=8080):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return 8080

def main():
    os.chdir(CALIBRATOR_DIR)
    port = find_free_port(8080)
    server_address = ('127.0.0.1', port)
    url = f"http://127.0.0.1:{port}/index.html"

    print("=" * 70)
    print("🎨 [SCORE WEIGHT CALIBRATOR] - KHỞI CHẠY ỨNG DỤNG WEB")
    print("=" * 70)
    print(f"[*] Thư mục phục vụ : {CALIBRATOR_DIR}")
    print(f"[*] Địa chỉ máy chủ : {url}")
    print(f"[*] Hướng dẫn       : Bấm Ctrl + C trong terminal này để dừng server khi hoàn tất.")
    print("=" * 70)

    # Đảm bảo dữ liệu 100 triplets đã tồn tại
    if not os.path.exists(os.path.join(CALIBRATOR_DIR, 'dataset_100_triplets.json')):
        print("[!] Chưa tìm thấy dataset_100_triplets.json. Đang tự động trích xuất dữ liệu...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(CALIBRATOR_DIR, 'extract_100_triplets.py')], check=True)

    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"\n[✓] Web server đang lắng nghe tại cổng {port}...")
    print(f"[✓] Đang tự động mở trình duyệt: {url}\n")
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Đang tắt Web server. Tạm biệt!")
        httpd.server_close()

if __name__ == '__main__':
    main()
