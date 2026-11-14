import os
import sys
import cv2
import torch
import numpy as np

# Path to anime-segmentation model directory
seg_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'anime-segmentation', 'model'))
if seg_model_dir not in sys.path:
    sys.path.insert(0, seg_model_dir)

from isnet import ISNetDIS

class AnimeBgRemover:
    def __init__(self, ckpt_path=None, device='cuda:0', img_size=1024):
        self.device = torch.device(device)
        self.img_size = img_size
        if ckpt_path is None:
            ckpt_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'anime-segmentation', 'saved_models', 'isnetis.ckpt')
            )
        self.ckpt_path = ckpt_path
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Anime-segmentation checkpoint not found at: {ckpt_path}")
            
        self.net = ISNetDIS().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        clean_sd = {k.replace('net.', ''): v for k, v in sd.items() if k.startswith('net.')}
        if not clean_sd:
            clean_sd = sd
        self.net.load_state_dict(clean_sd)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    def remove_background(self, img_bgr):
        """
        Input: img_bgr (np.ndarray HxWx3 uint8)
        Output: img_white_bg (np.ndarray HxWx3 uint8) where background is pure white
        """
        h0, w0 = img_bgr.shape[:2]
        s = self.img_size
        h, w = (s, int(s * w0 / h0)) if h0 > w0 else (int(s * h0 / w0), s)
        ph, pw = s - h, s - w
        
        inp = np.zeros([s, s, 3], dtype=np.float32)
        inp[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img_bgr / 255.0, (w, h))
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis, :]
        inp_tensor = torch.from_numpy(inp).float().to(self.device)
        
        with torch.no_grad():
            if 'cuda' in self.device.type:
                with torch.amp.autocast(device_type='cuda'):
                    pred = self.net(inp_tensor)[0][0].sigmoid()
            else:
                pred = self.net(inp_tensor)[0][0].sigmoid()
            
            pred = pred.to(dtype=torch.float32).cpu().numpy()[0]
            pred = np.transpose(pred, (1, 2, 0))[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
            mask = cv2.resize(pred, (w0, h0))
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
                
        img_white_bg = np.clip(mask * img_bgr + (1.0 - mask) * 255.0, 0, 255).astype(np.uint8)
        return img_white_bg
