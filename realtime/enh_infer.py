"""
enh_infer.py  --  CPU inference for the Phase 2 U-Net mask model.
Provides enhance_fn(noisy_np) -> enhanced_np  (1-D float32, same length, 16 kHz).
Put this file next to enh_unet_best.pth and stream_enh.py.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- constants (same as Phase 2) ---
N_FFT, HOP = 512, 128
device = "cpu"
window = torch.hann_window(N_FFT)


# --- model (copied from Phase 2) ---
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, b=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(1, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.bott = block(4 * b, 8 * b)
        self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2); self.d3 = block(8 * b, 4 * b)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2); self.d2 = block(4 * b, 2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2);     self.d1 = block(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        T = x.shape[-1]
        x = F.pad(x, (0, (-T) % 8))          # time must be divisible by 8 (3 poolings)
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.bott(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))[..., :T]


_here = os.path.dirname(os.path.abspath(__file__))
model = UNet()
model.load_state_dict(torch.load(os.path.join(_here, "enh_unet_best.pth"), map_location="cpu"))
model.eval()


@torch.no_grad()
def enhance_fn(noisy_np):
    """
    noisy waveform -> STFT -> mask on the first 256 bins (as in training)
    -> apply mask to the complex spectrum (keeps the NOISY phase) -> ISTFT.
    """
    x = np.asarray(noisy_np, dtype=np.float32).ravel()
    w = torch.from_numpy(x)[None]                                          # (1, samples)
    spec = torch.stft(w, N_FFT, HOP, window=window, return_complex=True)   # (1, 257, T)
    mag = spec.abs()[:, :256]                                              # model uses 256 bins
    mask = model(torch.log1p(mag).unsqueeze(1)).squeeze(1)                 # (1, 256, T)
    mask = torch.cat([mask, mask[:, -1:]], dim=1)                          # 257th (Nyquist) bin: reuse last mask value
    y = torch.istft(spec * mask, N_FFT, HOP, window=window, length=len(x))
    return y[0].numpy()