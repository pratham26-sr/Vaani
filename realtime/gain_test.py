"""
gain_test.py -- does input LEVEL explain the faded/mumbled speech?

Reads live_raw.wav (what your laptop mic recorded), amplifies it by different gains,
runs the SAME model on each, and writes the results for listening.

All output files are normalized to the same loudness (RMS 0.09), so when you listen
you compare how well the speech is PRESERVED, not which one is louder.
"""
import numpy as np
import soundfile as sf

from enh_infer import enhance_fn

SR, TARGET = 16000, 0.09
rms = lambda x: float(np.sqrt((x ** 2).mean()))
norm = lambda x: np.clip(x * (TARGET / (rms(x) + 1e-9)), -1, 1).astype(np.float32)

raw, sr = sf.read("live_raw.wav", dtype="float32")
assert sr == SR
g_auto = TARGET / rms(raw)            # gain that brings the recording to the training level
gains = [1.0, 0.5 * g_auto, g_auto]

sf.write("gain_ref_raw.wav", norm(raw), SR)
print(f"input RMS {rms(raw):.4f} | peak {np.abs(raw).max():.3f} | auto gain {g_auto:.2f}")
print(f"{'gain':>6s} {'in RMS':>8s} {'out RMS':>8s} {'out/in':>7s}  file")
for g in gains:
    x = np.clip(raw * g, -1, 1)
    y = enhance_fn(x)
    name = f"gain_x{g:.2f}.wav"
    sf.write(name, norm(y), SR)
    print(f"{g:6.2f} {rms(x):8.4f} {rms(y):8.4f} {rms(y) / rms(x):7.2f}  {name}")