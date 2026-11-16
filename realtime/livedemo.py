"""
live_demo.py  --  Real-time stage 2: live microphone -> streaming enhancement -> headphones

Put next to stream_enh.py, enh_infer.py and enh_unet_best.pth, then run:
    python live_demo.py --seconds 30

How it is built (2 threads, so the audio callback never waits for the model):
    audio callback (every 20 ms): copies the mic frame into in_q, and plays whatever
                                  enhanced audio is ready from the output buffer
    worker thread               : takes frames from in_q, runs ChunkedEnhancer,
                                  pushes enhanced samples to the output buffer
While it runs, press ENTER to toggle between ENHANCED and RAW mic (A/B listening).
USE HEADPHONES, otherwise the speakers feed back into the mic.

Measured at the end (does NOT include the audio devices' own buffering, e.g. Bluetooth):
    latency   : time from capturing a sample to its enhanced version being ready
    underruns : times the speaker callback had no audio ready (audible dropouts)
"""
import argparse
import queue
import threading
import time
from collections import deque

import numpy as np
import soundfile as sf

from stream_enh import ChunkedEnhancer, SR


class OutBuffer:
    """Thread-safe FIFO of audio samples."""

    def __init__(self):
        self.q, self.n, self.lock = deque(), 0, threading.Lock()

    def push(self, a):
        with self.lock:
            self.q.append(a)
            self.n += len(a)

    def pull(self, n):
        """Exactly n samples, or None if not enough are ready (underrun)."""
        with self.lock:
            if self.n < n:
                return None
            parts, need = [], n
            while need:
                a = self.q[0]
                if len(a) <= need:
                    parts.append(a)
                    need -= len(a)
                    self.q.popleft()
                else:
                    parts.append(a[:need])
                    self.q[0] = a[need:]
                    need = 0
            self.n -= n
        return np.concatenate(parts)


class LivePipeline:
    """Everything except the sound card, so the logic can be tested without audio."""

    def __init__(self, enhance_fn, win_s=0.5, hop_s=0.25, block=320):
        self.enh = ChunkedEnhancer(enhance_fn, win_s, hop_s)
        self.block = block
        self.in_q = queue.Queue()
        self.out = OutBuffer()
        self.bypass = False
        self.frame_ts, self.raw, self.enh_log, self.lat = [], [], [], []
        self.max_q = 0
        self.running = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def step(self, ts, x):
        """Process one mic frame captured at time ts."""
        self.frame_ts.append(ts)
        self.raw.append(x)
        n_before = self.enh.n_out                 # index of the first newly finished sample
        y = self.enh.process(x)
        if self.bypass:
            self.out.push(x)
        elif len(y):
            self.out.push(y)
        if len(y):
            self.enh_log.append(y)
            i = min(n_before // self.block, len(self.frame_ts) - 1)
            self.lat.append(time.perf_counter() - self.frame_ts[i])  # worst case in this block

    def _worker(self):
        while self.running.is_set() or not self.in_q.empty():
            try:
                ts, x = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.max_q = max(self.max_q, self.in_q.qsize())
            self.step(ts, x)

    def start(self):
        self.running.set()
        self.thread.start()

    def stop(self):
        self.running.clear()
        self.thread.join()

    def report(self, underruns, status_flags):
        lat = np.array(self.lat) * 1000
        wt = np.array(self.enh.window_times) * 1000
        print("\n===== RESULTS =====")
        if len(lat):
            print(f"capture -> enhanced latency (worst sample per block): "
                  f"mean {lat.mean():.0f} ms | p95 {np.percentile(lat, 95):.0f} ms | max {lat.max():.0f} ms")
        if len(wt):
            print(f"per-window compute: mean {wt.mean():.1f} ms | p95 {np.percentile(wt, 95):.1f} ms "
                  f"(hop = {self.enh.H / SR * 1000:.0f} ms)")
        print(f"underruns (audible dropouts): {underruns} | stream status flags: {status_flags} "
              f"| max input queue: {self.max_q} frames")

    def save(self, prefix="live"):
        if self.raw:
            sf.write(f"{prefix}_raw.wav", np.concatenate(self.raw), SR)
        if self.enh_log:
            sf.write(f"{prefix}_enh.wav", np.concatenate(self.enh_log), SR)
        print(f"saved {prefix}_raw.wav and {prefix}_enh.wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--win", type=float, default=0.5)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--block", type=int, default=320, help="samples per audio callback (320 = 20 ms)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--in-dev", type=int, default=None, help="input device index (default = system default)")
    ap.add_argument("--out-dev", type=int, default=None, help="output device index (default = system default)")
    ap.add_argument("--bypass", action="store_true", help="start in RAW mode")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    import sounddevice as sd
    from enh_infer import enhance_fn

    for _ in range(3):                                   # warm-up: first calls are slow
        enhance_fn(np.zeros(int(args.win * SR), dtype=np.float32))

    out_ch = min(2, sd.query_devices(args.out_dev, "output")["max_output_channels"])
    in_name = sd.query_devices(args.in_dev, "input")["name"]
    out_name = sd.query_devices(args.out_dev, "output")["name"]
    print(f"input : {in_name}\noutput: {out_name}\n"
          f"window {args.win}s / hop {args.hop}s | ENTER toggles ENHANCED/RAW | Ctrl+C stops early")

    pipe = LivePipeline(enhance_fn, args.win, args.hop, args.block)
    pipe.bypass = args.bypass
    cnt = {"underruns": 0, "status": 0}
    started = [False]

    def callback(indata, outdata, frames, time_info, status):
        if status:
            cnt["status"] += 1
        pipe.in_q.put((time.perf_counter(), indata[:, 0].copy()))
        y = pipe.out.pull(frames)
        if y is None:
            outdata.fill(0)
            if started[0]:
                cnt["underruns"] += 1
        else:
            started[0] = True
            outdata[:] = y[:, None]

    def toggler():
        while True:
            input()
            pipe.bypass = not pipe.bypass
            print("-> RAW mic" if pipe.bypass else "-> ENHANCED")

    threading.Thread(target=toggler, daemon=True).start()
    pipe.start()
    t_end = time.time() + args.seconds
    try:
        with sd.Stream(samplerate=SR, blocksize=args.block, device=(args.in_dev, args.out_dev),
                       channels=(1, out_ch), dtype="float32", callback=callback):
            last = 0
            while time.time() < t_end:
                time.sleep(0.1)
                if time.time() - last > 2:
                    last = time.time()
                    lat = f"{pipe.lat[-1] * 1000:.0f} ms" if pipe.lat else "-"
                    print(f"{'RAW' if pipe.bypass else 'ENH'} | latency {lat} | queue {pipe.in_q.qsize()} "
                          f"| underruns {cnt['underruns']}")
    except KeyboardInterrupt:
        pass
    pipe.stop()
    pipe.report(cnt["underruns"], cnt["status"])
    pipe.save()


if __name__ == "__main__":
    main()