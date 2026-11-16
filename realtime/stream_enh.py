"""
stream_enh.py  --  Real-time stage 1: chunked (streaming) enhancement + RTF benchmark

Wraps your OFFLINE enhance_fn (1-D float32 numpy in -> 1-D numpy out, same length)
so it can run on a live audio stream.

How it works (overlap-add, 50% overlap):
    window W = win_s seconds, hop H = W/2
    every time W samples are available -> run enhance_fn on them -> multiply the
    output by a periodic Hann window -> add it onto the previous window's second half.
    Two shifted periodic-Hann windows sum to exactly 1, so with a "do nothing" model
    the output equals the input (this is the self-test).
    Algorithmic latency = W (a sample must wait until its whole window has arrived),
    plus the compute time of one window.

Real-time condition: compute time per window  <  hop duration  (RTF < 1).

Usage on a machine with your model (laptop CPU):
    python stream_enh.py --selftest
    python stream_enh.py --module enh_infer --win 1.0 --hop 0.5 --threads 4
where enh_infer.py defines enhance_fn(noisy_np) -> enhanced_np.
"""
import argparse
import importlib
import time

import numpy as np

SR = 16000


def _fit(x, n):
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


class ChunkedEnhancer:
    def __init__(self, enhance_fn, win_s=1.0, hop_s=0.5, sr=SR):
        self.fn = enhance_fn
        self.W = int(round(win_s * sr))
        self.H = int(round(hop_s * sr))
        assert self.W == 2 * self.H, "this version needs 50% overlap (hop = win / 2)"
        # periodic Hann: copies shifted by W/2 add up to exactly 1
        self.win = np.hanning(self.W + 1)[:-1].astype(np.float32)
        self.window_times = []
        self.reset()

    def reset(self):
        # H zeros of history, so the first real samples get full weight
        self.buf = np.zeros(self.H, dtype=np.float32)
        self.tail = np.zeros(self.H, dtype=np.float32)  # second half of the last window
        self.first = True        # the very first block is just the zero padding -> dropped
        self.n_in = 0
        self.n_out = 0

    def clear_stats(self):
        self.window_times = []

    def process(self, chunk):
        """Feed any number of new samples; returns the newly finished output samples."""
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        self.n_in += len(chunk)
        self.buf = np.concatenate([self.buf, chunk])
        outs = []
        while len(self.buf) >= self.W:
            x = self.buf[:self.W]
            t0 = time.perf_counter()
            y = np.asarray(self.fn(x), dtype=np.float32).ravel()
            self.window_times.append(time.perf_counter() - t0)
            y = _fit(y, self.W) * self.win
            block = self.tail + y[:self.H]      # overlap-add: previous tail + this head
            self.tail = y[self.H:]
            self.buf = self.buf[self.H:]
            if self.first:
                self.first = False
            else:
                outs.append(block)
        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        self.n_out += len(out)
        return out

    def flush(self):
        """Call at the end of the stream: returns the last pending samples."""
        n_in, n_out = self.n_in, self.n_out
        out = self.process(np.zeros(self.W, dtype=np.float32))
        keep = max(0, n_in - n_out)
        self.n_in, self.n_out = n_in, n_out + keep
        return out[:keep]


def stream_file(enh, wav, frame_s=0.02):
    """Simulate a microphone: push the file through in 20 ms frames."""
    enh.reset()
    fr = int(frame_s * SR)
    outs = []
    for i in range(0, len(wav), fr):
        o = enh.process(wav[i:i + fr])
        if len(o):
            outs.append(o)
    outs.append(enh.flush())
    return np.concatenate(outs)


# ----------------------------------------------------------------------------- tests
def selftest():
    """With a 'do nothing' model, streaming must reproduce the input exactly."""
    rng = np.random.default_rng(0)
    ok = True
    for win_s, hop_s in [(1.0, 0.5), (0.5, 0.25), (2.0, 1.0)]:
        for n in [SR // 3, SR, 3 * SR + 123, 5 * SR]:
            x = rng.standard_normal(n).astype(np.float32)
            y = stream_file(ChunkedEnhancer(lambda a: a, win_s, hop_s), x)
            err = np.max(np.abs(x - y)) if len(y) == len(x) else float("inf")
            good = len(y) == len(x) and err < 1e-4
            ok &= good
            print(f"win={win_s} n={n:6d}  len_out={len(y):6d}  max_err={err:.2e}  {'OK' if good else 'FAIL'}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------- RTF
def benchmark_rtf(enhance_fn, win_s=1.0, hop_s=0.5, seconds=20, warmup=3, budget=0.5):
    """
    Times the streaming enhancer on `seconds` of audio.
    Content doesn't matter for compute time, so random noise is fine.
    budget: fraction of the hop you allow enhancement to use (the rest is for
            VAD, diarization, ASR, ...). 0.5 means enhancement may use at most half.
    """
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
    enh = ChunkedEnhancer(enhance_fn, win_s, hop_s)
    for _ in range(warmup):                 # first calls are slow (allocation, caches)
        enhance_fn(x[:enh.W])
    enh.clear_stats()
    t0 = time.perf_counter()
    stream_file(enh, x)
    total = time.perf_counter() - t0
    wt = np.array(enh.window_times) * 1000.0
    hop_ms = hop_s * 1000.0
    res = {
        "audio_s": seconds,
        "total_compute_s": total,
        "RTF": total / seconds,
        "win_ms_mean": wt.mean(),
        "win_ms_p95": np.percentile(wt, 95),
        "win_ms_max": wt.max(),
        "hop_ms": hop_ms,
        "latency_ms_est": win_s * 1000.0 + np.percentile(wt, 95),
    }
    res["realtime_ok"] = res["win_ms_p95"] < budget * hop_ms
    print(f"window={win_s:.2f}s hop={hop_s:.2f}s | RTF={res['RTF']:.3f} | "
          f"per-window ms: mean={res['win_ms_mean']:.1f} p95={res['win_ms_p95']:.1f} "
          f"max={res['win_ms_max']:.1f} (hop={hop_ms:.0f}) | "
          f"est. latency={res['latency_ms_est']:.0f} ms | "
          f"{'OK' if res['realtime_ok'] else 'TOO SLOW'} (budget {int(budget * 100)}% of hop)")
    return res


# ----------------------------------------------------------------------------- quality
def si_snr(est, ref, eps=1e-8):
    n = min(len(est), len(ref))
    est = est[:n] - est[:n].mean()
    ref = ref[:n] - ref[:n].mean()
    s = (est @ ref) / (ref @ ref + eps) * ref
    e = est - s
    return 10 * np.log10((s @ s + eps) / (e @ e + eps))


def compare_offline_vs_stream(enhance_fn, pairs, win_s=1.0, hop_s=0.5):
    """
    pairs: list of (clean_path, noisy_path) 16 kHz wavs.
    Prints mean STOI / SI-SNR for noisy, offline-enhanced and streaming-enhanced.
    """
    import soundfile as sf
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None
    enh = ChunkedEnhancer(enhance_fn, win_s, hop_s)
    rows = {"noisy": [], "offline": [], "streaming": []}
    for cp, np_ in pairs:
        clean, _ = sf.read(cp, dtype="float32")
        noisy, _ = sf.read(np_, dtype="float32")
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]
        off = _fit(np.asarray(enhance_fn(noisy), dtype=np.float32).ravel(), n)
        stm = _fit(stream_file(enh, noisy), n)
        for name, sig in [("noisy", noisy), ("offline", off), ("streaming", stm)]:
            rows[name].append((si_snr(sig, clean),
                               stoi(clean, sig, SR, extended=False) if stoi else float("nan")))
    print(f"window={win_s}s hop={hop_s}s  ({len(pairs)} files)")
    print(f"{'condition':10s} {'SI-SNR dB':>10s} {'STOI':>8s}")
    out = {}
    for name, v in rows.items():
        a = np.array(v)
        out[name] = (a[:, 0].mean(), a[:, 1].mean())
        print(f"{name:10s} {out[name][0]:10.2f} {out[name][1]:8.3f}")
    return out


# ----------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--module", help="python module that defines enhance_fn(noisy_np)")
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.module:
        if a.threads:
            import torch
            torch.set_num_threads(a.threads)
        fn = importlib.import_module(a.module).enhance_fn
        benchmark_rtf(fn, a.win, a.hop, a.seconds)
