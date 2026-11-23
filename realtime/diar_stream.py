import numpy as np, torch
import sherpa_onnx as s
from silero_vad import load_silero_vad

SR = 16000
FRAME = 512                      # Silero frame size at 16 kHz
WIN = int(1.9 * SR)              # fixed window (model collapses above ~2.0 s)
MIN_SHORT = int(0.75 * SR)       # shortest segment we will tile and label
MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"


class OnlineDiarizer:
    def __init__(self, thr_new=0.30, thr_short=0.20, max_spk=3,
                 vad_thr=0.5, end_silence_frames=10, norm_max=50.0):
        self.ext = s.SpeakerEmbeddingExtractor(
            s.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=1))
        self.vad = load_silero_vad(onnx=True)
        self.thr_new, self.thr_short = thr_new, thr_short
        self.max_spk, self.vad_thr = max_spk, vad_thr
        self.end_sil, self.norm_max = end_silence_frames, norm_max
        self.sums, self.counts = [], []     # running centroids
        self.rest = np.zeros(0, np.float32) # samples not yet a full VAD frame
        self.pend, self.pend_idx = [], []   # pending speech frames + frame idx
        self.sil_run, self.fidx = 0, 0
        self.bad_norm = 0

    def _embed(self, x):
        st = self.ext.create_stream()
        st.accept_waveform(sample_rate=SR, waveform=x.astype(np.float32))
        st.input_finished()
        e = np.array(self.ext.compute(st), dtype=np.float32)
        n = float(np.linalg.norm(e))
        if n > self.norm_max:               # collapsed embedding, ignore
            self.bad_norm += 1
            return None
        return e / (n + 1e-9)

    def _assign(self, e, short):
        if not self.sums:
            if short:
                return None, 0.0
            self.sums.append(e.copy()); self.counts.append(1)
            return 1, 1.0
        cents = [v / (np.linalg.norm(v) + 1e-9) for v in self.sums]
        sims = [float(c @ e) for c in cents]
        b = int(np.argmax(sims))
        thr = self.thr_short if short else self.thr_new
        if sims[b] >= thr:
            self.sums[b] += e; self.counts[b] += 1
            return b + 1, sims[b]
        if short:
            return None, sims[b]
        if len(self.sums) >= self.max_spk:  # cap reached: nearest speaker
            self.sums[b] += e; self.counts[b] += 1
            return b + 1, sims[b]
        self.sums.append(e.copy()); self.counts.append(1)
        return len(self.sums), sims[b]

    def _emit(self, short):
        x = np.concatenate(self.pend)
        t0 = self.pend_idx[0] * FRAME / SR
        t1 = (self.pend_idx[-1] + 1) * FRAME / SR
        if short:
            x = np.tile(x, int(np.ceil(WIN / len(x))))[:WIN]
        else:
            x = x[:WIN]
        e = self._embed(x)
        self.pend, self.pend_idx = [], []
        if e is None:
            return None
        spk, sim = self._assign(e, short)
        if spk is None:
            return None
        return dict(t0=t0, t1=t1, spk=spk, sim=sim, short=short)

    def _flush_segment(self):
        out = None
        n = sum(len(p) for p in self.pend)
        if n >= MIN_SHORT:
            out = self._emit(short=n < WIN)
        else:
            self.pend, self.pend_idx = [], []
        return out

    def feed(self, chunk):
        """chunk: float32 mono 16 kHz, any length. Returns list of label events."""
        events = []
        buf = np.concatenate([self.rest, chunk.astype(np.float32)])
        n_frames = len(buf) // FRAME
        for k in range(n_frames):
            fr = buf[k * FRAME:(k + 1) * FRAME]
            p = float(self.vad(torch.from_numpy(fr), SR))
            if p >= self.vad_thr:
                self.pend.append(fr); self.pend_idx.append(self.fidx)
                self.sil_run = 0
                if sum(len(q) for q in self.pend) >= WIN:
                    ev = self._emit(short=False)
                    if ev: events.append(ev)
            else:
                self.sil_run += 1
                if self.sil_run == self.end_sil and self.pend:
                    ev = self._flush_segment()
                    if ev: events.append(ev)
            self.fidx += 1
        self.rest = buf[n_frames * FRAME:]
        return events

    def finish(self):
        ev = self._flush_segment() if self.pend else None
        return [ev] if ev else []