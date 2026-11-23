import queue, threading, time
import numpy as np
from diar_stream import OnlineDiarizer


class DiarWorker:
    """Runs OnlineDiarizer in a background thread. push() never blocks."""

    def __init__(self, **kw):
        self.q = queue.Queue(maxsize=200)
        self.d = OnlineDiarizer(**kw)
        self.dropped = 0
        self.events = []
        self.t0 = time.time()
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def push(self, chunk):
        """chunk: float32 mono 16 kHz (the ENHANCED audio)."""
        try:
            self.q.put_nowait(np.array(chunk, dtype=np.float32, copy=True))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                x = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            for ev in self.d.feed(x):
                ev["wall"] = time.time() - self.t0
                self.events.append(ev)
                print(f"\n[Speaker S{ev['spk']}] speech {ev['t0']:.1f}-{ev['t1']:.1f} s "
                      f"(sim {ev['sim']:.2f}, shown at {ev['wall']:.1f} s)")

    def stop(self):
        self._stop.set()
        self.th.join(timeout=2)
        self.events += self.d.finish()
        return self.events