"""Runs OnlineDiarizer in a background thread. push() never blocks."""
import queue
import threading
import time
import numpy as np
from diar_stream import OnlineDiarizer




class DiarWorker:


    def __init__(self, **kw):
        self.q = queue.Queue(maxsize=200)
        self.d = OnlineDiarizer(**kw)
        self.dropped = 0
        self.events = []
        self.convo = None                 # set from livedemo.py (optional)
        self.t0 = time.time()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()


    def push(self, chunk):
        """chunk: float32 mono 16 kHz."""
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
                label = "S" + str(ev["spk"])
                shown = label
                if self.convo is not None:
                    self.convo.on_window(ev["t0"], ev["t1"], label)
                    shown = self.convo.display(label)
                tag = shown + " (" + label + ")" if shown != label else label
                print("\n[Speaker " + tag + "] speech "
                      + format(ev["t0"], ".1f") + "-" + format(ev["t1"], ".1f") + " s "
                      + "(sim " + format(ev["sim"], ".2f") + ", shown at "
                      + format(ev["wall"], ".1f") + " s)")


    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2)
        self.events += self.d.finish()
        return self.events