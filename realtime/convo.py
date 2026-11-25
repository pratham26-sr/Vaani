"""Conversation hub: joins diarization labels with ASR text, holds the transcript."""
import difflib
import re
import threading
import time


KNOWN_NAMES = ["Pratham", "Priyanshu"]     # edit before the demo


INTRO = re.compile(r"\b(?:i am|i'm|im|my name is|this is|call me)\s+([a-z]+)", re.I)
STOP = {"a", "the", "not", "going", "here", "just", "so", "very", "also", "now", "fine"}




def extract_name(text):
    """Return a known name if the text introduces someone, else None."""
    m = INTRO.search(text or "")
    if not m:
        return None
    w = m.group(1).lower()
    if w in STOP:
        return None
    close = difflib.get_close_matches(w, [n.lower() for n in KNOWN_NAMES], n=1, cutoff=0.6)
    if close:
        return next(n for n in KNOWN_NAMES if n.lower() == close[0])
    return None




class Conversation:
    """Thread-safe hub. Diarization calls on_window. ASR calls on_utterance."""


    def __init__(self, assistant=None, settle=2.5, max_gap=3.0):
        self.lock = threading.Lock()
        self.windows = []            # tuples of start, end, label
        self.names = {}              # label to name
        self.turns = []              # transcript entries
        self.speaking = threading.Event()   # set while the system is talking
        self.assistant = assistant
        self.settle = settle
        self.max_gap = max_gap
        self._stop = threading.Event()
        threading.Thread(target=self._resolver, daemon=True).start()


    # ---- inputs -------------------------------------------------------
    def on_window(self, t0, t1, label):
        with self.lock:
            self.windows.append((t0, t1, label))
            self.windows = self.windows[-300:]


    def on_utterance(self, text, t0, t1):
        if not text:
            return
        with self.lock:
            self.turns.append(dict(
                id=len(self.turns) + 1, t0=t0, t1=t1, text=text,
                label=None, speaker=None, emotion=None, reply=None,
                status="waiting", created=time.time()))
            self.turns = self.turns[-200:]


    def add_typed(self, text):
        """A message typed in the dashboard, routed straight to the assistant."""
        e = dict(id=len(self.turns) + 1, t0=0.0, t1=0.0, text=text,
                 label="typed", speaker="You (typed)", emotion=None, reply=None,
                 status="resolved", created=time.time())
        with self.lock:
            self.turns.append(e)
        if self.assistant is not None:
            self.assistant.submit(e)


    # ---- names --------------------------------------------------------
    def display(self, label):
        return self.names.get(label, label)


    def rename(self, label, name):
        with self.lock:
            self.names[label] = name


    # ---- speaker attribution ------------------------------------------
    def _best_label(self, t0, t1):
        """Overlapping speaker window, else the nearest one within max_gap."""
        best = None
        best_gap = 1e9
        for w0, w1, lab in self.windows:
            if min(t1, w1) - max(t0, w0) > 0:
                gap = 0.0
            else:
                gap = max(t0 - w1, w0 - t1)
            if gap < best_gap:
                best = lab
                best_gap = gap
        if best is not None and best_gap <= self.max_gap:
            return best
        return None


    def _resolver(self):
        """Waits for the slower speaker label, then finalizes each utterance."""
        while not self._stop.is_set():
            time.sleep(0.2)
            ready = []
            with self.lock:
                for e in self.turns:
                    if e["status"] != "waiting":
                        continue
                    if time.time() - e["created"] < self.settle:
                        continue
                    e["label"] = self._best_label(e["t0"], e["t1"]) or "S?"
                    nm = extract_name(e["text"])
                    if nm and e["label"] != "S?":
                        self.names[e["label"]] = nm
                    e["speaker"] = self.names.get(e["label"], e["label"])
                    e["status"] = "resolved"
                    ready.append(e)
            for e in ready:
                print("\n[" + str(e["speaker"]) + "] " + e["text"])
                if self.assistant is not None:
                    self.assistant.submit(e)


    def snapshot(self, n=25):
        """Recent transcript, names and windows for the dashboard."""
        with self.lock:
            out = []
            for e in self.turns[-n:]:
                d = dict(e)
                if d["label"]:
                    d["speaker"] = self.names.get(d["label"], d["label"])
                out.append(d)
            return out, dict(self.names), list(self.windows[-40:])


    def stop(self):
        self._stop.set()