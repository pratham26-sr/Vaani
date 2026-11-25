import re, difflib, time, threading

KNOWN_NAMES = ["Priyanshu", "Pratham"]  # edit before the demo

INTRO = re.compile(
    r"\b(?:i am|i'm|im|my name is|this is|call me|it's|its)\s+([a-z]+)",
    re.IGNORECASE,
)
STOP = {"a", "the", "not", "going", "here", "just", "so", "very", "also", "now"}

def extract_name(text: str):
    m = INTRO.search(text)
    if not m:
        return None
    word = m.group(1).lower()
    if word in STOP:
        return None
    close = difflib.get_close_matches(
        word, [n.lower() for n in KNOWN_NAMES], n=1, cutoff=0.6
    )
    if close:
        return next(n for n in KNOWN_NAMES if n.lower() == close[0])
    return None


class NameBinder:
    """Binds a heard name to the diarization label active at that time."""
    def __init__(self, max_wait=6.0):
        self.names = {}      # label -> name
        self.pending = []    # (name, t_start, t_end, created)
        self.windows = []    # (t_start, t_end, label)
        self.max_wait = max_wait
        self.lock = threading.Lock()

    def on_window(self, t0, t1, label):
        with self.lock:
            self.windows.append((t0, t1, label))
            self.windows = self.windows[-50:]
            self._resolve()

    def on_utterance(self, text, t0, t1):
        name = extract_name(text)
        if name:
            with self.lock:
                self.pending.append((name, t0, t1, time.time()))
                self._resolve()

    def rename(self, label, name):
        with self.lock:
            self.names[label] = name

    def display(self, label):
        return self.names.get(label, label)

    def _resolve(self):
        still = []
        for name, t0, t1, created in self.pending:
            best, best_ov = None, 0.0
            for w0, w1, lab in self.windows:
                ov = min(t1, w1) - max(t0, w0)
                if ov > best_ov:
                    best, best_ov = lab, ov
            if best and best_ov > 0.3:
                self.names[best] = name
            elif time.time() - created < self.max_wait:
                still.append((name, t0, t1, created))
        self.pending = still