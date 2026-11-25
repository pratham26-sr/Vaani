"""Streaming ASR on raw mic audio, gated while the system speaks."""
import queue
import threading
import numpy as np
import sherpa_onnx




class AsrWorker(threading.Thread):


    def __init__(self, model_dir, convo, sample_rate=16000, int8=False):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=200)
        self.convo = convo
        self.rate = sample_rate
        suf = ".int8" if int8 else ""
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=model_dir + "/tokens.txt",
            encoder=model_dir + "/encoder-epoch-99-avg-1" + suf + ".onnx",
            decoder=model_dir + "/decoder-epoch-99-avg-1.onnx",
            joiner=model_dir + "/joiner-epoch-99-avg-1" + suf + ".onnx",
            num_threads=1,
            sample_rate=sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            decoding_method="greedy_search",
        )
        self.stream = self.rec.create_stream()
        self.n = 0                 # samples consumed: the clock, in seconds from start
        self.utt_start = None
        self._need_reset = False


    def push(self, chunk):
        if self.convo.speaking.is_set():      # system is talking: ignore the mic
            self._need_reset = True
            self.n += len(chunk)              # keep the clock running
            return
        try:
            self.q.put_nowait(np.asarray(chunk, dtype=np.float32))
        except queue.Full:
            pass                              # never block the audio path


    def run(self):
        while True:
            chunk = self.q.get()
            if self._need_reset:
                self.rec.reset(self.stream)
                self.utt_start = None
                self._need_reset = False
            self.stream.accept_waveform(self.rate, chunk)
            self.n += len(chunk)
            while self.rec.is_ready(self.stream):
                self.rec.decode_stream(self.stream)


            r = self.rec.get_result(self.stream)
            text = (r if isinstance(r, str) else r.text).strip().lower()
            now = self.n / self.rate
            if text and self.utt_start is None:
                self.utt_start = max(0.0, now - 1.0)


            if self.rec.is_endpoint(self.stream):
                if len(text) > 2:
                    self.convo.on_utterance(text, self.utt_start or now - 2, now)
                self.rec.reset(self.stream)
                self.utt_start = None