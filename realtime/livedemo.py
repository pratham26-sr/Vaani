"""
Real-time speech enhancement, diarization, ASR, emotion and spoken replies.


    python livedemo.py --diar --assist --mute
    then open http://localhost:8000 and drive everything from the browser.
"""
import argparse
import queue
import threading
import time
from collections import deque


import numpy as np
import soundfile as sf


from stream_enh import ChunkedEnhancer, SR


ASR_MODEL = "models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"




class OutBuffer:
    def __init__(self):
        self.q = deque()
        self.n = 0
        self.lock = threading.Lock()


    def push(self, a):
        with self.lock:
            self.q.append(a)
            self.n += len(a)


    def pull(self, n):
        with self.lock:
            if self.n < n:
                return None
            parts = []
            need = n
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
    def __init__(self, enhance_fn, win_s=0.5, hop_s=0.25, block=320):
        self.enh = ChunkedEnhancer(enhance_fn, win_s, hop_s)
        self.block = block
        self.in_q = queue.Queue()
        self.out = OutBuffer()
        self.bypass = False
        self.frame_ts = []
        self.raw = []
        self.enh_log = []
        self.lat = []
        self.level = 0.0
        self.keep = 300 * SR // block       # keep about the last 5 minutes
        self.max_q = 0
        self.diar = None
        self.diar_raw = True
        self.asr = None
        self.convo = None
        self._dbuf = []
        self._dn = 0
        self.running = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)


    def step(self, ts, x):
        self.frame_ts.append(ts)
        self.raw.append(x)
        self.level = float(np.sqrt(np.mean(x * x)))
        muted = self.convo is not None and self.convo.speaking.is_set()


        if self.asr is not None:
            self.asr.push(x)                      # gated inside AsrWorker


        if self.diar is not None and self.diar_raw and not muted:
            self._dbuf.append(x)
            self._dn += len(x)
            if self._dn >= self.enh.H:
                self.diar.push(np.concatenate(self._dbuf))
                self._dbuf = []
                self._dn = 0


        n_before = self.enh.n_out
        y = self.enh.process(x)
        if self.bypass:
            self.out.push(x)
        elif len(y):
            self.out.push(y)
        if len(y):
            self.enh_log.append(y)
            if self.diar is not None and not self.diar_raw and not muted:
                self.diar.push(y)
            i = min(n_before // self.block, len(self.frame_ts) - 1)
            self.lat.append(time.perf_counter() - self.frame_ts[i])


        if len(self.raw) > self.keep:
            self.raw = self.raw[-self.keep:]
            self.frame_ts = self.frame_ts[-self.keep:]
        if len(self.enh_log) > self.keep:
            self.enh_log = self.enh_log[-self.keep:]
        if len(self.lat) > 5000:
            self.lat = self.lat[-5000:]


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
            print("capture -> enhanced latency: mean " + format(lat.mean(), ".0f")
                  + " ms | p95 " + format(np.percentile(lat, 95), ".0f")
                  + " ms | max " + format(lat.max(), ".0f") + " ms")
        if len(wt):
            print("per-window compute: mean " + format(wt.mean(), ".1f")
                  + " ms | p95 " + format(np.percentile(wt, 95), ".1f")
                  + " ms (hop = " + format(self.enh.H / SR * 1000, ".0f") + " ms)")
        print("underruns: " + str(underruns) + " | status flags: " + str(status_flags)
              + " | max queue: " + str(self.max_q))


    def save(self, prefix="live"):
        if self.raw:
            sf.write(prefix + "_raw.wav", np.concatenate(self.raw), SR)
        if self.enh_log:
            sf.write(prefix + "_enh.wav", np.concatenate(self.enh_log), SR)
        print("saved " + prefix + "_raw.wav and " + prefix + "_enh.wav")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0,
                    help="auto-stop after N seconds (0 = run until stopped)")
    ap.add_argument("--win", type=float, default=0.5)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--block", type=int, default=320)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--in-dev", type=int, default=None)
    ap.add_argument("--out-dev", type=int, default=None)
    ap.add_argument("--bypass", action="store_true")
    ap.add_argument("--diar", action="store_true", help="speaker labels")
    ap.add_argument("--diar-src", choices=["raw", "enh"], default="raw")
    ap.add_argument("--assist", action="store_true",
                    help="ASR, emotion, spoken replies and dashboard")
    ap.add_argument("--wake", default="system", help="word that triggers a reply")
    ap.add_argument("--reply-all", action="store_true")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--asr-int8", action="store_true", help="faster quantized ASR")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mute", action="store_true", help="silence the output")
    args = ap.parse_args()


    import torch
    torch.set_num_threads(args.threads)
    import sounddevice as sd
    from enh_infer import enhance_fn


    for _ in range(3):
        enhance_fn(np.zeros(int(args.win * SR), dtype=np.float32))


    out_ch = min(2, sd.query_devices(args.out_dev, "output")["max_output_channels"])
    print("input : " + str(sd.query_devices(args.in_dev, "input")["name"]))
    print("output: " + str(sd.query_devices(args.out_dev, "output")["name"]))
    print("window " + str(args.win) + "s / hop " + str(args.hop)
          + "s | ENTER toggles ENHANCED/RAW | Ctrl+C stops")


    pipe = LivePipeline(enhance_fn, args.win, args.hop, args.block)
    pipe.bypass = args.bypass
    pipe.diar_raw = args.diar_src == "raw"


    stats = {}
    convo = None
    assistant = None
    ctrl = {"bypass": args.bypass, "reply_all": args.reply_all,
            "tts": not args.no_tts, "listening": True, "quit": False}


    if args.assist:
        from convo import Conversation
        from assistant import Assistant
        from asr_worker import AsrWorker
        from dashboard import start_dashboard
        convo = Conversation()
        assistant = Assistant(convo, wake=args.wake, reply_all=args.reply_all,
                              speak=not args.no_tts)
        convo.assistant = assistant
        assistant.start()
        asr = AsrWorker(ASR_MODEL, convo, int8=args.asr_int8)
        asr.start()
        pipe.asr = asr
        pipe.convo = convo
        start_dashboard(convo, stats, ctrl, port=args.port)
        trig = "every utterance" if args.reply_all else 'utterances containing "' + args.wake + '"'
        print("assistant ON (replies to " + trig + ") | dashboard http://localhost:"
              + str(args.port))


    dw = None
    if args.diar:
        from diar_worker import DiarWorker
        dw = DiarWorker()
        dw.convo = convo
        pipe.diar = dw
        print("diarization ON (" + args.diar_src + " audio, labels about 2 s after speech)")


    cnt = {"underruns": 0, "status": 0}
    started = [False]


    def callback(indata, outdata, frames, time_info, status):
        if status:
            cnt["status"] += 1
        if ctrl["listening"]:
            pipe.in_q.put((time.perf_counter(), indata[:, 0].copy()))
        y = pipe.out.pull(frames)
        if y is None:
            outdata.fill(0)
            if started[0] and ctrl["listening"]:
                cnt["underruns"] += 1
        else:
            started[0] = True
            outdata[:] = y[:, None]
            if args.mute:
                outdata.fill(0)


    def toggler():
        while True:
            try:
                input()
            except EOFError:
                return
            ctrl["bypass"] = not ctrl["bypass"]
            print("-> RAW mic" if ctrl["bypass"] else "-> ENHANCED")


    threading.Thread(target=toggler, daemon=True).start()
    pipe.start()
    t_end = time.time() + args.seconds
    try:
        with sd.Stream(samplerate=SR, blocksize=args.block,
                       device=(args.in_dev, args.out_dev), channels=(1, out_ch),
                       dtype="float32", callback=callback):
            last = 0
            while not ctrl["quit"] and (args.seconds <= 0 or time.time() < t_end):
                time.sleep(0.1)
                stats["underruns"] = cnt["underruns"]
                stats["rms"] = round(pipe.level, 4)
                pipe.bypass = ctrl["bypass"]
                if assistant is not None:
                    assistant.reply_all = ctrl["reply_all"]
                    assistant.speak_on = ctrl["tts"]
                if pipe.lat:
                    stats["latency_ms"] = round(pipe.lat[-1] * 1000)
                if pipe.enh.window_times:
                    stats["compute_ms"] = round(pipe.enh.window_times[-1] * 1000)
                if time.time() - last > 10:
                    last = time.time()
                    lat = format(pipe.lat[-1] * 1000, ".0f") + " ms" if pipe.lat else "-"
                    print(("RAW" if ctrl["bypass"] else "ENH") + " | latency " + lat
                          + " | queue " + str(pipe.in_q.qsize())
                          + " | underruns " + str(cnt["underruns"]))
    except KeyboardInterrupt:
        pass
    pipe.stop()
    if dw is not None:
        ev = dw.stop()
        print("\n===== SPEAKERS =====\nlabels " + str(len(ev))
              + " | speakers " + str(len(dw.d.sums))
              + " | dropped " + str(dw.dropped)
              + " | skipped " + str(dw.d.bad_norm))
    if convo is not None:
        print("names bound: " + str(convo.names))
    pipe.report(cnt["underruns"], cnt["status"])
    pipe.save()




if __name__ == "__main__":
    main()