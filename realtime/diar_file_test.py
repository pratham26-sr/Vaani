import glob, os, itertools
import numpy as np, soundfile as sf
from diar_stream import OnlineDiarizer, SR

CLEAN = r"C:\MAJOR\speech_enhancement_project\data\processed\vctk_16k\clean_testset_wav"
files = {k: sorted(glob.glob(os.path.join(CLEAN, f"{k}_*.wav"))) for k in ("p232", "p257")}
used = {"p232": 0, "p257": 0}
sil_u, sil_t = np.zeros(int(0.4 * SR), np.float32), np.zeros(int(0.6 * SR), np.float32)

turns = [("p232", 3), ("p257", 3), ("p232", 3), ("p257", 3)]
audio, truth, t = [], [], 0.0
for spk, n in turns:
    start = t
    for j in range(n):
        x, sr = sf.read(files[spk][used[spk]], dtype="float32"); used[spk] += 1
        audio += [x, sil_u]; t += (len(x) + len(sil_u)) / SR
    truth.append((start, t, spk))
    audio.append(sil_t); t += len(sil_t) / SR
audio = np.concatenate(audio)
print(f"conversation: {len(audio)/SR:.1f} s, {len(turns)} turns")

d = OnlineDiarizer()
events = []
for i in range(0, len(audio), 1600):
    events += d.feed(audio[i:i + 1600])
events += d.finish()

rows = []
print("\n| t0 | t1 | pred | sim | short | truth |")
print("|---|---|---|---|---|---|")
for e in events:
    mid = (e["t0"] + e["t1"]) / 2
    tr = next((s for a, b, s in truth if a <= mid < b), "?")
    rows.append((e["spk"], tr))
    print(f"| {e['t0']:.1f} | {e['t1']:.1f} | S{e['spk']} | {e['sim']:.2f} | {e['short']} | {tr} |")

preds = sorted({r[0] for r in rows})
best = 0
for perm in itertools.permutations(["p232", "p257", "?"], len(preds)):
    m = dict(zip(preds, perm))
    best = max(best, sum(m[p] == tr for p, tr in rows))
print(f"\nspeakers found: {len(preds)} | labelled windows: {len(rows)} | "
      f"accuracy: {best/len(rows):.3f} | collapsed embeddings skipped: {d.bad_norm}")