import glob, os, itertools
import numpy as np, soundfile as sf
from diar_stream import OnlineDiarizer, SR
from enh_infer import enhance_fn
from stream_enh import ChunkedEnhancer, stream_file

BASE = r"C:\MAJOR\speech_enhancement_project\data\processed\vctk_16k"
CLEAN = os.path.join(BASE, "clean_testset_wav")
NOISY = os.path.join(BASE, "noisy_testset_wav")

def enhance_audio(x):
    """Streaming enhancement, same settings as live_demo.py (0.5 s win / 0.25 s hop)."""
    enh = ChunkedEnhancer(enhance_fn, win_s=0.5, hop_s=0.25)
    return stream_file(enh, x)

def build(folder, names):
    sil_u = np.zeros(int(0.4 * SR), np.float32)
    sil_t = np.zeros(int(0.6 * SR), np.float32)
    turns = [("p232", 3), ("p257", 3), ("p232", 3), ("p257", 3)]
    used = {"p232": 0, "p257": 0}
    audio, truth, t = [], [], 0.0
    for spk, n in turns:
        start = t
        for _ in range(n):
            f = names[spk][used[spk]]; used[spk] += 1
            x, _ = sf.read(os.path.join(folder, f), dtype="float32")
            audio += [x, sil_u]; t += (len(x) + len(sil_u)) / SR
        truth.append((start, t, spk))
        audio.append(sil_t); t += len(sil_t) / SR
    return np.concatenate(audio), truth

def diarize(audio, truth):
    d = OnlineDiarizer()
    ev = []
    for i in range(0, len(audio), 1600):
        ev += d.feed(audio[i:i + 1600])
    ev += d.finish()
    rows = []
    for e in ev:
        mid = (e["t0"] + e["t1"]) / 2
        tr = next((s for a, b, s in truth if a <= mid < b), "?")
        rows.append((e["spk"], tr, e["sim"]))
    preds = sorted({r[0] for r in rows})
    best = 0
    for perm in itertools.permutations(["p232", "p257", "?"], len(preds)):
        m = dict(zip(preds, perm))
        best = max(best, sum(m[p] == tr for p, tr, _ in rows))
    return dict(acc=best / max(len(rows), 1), spk=len(preds), n=len(rows),
                skipped=d.bad_norm, sim=float(np.mean([r[2] for r in rows])))

# same utterances in clean and noisy folders (VCTK-DEMAND filenames match)
names = {k: sorted(os.path.basename(p) for p in glob.glob(os.path.join(CLEAN, f"{k}_*.wav")))
         for k in ("p232", "p257")}
results = {}
clean_a, truth = build(CLEAN, names)
noisy_a, _ = build(NOISY, names)
results["clean"] = diarize(clean_a, truth)
results["noisy"] = diarize(noisy_a, truth)
try:
    results["enhanced"] = diarize(enhance_audio(noisy_a).astype(np.float32), truth)
except NotImplementedError:
    print("enhanced: skipped (enhance_audio not implemented yet)")

print("\n| input | accuracy | speakers found | windows | skipped emb. | mean sim |")
print("|---|---|---|---|---|---|")
for k, r in results.items():
    print(f"| {k} | {r['acc']:.3f} | {r['spk']} | {r['n']} | {r['skipped']} | {r['sim']:.2f} |")

    # ---- diagnostic: why did enhanced collapse? ----
enh_a = enhance_audio(noisy_a).astype(np.float32)
sf.write("enh_test.wav", enh_a, SR)
rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
print(f"\nRMS noisy {rms(noisy_a):.3f} | enhanced {rms(enh_a):.3f}")
d = OnlineDiarizer(); ev = []
for i in range(0, len(enh_a), 1600):
    ev += d.feed(enh_a[i:i + 1600])
ev += d.finish()
print("| t0 | t1 | pred | sim | truth |")
print("|---|---|---|---|---|")
for e in ev:
    mid = (e["t0"] + e["t1"]) / 2
    tr = next((s for a, b, s in truth if a <= mid < b), "?")
    print(f"| {e['t0']:.1f} | {e['t1']:.1f} | S{e['spk']} | {e['sim']:.2f} | {tr} |")