import glob, os
import numpy as np, soundfile as sf
import sherpa_onnx as s

MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
CLEAN = r"C:\MAJOR\speech_enhancement_project\data\processed\vctk_16k\clean_testset_wav"
SPK = ["p232", "p257"]
N_FILES = 25
WINS = [1.0, 1.5, 2.0, 3.0, 5.0]
SR = 16000

ext = s.SpeakerEmbeddingExtractor(
    s.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=1))

def embed(x):
    st = ext.create_stream()
    st.accept_waveform(sample_rate=SR, waveform=x.astype(np.float32))
    st.input_finished()
    return np.array(ext.compute(st), dtype=np.float32)

audio = {}
for spk in SPK:
    fs = sorted(glob.glob(os.path.join(CLEAN, f"{spk}_*.wav")))[:N_FILES]
    audio[spk] = np.concatenate([sf.read(f, dtype="float32")[0] for f in fs])
    print(f"{spk}: {len(audio[spk])/SR:.0f} s")

def stats(E, lab, center):
    if center:
        E = E - E.mean(axis=0, keepdims=True)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    same = (lab[:, None] == lab[None, :])[iu]
    v = S[iu]
    sv, dv = v[same], v[~same]
    # best single threshold accuracy
    ths = np.linspace(-0.5, 1, 301)
    acc = max(((sv > t).sum() + (dv <= t).sum()) / len(v) for t in ths)
    return sv.mean(), dv.mean(), acc

print("\n| win s | n/spk | centered | same mean | diff mean | best-thr acc |")
print("|---|---|---|---|---|---|")
for w in WINS:
    n = int(w * SR)
    E, lab = [], []
    for spk in SPK:
        x = audio[spk]
        wins = [x[i:i + n] for i in range(0, len(x) - n, n)][:40]
        E += [embed(c) for c in wins]
        lab += [spk] * len(wins)
    E, lab = np.stack(E), np.array(lab)
    for c in (False, True):
        sm, dm, acc = stats(E, lab, c)
        print(f"| {w} | {len(lab)//2} | {c} | {sm:.3f} | {dm:.3f} | {acc:.3f} |")