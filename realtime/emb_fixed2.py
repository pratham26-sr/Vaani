import glob, os
import numpy as np, soundfile as sf
import sherpa_onnx as s

MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
CLEAN = r"C:\MAJOR\speech_enhancement_project\data\processed\vctk_16k\clean_testset_wav"
SPK = ["p232", "p257"]
SR = 16000
N_FILES, N_WIN = 25, 40
rng = np.random.default_rng(0)

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

def report(tag, E, lab):
    norms = np.linalg.norm(E, axis=1)
    iu = np.triu_indices(len(E), 1)
    same = (lab[:, None] == lab[None, :])[iu]
    En = E / (norms[:, None] + 1e-9)
    Sn = (En @ En.T)[iu]
    C = E - E.mean(0, keepdims=True)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    v = (C @ C.T)[iu]
    acc = max(((v[same] > t).sum() + (v[~same] <= t).sum()) / len(v)
              for t in np.linspace(-0.5, 1, 301))
    print(f"| {tag} | {norms.mean():.1f} | {norms.std():.1f} | "
          f"{Sn[same].mean():.3f} | {Sn[~same].mean():.3f} | {acc:.3f} |")

print("| case | norm mean | norm std | same mean | diff mean | acc (centered) |")
print("|---|---|---|---|---|---|")

# A: random start offsets, lengths around 2.0 s
for sec in [1.9, 1.95, 2.0, 2.05, 2.1]:
    n = int(sec * SR)
    E, lab = [], []
    for spk in SPK:
        x = audio[spk]
        for a in rng.integers(0, len(x) - n, N_WIN):
            E.append(embed(x[a:a + n])); lab.append(spk)
    report(f"random start, {sec} s", np.stack(E), np.array(lab))

# B: short chunks tiled (repeated) up to exactly 2.0 s
N2 = 2 * SR
for sec in [0.75, 1.0, 1.5]:
    n = int(sec * SR)
    E, lab = [], []
    for spk in SPK:
        x = audio[spk]
        for a in rng.integers(0, len(x) - n, N_WIN):
            c = x[a:a + n]
            c = np.tile(c, int(np.ceil(N2 / n)))[:N2]
            E.append(embed(c)); lab.append(spk)
    report(f"{sec} s tiled to 2.0 s", np.stack(E), np.array(lab))