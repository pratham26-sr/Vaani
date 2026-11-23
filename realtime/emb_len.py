import glob, os
import numpy as np, soundfile as sf
import sherpa_onnx as s

MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
CLEAN = r"C:\MAJOR\speech_enhancement_project\data\processed\vctk_16k\clean_testset_wav"
SPK = ["p232", "p257"]
SR = 16000
LENS = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0]
N_FILES, N_WIN = 25, 40

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

print("| len s | raw norm mean | norm std | same mean | diff mean | acc (centered) |")
print("|---|---|---|---|---|---|")
for w in LENS:
    n = int(w * SR)
    E, lab = [], []
    for spk in SPK:
        x = audio[spk]
        starts = [i * SR for i in range(N_WIN) if i * SR + n <= len(x)]
        E += [embed(x[a:a + n]) for a in starts]
        lab += [spk] * len(starts)
    E, lab = np.stack(E), np.array(lab)
    norms = np.linalg.norm(E, axis=1)
    C = E - E.mean(0, keepdims=True)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    S = C @ C.T
    iu = np.triu_indices(len(C), 1)
    same = (lab[:, None] == lab[None, :])[iu]
    v = S[iu]
    acc = max(((v[same] > t).sum() + (v[~same] <= t).sum()) / len(v)
              for t in np.linspace(-0.5, 1, 301))
    En = E / (norms[:, None] + 1e-9)
    Sn = (En @ En.T)[iu]
    print(f"| {w} | {norms.mean():.2f} | {norms.std():.2f} | "
          f"{Sn[same].mean():.3f} | {Sn[~same].mean():.3f} | {acc:.3f} |")