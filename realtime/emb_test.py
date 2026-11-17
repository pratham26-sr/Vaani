import glob, os, sys
import numpy as np
import soundfile as sf
import sherpa_onnx as s

MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
CLEAN = r"C:\MAJOR\speech_enhancement_project\data\processed\vctk_16k\clean_testset_wav"
N_PER_SPK = 6          # utterances per speaker
SPEAKERS = ["p232", "p257"]

cfg = s.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=1)
assert cfg.validate(), "bad config"
ext = s.SpeakerEmbeddingExtractor(cfg)


def embed(wav_path):
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == 16000, f"{wav_path}: sr={sr}"
    st = ext.create_stream()
    st.accept_waveform(sample_rate=sr, waveform=x)
    st.input_finished()
    assert ext.is_ready(st), f"too short: {wav_path}"
    e = np.array(ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


embs, labels, names = [], [], []
for spk in SPEAKERS:
    files = sorted(glob.glob(os.path.join(CLEAN, f"{spk}_*.wav")))[:N_PER_SPK]
    if not files:
        sys.exit(f"No files found for {spk} in {CLEAN}")
    for f in files:
        embs.append(embed(f))
        labels.append(spk)
        names.append(os.path.basename(f))

E = np.stack(embs)
S = E @ E.T                      # cosine similarity (rows are unit-norm)
labels = np.array(labels)

same = np.zeros_like(S, dtype=bool)
for i in range(len(labels)):
    for j in range(len(labels)):
        same[i, j] = (labels[i] == labels[j]) and i != j
diff = labels[:, None] != labels[None, :]

print(f"\nFiles used: {len(names)}")
np.set_printoptions(precision=2, linewidth=200, suppress=True)
print("\nCosine similarity matrix:\n", S)
print("\n| Pair type | n | mean | min | max |")
print("|---|---|---|---|---|")
print(f"| same speaker | {same.sum()//2} | {S[same].mean():.3f} | {S[same].min():.3f} | {S[same].max():.3f} |")
print(f"| different speaker | {diff.sum()//2} | {S[diff].mean():.3f} | {S[diff].min():.3f} | {S[diff].max():.3f} |")
print(f"\nGap (min same - max diff): {S[same].min() - S[diff].max():.3f}")