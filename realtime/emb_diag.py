import glob, os
import numpy as np, soundfile as sf
import sherpa_onnx as s

MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
CLEAN = r"C:\MAJOR\speech_enhancement_project\data\processed\vctk_16k\clean_testset_wav"
cfg = s.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=1)
ext = s.SpeakerEmbeddingExtractor(cfg)

def embed_arr(x, sr=16000):
    st = ext.create_stream()
    st.accept_waveform(sample_rate=sr, waveform=x.astype(np.float32))
    st.input_finished()
    e = np.array(ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

print("| file | sr | dur s | rms | peak |")
print("|---|---|---|---|---|")
data = {}
for spk in ["p232", "p257"]:
    files = sorted(glob.glob(os.path.join(CLEAN, f"{spk}_*.wav")))
    print(f"# {spk}: {len(files)} files total")
    for f in files[:6]:
        x, sr = sf.read(f, dtype="float32")
        data[os.path.basename(f)] = x
        print(f"| {os.path.basename(f)} | {sr} | {len(x)/sr:.2f} | {np.sqrt((x**2).mean()):.3f} | {np.abs(x).max():.2f} |")

# Test A: first half vs second half of the SAME file
print("\nTest A: same file, first half vs second half (cosine)")
for name, x in data.items():
    h = len(x) // 2
    print(f"  {name}: {float(embed_arr(x[:h]) @ embed_arr(x[h:])):.3f}")

# Test B: concatenate 6 utterances per speaker (~20 s) -> speaker 'profile'
names = list(data)
prof = {}
for spk in ["p232", "p257"]:
    x = np.concatenate([data[n] for n in names if n.startswith(spk)])
    prof[spk] = embed_arr(x)
    print(f"\n{spk}: concatenated {len(x)/16000:.1f} s")
print("\nTest B: profile p232 vs p257:", f"{float(prof['p232'] @ prof['p257']):.3f}")