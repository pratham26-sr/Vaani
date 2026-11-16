"""
Phase 1 — Data Preprocessing & Baseline Metrics
=================================================
Run AFTER downloading datasets and running setup_project.py.

This script:
  1. Resamples all VCTK-DEMAND audio from 48kHz → 16kHz
  2. Computes baseline metrics on noisy test set (PESQ, STOI, SNR)
  3. Generates spectrograms for visual comparison
  4. Parses RAVDESS filenames into an emotion label CSV
  5. Prints a full data inventory report

Usage:
  cd speech_enhancement_project/
  python preprocess_phase1.py
"""

import os
import glob
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import soundfile as sf


def _load_audio(path):
    """Load a wav with soundfile -> (tensor [channels, time], sample_rate)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _save_audio(path, waveform, sr):
    """Save a [channels, time] tensor as a wav with soundfile."""
    sf.write(path, waveform.numpy().T, sr)


warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = "."  # run from inside speech_enhancement_project/
TARGET_SR = 16000

VCTK_DIR       = os.path.join(PROJECT_ROOT, "data/raw/vctk_demand")
RAVDESS_DIR    = os.path.join(PROJECT_ROOT, "data/raw/ravdess")
CREMA_DIR      = os.path.join(PROJECT_ROOT, "data/raw/crema_d/AudioWAV")
PROCESSED_DIR  = os.path.join(PROJECT_ROOT, "data/processed")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results/metrics")

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

CREMA_EMOTIONS = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad"
}


# ============================================================
# 1. RESAMPLE VCTK-DEMAND AUDIO (48kHz → 16kHz)
# ============================================================
def resample_vctk_demand():
    """Resample all VCTK-DEMAND wav files from 48kHz to 16kHz in-place."""
    print("\n" + "=" * 60)
    print("  STEP 1: Resampling VCTK-DEMAND to 16kHz")
    print("=" * 60)

    subfolders = [
        "clean_trainset_28spk_wav",
        "noisy_trainset_28spk_wav",
        "clean_testset_wav",
        "noisy_testset_wav",
    ]

    for folder in subfolders:
        src_dir = os.path.join(VCTK_DIR, folder)
        out_dir = os.path.join(PROCESSED_DIR, "vctk_16k", folder)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"  [!] Skipping {folder} — not found at {src_dir}")
            continue

        wav_files = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
        if not wav_files:
            print(f"  [!] No .wav files in {src_dir}")
            continue

        already_done = len(glob.glob(os.path.join(out_dir, "*.wav")))
        if already_done == len(wav_files):
            print(f"  [✓] {folder}: already resampled ({already_done} files)")
            continue

        print(f"  Resampling {folder}: {len(wav_files)} files...")
        resampler = None

        for f in tqdm(wav_files, desc=f"    {folder}", ncols=80):
            waveform, sr = _load_audio(f)
            if sr != TARGET_SR:
                if resampler is None or resampler.orig_freq != sr:
                    resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)
            out_path = os.path.join(out_dir, os.path.basename(f))
            _save_audio(out_path, waveform, TARGET_SR)

        print(f"  [✓] {folder}: {len(wav_files)} files → {out_dir}")


# ============================================================
# 2. COMPUTE BASELINE METRICS ON NOISY TEST SET
# ============================================================
def compute_baseline_metrics():
    """Compute PESQ, STOI, and SNR on the noisy test set vs clean reference."""
    print("\n" + "=" * 60)
    print("  STEP 2: Computing Baseline Metrics (noisy test set)")
    print("=" * 60)

    clean_dir = os.path.join(PROCESSED_DIR, "vctk_16k", "clean_testset_wav")
    noisy_dir = os.path.join(PROCESSED_DIR, "vctk_16k", "noisy_testset_wav")

    # Fall back to raw dirs if 16k not yet created
    if not os.path.exists(clean_dir):
        clean_dir = os.path.join(VCTK_DIR, "clean_testset_wav")
        noisy_dir = os.path.join(VCTK_DIR, "noisy_testset_wav")

    if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir):
        print("  [!] Test set not found. Download VCTK-DEMAND first.")
        return

    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))

    if not clean_files:
        print("  [!] No clean test files found.")
        return

    # Try importing metric libraries
    try:
        from pesq import pesq
    except ImportError:
        print("  [!] 'pesq' not installed. Run: pip install pesq")
        pesq = None

    try:
        from pystoi import stoi
    except ImportError:
        print("  [!] 'pystoi' not installed. Run: pip install pystoi")
        stoi = None

    def compute_snr(clean, noisy):
        noise = noisy - clean
        snr_val = 10 * np.log10(
            np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-10)
        )
        return snr_val

    print(f"  Found {len(clean_files)} clean / {len(noisy_files)} noisy test files.")
    print(f"  Computing metrics on first 100 pairs (for speed)...\n")

    n = min(100, len(clean_files))
    pesq_scores, stoi_scores, snr_scores = [], [], []

    for i in tqdm(range(n), desc="    Metrics", ncols=80):
        c_wav, c_sr = _load_audio(clean_files[i])
        n_wav, n_sr = _load_audio(noisy_files[i])

        c_np = c_wav.squeeze().numpy()
        n_np = n_wav.squeeze().numpy()

        # Ensure same length
        min_len = min(len(c_np), len(n_np))
        c_np, n_np = c_np[:min_len], n_np[:min_len]

        sr = c_sr  # should be 16000 if resampled

        # SNR (always works)
        snr_scores.append(compute_snr(c_np, n_np))

        # PESQ
        if pesq is not None:
            try:
                score = pesq(sr, c_np, n_np, "wb" if sr >= 16000 else "nb")
                pesq_scores.append(score)
            except Exception:
                pass

        # STOI
        if stoi is not None:
            try:
                score = stoi(c_np, n_np, sr, extended=False)
                stoi_scores.append(score)
            except Exception:
                pass

    # Print results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n  {'─' * 45}")
    print(f"  BASELINE METRICS (noisy test set, n={n})")
    print(f"  {'─' * 45}")

    results = {"n_samples": n}

    if snr_scores:
        avg = np.mean(snr_scores)
        std = np.std(snr_scores)
        print(f"  SNR:   {avg:.2f} ± {std:.2f} dB")
        results["snr_mean"] = round(avg, 3)
        results["snr_std"] = round(std, 3)

    if pesq_scores:
        avg = np.mean(pesq_scores)
        std = np.std(pesq_scores)
        print(f"  PESQ:  {avg:.3f} ± {std:.3f}  (range: 1.0 to 4.5)")
        results["pesq_mean"] = round(avg, 3)
        results["pesq_std"] = round(std, 3)
    else:
        print(f"  PESQ:  not computed (install pesq package)")

    if stoi_scores:
        avg = np.mean(stoi_scores)
        std = np.std(stoi_scores)
        print(f"  STOI:  {avg:.3f} ± {std:.3f}  (range: 0.0 to 1.0)")
        results["stoi_mean"] = round(avg, 3)
        results["stoi_std"] = round(std, 3)
    else:
        print(f"  STOI:  not computed (install pystoi package)")

    print(f"  {'─' * 45}")

    # Save to JSON
    out_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  [✓] Saved to {out_path}")


# ============================================================
# 3. GENERATE SPECTROGRAM COMPARISONS
# ============================================================
def generate_spectrograms():
    """Generate side-by-side spectrograms of clean vs noisy audio."""
    print("\n" + "=" * 60)
    print("  STEP 3: Generating Spectrogram Comparisons")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not installed. Skipping spectrogram generation.")
        return

    clean_dir = os.path.join(PROCESSED_DIR, "vctk_16k", "clean_testset_wav")
    noisy_dir = os.path.join(PROCESSED_DIR, "vctk_16k", "noisy_testset_wav")

    if not os.path.exists(clean_dir):
        clean_dir = os.path.join(VCTK_DIR, "clean_testset_wav")
        noisy_dir = os.path.join(VCTK_DIR, "noisy_testset_wav")

    if not os.path.exists(clean_dir):
        print("  [!] Test data not found. Skipping.")
        return

    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))[:5]
    plot_dir = os.path.join(PROJECT_ROOT, "results/plots")
    os.makedirs(plot_dir, exist_ok=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=512, hop_length=256, n_mels=80
    )
    amp_to_db = T.AmplitudeToDB()

    for cf in clean_files:
        fname = os.path.basename(cf)
        nf = os.path.join(noisy_dir, fname)
        if not os.path.exists(nf):
            continue

        c_wav, _ = _load_audio(cf)
        n_wav, _ = _load_audio(nf)

        c_mel = amp_to_db(mel_transform(c_wav)).squeeze().numpy()
        n_mel = amp_to_db(mel_transform(n_wav)).squeeze().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].imshow(c_mel, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title(f"Clean: {fname}")
        axes[0].set_ylabel("Mel bin")
        axes[0].set_xlabel("Frame")

        axes[1].imshow(n_mel, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title(f"Noisy: {fname}")
        axes[1].set_xlabel("Frame")

        plt.tight_layout()
        out_path = os.path.join(plot_dir, f"spectrogram_{fname.replace('.wav', '.png')}")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  [✓] {out_path}")

    print(f"  Done! Check results/plots/ for spectrograms.")


# ============================================================
# 4. PARSE RAVDESS INTO EMOTION LABELS CSV
# ============================================================
def parse_ravdess_labels():
    """Parse RAVDESS filenames into a CSV with columns: file, actor, emotion, intensity."""
    print("\n" + "=" * 60)
    print("  STEP 4: Parsing RAVDESS Emotion Labels")
    print("=" * 60)

    if not os.path.exists(RAVDESS_DIR):
        print(f"  [!] RAVDESS not found at {RAVDESS_DIR}. Skipping.")
        return

    wav_files = sorted(glob.glob(os.path.join(RAVDESS_DIR, "**", "*.wav"), recursive=True))
    if not wav_files:
        print(f"  [!] No .wav files in {RAVDESS_DIR}")
        return

    rows = []
    for f in wav_files:
        fname = os.path.basename(f)
        parts = fname.replace(".wav", "").split("-")
        if len(parts) < 7:
            continue

        modality, vocal_ch, emotion, intensity, statement, rep, actor = parts
        rows.append({
            "file": f,
            "filename": fname,
            "actor": int(actor),
            "emotion_code": emotion,
            "emotion": RAVDESS_EMOTIONS.get(emotion, "unknown"),
            "intensity": "normal" if intensity == "01" else "strong",
            "vocal_channel": "speech" if vocal_ch == "01" else "song",
        })

    os.makedirs(os.path.join(PROCESSED_DIR, "features"), exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "features", "ravdess_labels.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Print distribution
    from collections import Counter
    emo_counts = Counter(r["emotion"] for r in rows)
    print(f"  Parsed {len(rows)} files:")
    for emo, cnt in sorted(emo_counts.items(), key=lambda x: -x[1]):
        print(f"    {emo:12s}: {cnt:4d} samples")
    print(f"  [✓] Labels saved to {out_path}")


# ============================================================
# 5. PARSE CREMA-D INTO EMOTION LABELS CSV
# ============================================================
def parse_cremad_labels():
    """Parse CREMA-D filenames into a labeled CSV."""
    print("\n" + "=" * 60)
    print("  STEP 5: Parsing CREMA-D Emotion Labels")
    print("=" * 60)

    if not os.path.exists(CREMA_DIR):
        print(f"  [!] CREMA-D not found at {CREMA_DIR}. Skipping.")
        return

    wav_files = sorted(glob.glob(os.path.join(CREMA_DIR, "*.wav")))
    if not wav_files:
        print(f"  [!] No .wav files in {CREMA_DIR}")
        return

    rows = []
    for f in wav_files:
        fname = os.path.basename(f)
        parts = fname.replace(".wav", "").split("_")
        if len(parts) < 4:
            continue

        actor_id, sentence, emo_code, intensity = parts[0], parts[1], parts[2], parts[3]
        rows.append({
            "file": f,
            "filename": fname,
            "actor": actor_id,
            "sentence": sentence,
            "emotion_code": emo_code,
            "emotion": CREMA_EMOTIONS.get(emo_code, "unknown"),
            "intensity": intensity,
        })

    os.makedirs(os.path.join(PROCESSED_DIR, "features"), exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "features", "cremad_labels.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    emo_counts = Counter(r["emotion"] for r in rows)
    print(f"  Parsed {len(rows)} files:")
    for emo, cnt in sorted(emo_counts.items(), key=lambda x: -x[1]):
        print(f"    {emo:12s}: {cnt:4d} samples")
    print(f"  [✓] Labels saved to {out_path}")


# ============================================================
# 6. DATA INVENTORY REPORT
# ============================================================
def data_inventory():
    """Print a summary of what's downloaded and ready."""
    print("\n" + "=" * 60)
    print("  DATA INVENTORY REPORT")
    print("=" * 60)

    checks = [
        ("VCTK-DEMAND (train clean)", os.path.join(VCTK_DIR, "clean_trainset_28spk_wav"), "*.wav"),
        ("VCTK-DEMAND (train noisy)", os.path.join(VCTK_DIR, "noisy_trainset_28spk_wav"), "*.wav"),
        ("VCTK-DEMAND (test clean)",  os.path.join(VCTK_DIR, "clean_testset_wav"), "*.wav"),
        ("VCTK-DEMAND (test noisy)",  os.path.join(VCTK_DIR, "noisy_testset_wav"), "*.wav"),
        ("RAVDESS",                   RAVDESS_DIR, "**/*.wav"),
        ("CREMA-D",                   CREMA_DIR, "*.wav"),
        ("LibriSpeech test-clean",    os.path.join(PROJECT_ROOT, "data/raw/librispeech"), "**/*.flac"),
    ]

    total_files = 0
    for name, path, pattern in checks:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, pattern), recursive=True)
            count = len(files)
            total_files += count
            status = f"✅ {count:>6,d} files"
        else:
            status = "❌ not found"
        print(f"  {name:35s} {status}")

    print(f"\n  Total audio files: {total_files:,d}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "🔊 " * 20)
    print("  PHASE 1 — DATA PREPROCESSING & BASELINE METRICS")
    print("🔊 " * 20)

    # Always run inventory first
    data_inventory()

    # Run each step (skips gracefully if data missing)
    resample_vctk_demand()
    compute_baseline_metrics()
    generate_spectrograms()
    parse_ravdess_labels()
    parse_cremad_labels()

    print("\n" + "=" * 60)
    print("  ✅ PHASE 1 COMPLETE!")
    print("=" * 60)
    print("  What you now have:")
    print("    • Resampled audio at 16kHz")
    print("    • Baseline PESQ/STOI/SNR metrics (your 'before' numbers)")
    print("    • Clean vs noisy spectrogram comparisons")
    print("    • RAVDESS + CREMA-D emotion label CSVs")
    print("\n  Next: Phase 2 — Train the speech enhancement model!")
    print("=" * 60 + "\n")