"""
Phase 1 Setup Script — AI Speech Enhancement Project
=====================================================
Run this script to:
  1. Create the full project directory structure
  2. Print download instructions for each dataset
  3. Install all required Python packages
  4. Verify your setup

Usage:
  python setup_project.py          # Create dirs + print instructions
  python setup_project.py --install # Also install pip packages
  python setup_project.py --verify  # Verify everything is in place
"""

import os
import sys
import argparse
import subprocess

# ============================================================
# PROJECT DIRECTORY STRUCTURE
# ============================================================
PROJECT_ROOT = "speech_enhancement_project"

DIRS = [
    # ---- Data directories ----
    "data/raw/vctk_demand/clean_trainset_28spk_wav",
    "data/raw/vctk_demand/noisy_trainset_28spk_wav",
    "data/raw/vctk_demand/clean_testset_wav",
    "data/raw/vctk_demand/noisy_testset_wav",

    "data/raw/ravdess",            # emotion recognition audio files
    "data/raw/crema_d/AudioWAV",   # emotion recognition (diverse speakers)
    "data/raw/librispeech/test-clean",  # ASR benchmarking

    "data/processed/spectrograms",      # mel/STFT spectrograms
    "data/processed/features",          # extracted MFCCs, pitch, energy
    "data/processed/enhanced",          # output from enhancement model

    # ---- Source code ----
    "src/enhancement",     # speech enhancement model code
    "src/diarization",     # speaker diarization module
    "src/emotion",         # emotion recognition module
    "src/narrative",       # ASR + summarization
    "src/utils",           # shared utilities (audio I/O, metrics, etc.)
    "src/dashboard",       # Streamlit/Gradio web app

    # ---- Notebooks for exploration ----
    "notebooks",

    # ---- Trained model weights ----
    "models/enhancement",
    "models/emotion",

    # ---- Results & reports ----
    "results/metrics",
    "results/audio_samples",
    "results/plots",

    # ---- Configs ----
    "configs",

    # ---- Documentation ----
    "docs",
]

# ============================================================
# DATASET INFO — LINKS & INSTRUCTIONS
# ============================================================
DATASETS = {
    "VCTK-DEMAND (Speech Enhancement — PRIMARY)": {
        "description": (
            "Clean + noisy parallel speech pairs. 28 speakers train, 2 speakers test.\n"
            "  10 noise types × 4 SNR levels (0, 5, 10, 15 dB) for training.\n"
            "  5 unseen noise types × 4 SNR levels (2.5, 7.5, 12.5, 17.5 dB) for testing."
        ),
        "size": "~11 GB",
        "download_url": "https://datashare.ed.ac.uk/handle/10283/2791",
        "alt_url": "https://doi.org/10.7488/ds/2117",
        "steps": [
            "1. Go to: https://datashare.ed.ac.uk/handle/10283/2791",
            "2. Click 'Download All' to get the full ZIP (~11 GB)",
            "3. You will get 4 folders after extraction:",
            "   - clean_trainset_28spk_wav/   (11,572 clean .wav files)",
            "   - noisy_trainset_28spk_wav/   (11,572 noisy .wav files)",
            "   - clean_testset_wav/           (872 clean .wav files)",
            "   - noisy_testset_wav/           (872 noisy .wav files)",
            "4. Move all 4 folders into: data/raw/vctk_demand/",
            "5. All audio is 48kHz — we will downsample to 16kHz in preprocessing",
        ],
        "target_dir": "data/raw/vctk_demand/",
    },

    "RAVDESS (Emotion Recognition — PRIMARY)": {
        "description": (
            "24 professional actors (12M, 12F), 8 emotions, 2 intensity levels.\n"
            "  Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised.\n"
            "  Audio-only files are 16-bit 48kHz WAV."
        ),
        "size": "~1.2 GB (audio-only subset via Kaggle) or ~24.8 GB (full AV from Zenodo)",
        "download_url_kaggle": "https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio",
        "download_url_zenodo": "https://zenodo.org/records/1188976",
        "steps": [
            "OPTION A — Kaggle (RECOMMENDED, audio-only, ~1.2 GB):",
            "  1. Go to: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio",
            "  2. Click 'Download' (needs free Kaggle account)",
            "  3. OR use Kaggle CLI:",
            "     pip install kaggle",
            "     kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio",
            "  4. Extract into: data/raw/ravdess/",
            "     You'll get folders: Actor_01/ through Actor_24/",
            "",
            "OPTION B — Zenodo (full AV, ~24.8 GB):",
            "  1. Go to: https://zenodo.org/records/1188976",
            "  2. Download the Audio_Speech_Actors_01-24.zip files",
            "  3. Extract into: data/raw/ravdess/",
            "",
            "FILENAME FORMAT: 03-01-06-01-02-01-12.wav",
            "  Positions: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor",
            "  Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad,",
            "                 05=angry, 06=fearful, 07=disgust, 08=surprised",
        ],
        "target_dir": "data/raw/ravdess/",
    },

    "CREMA-D (Emotion Recognition — SECONDARY)": {
        "description": (
            "91 actors (48M, 43F), ages 20-74, diverse ethnicities.\n"
            "  6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad.\n"
            "  4 intensity levels. 7,442 clips total."
        ),
        "size": "~5 GB",
        "download_url": "https://github.com/CheyneyComputerScience/CREMA-D",
        "steps": [
            "1. Install Git LFS first:",
            "   git lfs install",
            "2. Clone the repo (includes audio via Git LFS):",
            "   git clone https://github.com/CheyneyComputerScience/CREMA-D.git",
            "3. The audio files are in: CREMA-D/AudioWAV/",
            "4. Move or symlink AudioWAV/ into: data/raw/crema_d/AudioWAV/",
            "",
            "ALTERNATIVE — Kaggle:",
            "   https://www.kaggle.com/datasets/ejlok1/cremad",
            "",
            "FILENAME FORMAT: 1001_DFA_ANG_XX.wav",
            "  1001 = Actor ID, DFA = Sentence, ANG = Emotion, XX = Intensity",
        ],
        "target_dir": "data/raw/crema_d/",
    },

    "LibriSpeech test-clean (ASR Benchmarking)": {
        "description": (
            "Clean read English speech for testing ASR (Whisper).\n"
            "  We only need the test-clean subset for evaluation."
        ),
        "size": "~346 MB (test-clean only)",
        "download_url": "https://www.openslr.org/12",
        "steps": [
            "1. Download ONLY test-clean:",
            "   wget https://www.openslr.org/resources/12/test-clean.tar.gz",
            "2. Extract:",
            "   tar -xzf test-clean.tar.gz",
            "3. Move the LibriSpeech/test-clean/ folder to:",
            "   data/raw/librispeech/test-clean/",
            "",
            "OR use torchaudio (automatic download):",
            "   import torchaudio",
            "   dataset = torchaudio.datasets.LIBRISPEECH(",
            "       root='data/raw/librispeech',",
            "       url='test-clean',",
            "       download=True",
            "   )",
        ],
        "target_dir": "data/raw/librispeech/",
    },

    "Microsoft DNS Challenge (OPTIONAL — Advanced)": {
        "description": (
            "500+ hours of clean speech, 180+ hours of noise.\n"
            "  Use only if you want to train on a larger, more diverse dataset.\n"
            "  Includes scripts to synthesize your own noisy-clean pairs."
        ),
        "size": "~50 GB+ (full), you can use a subset",
        "download_url": "https://github.com/microsoft/DNS-Challenge",
        "steps": [
            "1. git lfs install",
            "2. git clone https://github.com/microsoft/DNS-Challenge.git",
            "3. Run the download script inside the repo:",
            "   bash download-dns-challenge-5-headset-training.sh",
            "4. This is OPTIONAL — start with VCTK-DEMAND first",
        ],
        "target_dir": "data/raw/dns_challenge/",
    },
}


# ============================================================
# PYTHON PACKAGES
# ============================================================
REQUIREMENTS = """# Core
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.1

# Speech enhancement metrics
pesq>=0.0.4
pystoi>=0.3.3

# Emotion recognition features
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# ASR (Whisper)
openai-whisper>=20231117

# Speaker diarization
pyannote.audio>=3.1.0

# Summarization
transformers>=4.30.0

# Dashboard
streamlit>=1.28.0
gradio>=4.0.0

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0

# Utilities
tqdm>=4.65.0
pandas>=2.0.0
pyyaml>=6.0.0
"""


def create_directories():
    """Create the full project directory tree."""
    print(f"\n{'='*60}")
    print(f"  CREATING PROJECT: {PROJECT_ROOT}/")
    print(f"{'='*60}\n")

    for d in DIRS:
        path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(path, exist_ok=True)
        print(f"  [+] {d}/")

    # Create __init__.py files in src packages
    src_dirs = [d for d in DIRS if d.startswith("src/")]
    for d in src_dirs:
        init_path = os.path.join(PROJECT_ROOT, d, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")
    # Also create root src/__init__.py
    init_path = os.path.join(PROJECT_ROOT, "src", "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    # Write requirements.txt
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(REQUIREMENTS)
    print(f"\n  [+] requirements.txt")

    # Write a basic .gitignore
    gitignore = os.path.join(PROJECT_ROOT, ".gitignore")
    with open(gitignore, "w") as f:
        f.write(
            "data/raw/\n"
            "models/\n"
            "*.pyc\n"
            "__pycache__/\n"
            ".ipynb_checkpoints/\n"
            "*.wav\n"
            "*.mp3\n"
            "*.pt\n"
            "*.pth\n"
            "*.h5\n"
            ".env\n"
        )
    print(f"  [+] .gitignore")

    # Write a starter config
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(
            "# Project Configuration\n\n"
            "data:\n"
            "  vctk_demand:\n"
            "    clean_train: data/raw/vctk_demand/clean_trainset_28spk_wav\n"
            "    noisy_train: data/raw/vctk_demand/noisy_trainset_28spk_wav\n"
            "    clean_test: data/raw/vctk_demand/clean_testset_wav\n"
            "    noisy_test: data/raw/vctk_demand/noisy_testset_wav\n"
            "  ravdess: data/raw/ravdess\n"
            "  crema_d: data/raw/crema_d/AudioWAV\n"
            "  librispeech: data/raw/librispeech/test-clean\n\n"
            "audio:\n"
            "  sample_rate: 16000\n"
            "  segment_length: 4.0  # seconds\n"
            "  n_fft: 512\n"
            "  hop_length: 256\n"
            "  n_mels: 80\n\n"
            "training:\n"
            "  batch_size: 16\n"
            "  learning_rate: 0.001\n"
            "  epochs: 50\n"
            "  device: auto  # auto, cpu, cuda, mps\n"
        )
    print(f"  [+] configs/config.yaml")

    print(f"\n  Done! Project tree created at ./{PROJECT_ROOT}/\n")


def print_dataset_instructions():
    """Print download links and steps for each dataset."""
    print(f"\n{'='*60}")
    print(f"  DATASET DOWNLOAD INSTRUCTIONS")
    print(f"{'='*60}")

    for name, info in DATASETS.items():
        print(f"\n{'─'*60}")
        print(f"  📦 {name}")
        print(f"{'─'*60}")
        print(f"  {info['description']}")
        print(f"\n  Size: {info['size']}")
        print(f"  Target: {PROJECT_ROOT}/{info['target_dir']}")

        if "download_url" in info:
            print(f"  URL:  {info['download_url']}")
        if "download_url_kaggle" in info:
            print(f"  Kaggle: {info['download_url_kaggle']}")
        if "download_url_zenodo" in info:
            print(f"  Zenodo: {info['download_url_zenodo']}")
        if "alt_url" in info:
            print(f"  Alt:  {info['alt_url']}")

        print(f"\n  Steps:")
        for step in info["steps"]:
            print(f"    {step}")

    print(f"\n{'─'*60}")
    print("  PRIORITY ORDER:")
    print("    1. VCTK-DEMAND  — download FIRST (core of the project)")
    print("    2. RAVDESS      — download SECOND (for emotion module)")
    print("    3. LibriSpeech  — small, quick download for ASR testing")
    print("    4. CREMA-D      — optional, adds speaker diversity")
    print("    5. DNS Challenge — optional, only if you need more data")
    print(f"{'─'*60}\n")


def install_packages():
    """Install Python dependencies."""
    print(f"\n{'='*60}")
    print(f"  INSTALLING PYTHON PACKAGES")
    print(f"{'='*60}\n")

    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.exists(req_path):
        print("  [!] requirements.txt not found. Run without --install first.")
        return

    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("\n  Done! All packages installed.\n")


def verify_setup():
    """Check which datasets are present and which packages are installed."""
    print(f"\n{'='*60}")
    print(f"  VERIFYING SETUP")
    print(f"{'='*60}\n")

    # Check directories
    print("  📁 Directory structure:")
    for d in DIRS[:8]:  # Check key data dirs
        path = os.path.join(PROJECT_ROOT, d)
        files = []
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(('.wav', '.flac', '.mp3'))]
        status = f"✅ {len(files)} audio files" if files else "⬜ empty"
        print(f"    {d}: {status}")

    # Check packages
    print("\n  📦 Key packages:")
    packages = [
        "torch", "torchaudio", "librosa", "soundfile",
        "pesq", "pystoi", "whisper", "pyannote.audio",
        "sklearn", "streamlit", "transformers"
    ]
    for pkg in packages:
        try:
            mod = pkg.replace(".", "_").replace("-", "_")
            if mod == "sklearn":
                __import__("sklearn")
            elif mod == "whisper":
                __import__("whisper")
            elif mod == "pyannote_audio":
                __import__("pyannote.audio")
            else:
                __import__(mod)
            print(f"    ✅ {pkg}")
        except ImportError:
            print(f"    ❌ {pkg} — not installed")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Project Setup")
    parser.add_argument("--install", action="store_true", help="Install pip packages")
    parser.add_argument("--verify", action="store_true", help="Verify setup completeness")
    args = parser.parse_args()

    create_directories()
    print_dataset_instructions()

    if args.install:
        install_packages()

    if args.verify:
        verify_setup()

    if not args.install and not args.verify:
        print("  Next steps:")
        print("    1. Download the datasets using the links above")
        print("    2. Run: python setup_project.py --install")
        print("    3. Run: python setup_project.py --verify")
        print()