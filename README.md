# Turkic-TTS 🎙️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![Languages: Turkic](https://img.shields.io/badge/languages-Turkic%20(Kazakh)-009688.svg)](#-supported-languages)

**High-quality Text-to-Speech system for Turkic languages, with a primary focus on Kazakh.**

> Open-source, multi-speaker, emotion-controllable TTS for Turkic languages — synthesize natural Kazakh speech across 3 voices and 6 emotions from plain text.

Turkic-TTS delivers natural-sounding speech synthesis powered by state-of-the-art neural architectures. Built on Grad-TTS with emotional voice synthesis capabilities, this system supports multiple speakers and emotions, making it ideal for applications ranging from voice assistants to audiobook narration.

## ✨ Features

- 🗣️ **Multi-speaker support** - Generate speech with different voice profiles (Female and Male speakers)
- 😊 **Emotional synthesis** - Control emotions: neutral, happy, sad, angry, scared, surprised
- 🎯 **High-quality output** - Neural vocoder (HiFi-GAN) produces 22.05kHz audio
- 🔧 **Flexible architecture** - Based on Grad-TTS with diffusion probabilistic modeling
- 🌍 **Turkic language support** - Optimized for Kazakh with IPA phonetic conversion for multiple Turkic languages
- ⚡ **Fast inference** - Adjustable timesteps for speed-quality trade-off

## 🏗️ Architecture Overview

The system consists of three main components:

1. **Text Encoder** - Converts text to phoneme embeddings with speaker/emotion conditioning
2. **Diffusion Decoder** - Grad-TTS based mel-spectrogram generator with emotion control
3. **Neural Vocoder** - HiFi-GAN converts mel-spectrograms to high-fidelity audio

### Key Technical Features:
- Monotonic Alignment Search (MAS) for text-audio alignment
- Classifier-free guidance for enhanced emotion control
- Exponential Moving Average (EMA) training for stability
- Support for both emotional and neutral speech synthesis

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/stukenov/turkic-tts.git
cd turkic-tts
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Build monotonic alignment module**
```bash
cd model/monotonic_align
python setup.py build_ext --inplace
cd ../..
```

## 🚀 Quick Start

### Download Pre-trained Models

Pre-trained models are available on HuggingFace:
- **Main repository**: [stukenov/turkic-tts-models](https://huggingface.co/stukenov/turkic-tts-models)

Download the models and place them in the appropriate directories:
- TTS model checkpoint → `pt_10000/`
- HiFi-GAN vocoder → `pre_trained_3/`

### Inference

Create a text file with your input (e.g., `filelists/my_text.txt`):
```
Сәлем, қалайсың?|0|1
```

Format: `text|emotion_id|speaker_id`

**Emotion IDs:**
- 0: Angry
- 1: Fear
- 2: Happy
- 3: Neutral
- 4: Sad
- 5: Surprised

**Speaker IDs:**
- 0: M1 (Male 1)
- 1: F1 (Female 1)
- 2: M2 (Male 2)

**Run inference:**
```bash
python inference_EMA.py \
  -c configs/train_grad.json \
  -m pt_10000/EMA_grad_10000.pt \
  -t 10 \
  -g 100 \
  -f filelists/my_text.txt \
  -r output_audio/
```

**Parameters:**
- `-c`: Configuration file
- `-m`: Model checkpoint path
- `-t`: Number of diffusion timesteps (higher = better quality, slower)
- `-g`: Classifier-free guidance level (recommended: 100)
- `-f`: Input text file
- `-r`: Output directory for audio files

### Example Usage

```python
# Coming soon: Python API example
```

### 🔊 Audio Samples

> **Note:** Pre-rendered audio samples are not yet bundled in this repository. To hear the model, download the pre-trained checkpoints (see above) and run the inference command — generated `.wav` files are written to your chosen output directory (e.g. `output_audio/`). Contributions of curated demo clips are welcome.

## 🏋️ Training

### Data Preparation

1. Organize your dataset with audio files and transcriptions
2. Create filelists in the format: `audio_path|speaker_id|emotion_id|text`

```bash
python data_preparation.py -d /path/to/your/dataset
```

### Training the Model

```bash
CUDA_VISIBLE_DEVICES=0 python train_EMA.py \
  -c configs/train_grad.json \
  -m logs/train_logs
```

The training script includes:
- Exponential Moving Average (EMA) for model stability
- Duration prediction with Monotonic Alignment Search
- Multi-speaker and emotion embedding
- Tensorboard logging

## 📊 Model Performance

The model has been trained on high-quality Kazakh speech data with:
- Multiple speakers and emotional expressions
- 80-dimensional mel-spectrograms
- 22.05kHz sampling rate
- Diffusion-based generation with controllable quality

## 🔧 Configuration

Key configuration options in `configs/train_grad.json`:

- **Model architecture**: encoder layers, channels, attention heads
- **Training parameters**: learning rate, batch size, optimization
- **Data processing**: sampling rate, hop length, mel bins
- **Speaker/emotion settings**: number of speakers, emotions, embedding dimensions

## 📁 Project Structure

```
turkic-tts/
├── model/               # Core model architectures
│   ├── tts.py          # Main Grad-TTS model
│   ├── diffusion.py    # Diffusion decoder
│   ├── text_encoder.py # Text encoding module
│   └── monotonic_align/ # MAS alignment
├── text/               # Text processing
│   ├── cleaners.py     # Text normalization
│   └── symbols.py      # Phoneme symbols
├── configs/            # Configuration files
├── inference_EMA.py    # Inference script
├── train_EMA.py        # Training script
├── models.py           # HiFi-GAN vocoder
├── ipa_convert.py      # IPA phonetic conversion
└── requirements.txt    # Dependencies
```

## 🌍 Supported Languages

While optimized for **Kazakh**, the IPA conversion module includes support for:
- Kazakh
- Turkish
- Kyrgyz
- Uzbek
- Azerbaijani
- Turkmen
- Tatar
- Bashkir
- Sakha (Yakut)
- Uyghur

## 🔗 Related projects

Turkic-TTS is part of a broader open-source Kazakh-language AI stack. If you're building speech and language tools for Kazakh, you may also find these useful:

- [kazakh-nlp-toolkit](https://github.com/stukenov/kazakh-nlp-toolkit) — Natural language processing utilities for Kazakh.
- [slm](https://github.com/stukenov/slm) — Small language model work for Kazakh / low-resource settings.
- [kazakh-speech-pipeline](https://github.com/stukenov/kazakh-speech-pipeline) — End-to-end speech data and processing pipeline for Kazakh.
- [qazlang](https://github.com/stukenov/qazlang) — Kazakh language resources and tooling.

## 🤝 Contributing

Contributions are welcome! Whether it's:
- Bug fixes
- New features
- Documentation improvements
- Training data contributions
- Support for additional Turkic languages

Please feel free to submit issues and pull requests.

## 📝 Citation

If you use this code or models in your research, please cite:

```bibtex
@software{turkic_tts_2024,
  author = {Tukenov, Saken},
  title = {Turkic-TTS: High-Quality Text-to-Speech for Turkic Languages},
  year = {2024},
  url = {https://github.com/stukenov/turkic-tts}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS) - Diffusion probabilistic TTS
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - Neural vocoder
- [KazEmoTTS](https://github.com/IS2AI/KazEmoTTS) - Kazakh emotional TTS dataset

## 📧 Contact

For questions, suggestions, or collaborations:
- GitHub: [@stukenov](https://github.com/stukenov)
- Issues: [Project Issues](https://github.com/stukenov/turkic-tts/issues)

---

**Star ⭐ this repository if you find it useful!**
