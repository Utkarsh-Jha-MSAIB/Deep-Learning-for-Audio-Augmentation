# Expressive Audio Augmentation with Multi-Instrument Neural Architectures

**A deep learning framework for sound-conditioned musical creativity**

## Overview

This project explores deep learning approaches for expressive audio augmentation, shifting from pure text-to-music generation toward sound-conditioned creative transformation.
Instead of generating music from text alone, the system listens to audio, extracts expressive characteristics, and produces musically coherent variations grounded in real sonic input.

The system integrates three key technical directions:
- **Multi-Instrument Neural-DSP Pipeline** (audio-conditioned RNN + Transformer)
  - Each instrument has its own Instrument RNN (NeuralSynthesizer)
  - DDSP modules provide realistic harmonic & noise modeling
  - Transformer Arranger predicts follower-instrument dynamics based on a leader signal
  - Ensemble arrangement (bass + drums + guitar + etc.)

- **Audio Retrieval-Augmented Generation (RAG)** (for extension, coherence, and style grounding)
  - Large feature database of audio chunks
  - Retrieval based on similarity (chroma, energy, coherence)
  - Weighted scoring: relevance + seam smoothness
  - Entire multi-instrument pipeline can generate extended Top-K continuations

📂 **Datasets**

1. **Essen Folksong (KRN)**: Symbolic-only dataset for early pipeline validation.

2. **BabySlakh**: Small multitrack dataset for debugging and small-scale decoder training.

3. **LSX**: Large-scale (~28 GB) multitrack dataset enabling
  - Multi-instrument arrangement
  - Temporal synchronization
  - Realistic ensemble modeling

4. **URMP**: Classical multi-instrument stems; used for acoustic timbre modeling.

# Pipeline

| Script | Description |
|--------|-------------|
| src/data/converters.py | KRN → MIDI conversion and symbolic normalization. |
| src/data/download_slakh.py | Download BabySlakh dataset. |
| src/data/download_urmp.py | Download URMP dataset. |
| src/data/extraction.py | Extract f0, loudness, and frame-level features. |
| src/data/loader.py | Dataset loading utilities. |
| src/data/preprocess.py | Windowing, normalization, feature preparation. |
| src/data/process_band.py | Multi-instrument alignment (LSX stems). |
| src/models/audio_RAG.py | Audio retrieval-augmented generation logic. |
| src/models/decoder_conductor.py | Transformer arranger predicting follower loudness. |
| src/models/decoder_instrument.py | Instrument-specific GRU + DDSP synthesizer. |
| src/models/perform_music.py | End-to-end generation + multi-instrument synthesis. |
| src/models/signal_processing.py | Harmonic & noise DDSP utilities. |
| src/models/train_conductor.py | Training loop for the Transformer arranger. |
| src/models/train_instrument.py | Training loop for instrument decoders. |
| src/visualization/song_in_detail.py | Mel plots, loudness curves, waveform diagnostics. |

