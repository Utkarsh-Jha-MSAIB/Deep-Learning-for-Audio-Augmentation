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
 

<p align="center">
  <img src="https://github.com/user-attachments/assets/e91e9aa9-8a7d-4168-b35e-d09d205021ed" width="300"/>
</p>
     


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

# Pipeline: a modular coding structure, with following desccription of main scripts in the process

| Script | Description |
|--------|-------------|
| src/models/audio_RAG.py | Audio retrieval-augmented generation logic. |
| src/models/decoder_conductor.py | Transformer arranger predicting follower loudness. |
| src/models/decoder_instrument.py | Instrument-specific GRU + DDSP synthesizer. |
| src/models/perform_music.py | End-to-end generation + multi-instrument synthesis. |
| src/models/signal_processing.py | Harmonic & noise DDSP utilities. |
| src/models/train_conductor.py | Training loop for the Transformer arranger. |
| src/models/train_instrument.py | Training loop for instrument decoders. |


graph TD
    subgraph "1. Training Phase (PyTorch Lightning)"
        DATA[SynthDataset: Pitch, Loudness, Audio] --> DL[DataLoader]
        DL --> STEP[Training Step]
        STEP --> FORWARD[Model Forward Pass]
        FORWARD --> SPEC[Mel-Spectrogram Transform]
        SPEC --> LOSS[L1 Spectral Loss]
        LOSS --> OPT[Adam Optimizer]
    end

    subgraph "2. Neural Synthesizer (The Brain)"
        INPUT[Pitch Hz & Loudness] --> MLP[Input MLP & GELU]
        MLP --> GRU[3-Layer Stacked GRU]
        GRU --> NORM[LayerNorm]
        NORM --> H1[Amplitude Head]
        NORM --> H2[Harmonic Head]
        NORM --> H3[Noise Head]
    end

    subgraph "3. DDSP Synthesis (Signal Processing)"
        H1 & H2 & INPUT --> H_SYNTH[Harmonic Synthesis Additive]
        H3 --> N_SYNTH[Noise Synthesis Subtractive]
        H_SYNTH --> MIX[Summation]
        N_SYNTH --> MIX
    end

    MIX --> |Generated Audio| SPEC
    MIX --> |Final Output| WAV[.wav File]
