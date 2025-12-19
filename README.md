# Expressive Audio Augmentation with Multi-Instrument Neural Architectures

**A deep learning framework for sound-conditioned musical creativity**

## Overview

This project implements a multi-instrument audio-conditioned framework for expressive music augmentation. The system utilizes a specialist-model strategy to transform raw audio seeds into structured ensemble performances. By pairing a Transformer-based arrangement module for global structural planning with Neural-DSP (DDSP) decoders for high-clarity sound synthesis, the pipeline captures instrument-specific nuances and hierarchical musical patterns. 

💡 A core innovation is the integration of a Music-Specific Retrieval-Augmented Generation (RAG) mechanism. This allows the system to analyze the structure of the input audio and bridge the gap between consecutive sounds by identifying the most compatible segments from a musical database. By evaluating mathematical similarity and coherence across pitch and rhythm, the RAG logic ensures that any generated extension or accompaniment feels like a natural continuation of the original performance.

The system integrates three key technical directions:
- **Multi-Instrument Neural-DSP Pipeline** (audio-conditioned RNN + Transformer)
  - Each instrument has its own Instrument RNN (NeuralSynthesizer)
  - DDSP modules provide realistic harmonic & noise modeling
  - Transformer Arranger predicts follower-instrument dynamics based on a leader signal
  - Ensemble arrangement (bass + drums + guitar + etc.)
 

<p align="center">
  <img src="https://github.com/user-attachments/assets/e91e9aa9-8a7d-4168-b35e-d09d205021ed" width="500"/>
</p>
     


- **Audio Retrieval-Augmented Generation (RAG)** (for extension, coherence, and style grounding)
  - Large feature database of audio chunks
  - Retrieval based on similarity (chroma, energy, coherence)
  - Weighted scoring: relevance + seam smoothness
  - Entire multi-instrument pipeline can generate extended Top-K continuations
 
  <p align="center">
  <img src="https://github.com/user-attachments/assets/e9cbfe7c-8dc5-4ca3-8e43-559121c288e3" width="300"/>
</p> 
  

📂 **Datasets**

1. **Essen Folksong (KRN)**: Symbolic-only dataset for early pipeline validation.

2. **BabySlakh**: Small multitrack dataset for debugging and small-scale decoder training.

3. **LSX**: Large-scale (~28 GB) multitrack dataset enabling
  - Multi-instrument arrangement
  - Temporal synchronization
  - Realistic ensemble modeling

4. **URMP**: Classical multi-instrument stems; used for acoustic timbre modeling.

# Project Pipeline & Modular Structure

| Category | Script | Description |
|--------|--------|-------------|
| Inference | `src/models/perform_music.py` | Main entry point. Handles end-to-end generation and multi-instrument ensemble synthesis |
| Inference | `src/models/audio_RAG.py` | Implements Retrieval-Augmented Generation to extend user audio using style-matched database segments |
| Orchestration | `src/models/decoder_conductor.py` | Transformer arranger logic that predicts dynamic loudness curves for follower instruments |
| Orchestration | `src/models/train_conductor.py` | Training loop for learning musical “conduction” patterns |
| Synthesis | `src/models/decoder_instrument.py` | Instrument decoder combining a 3-layer GRU with DDSP synthesis heads |
| Synthesis | `src/models/train_instrument.py` | Training environment for learning instrument-specific timbre |
| DSP Core | `src/models/signal_processing.py` | Core DSP engine for harmonic additive synthesis and filtered noise generation |


