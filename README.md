# Expressive Audio Augmentation with Multi-Instrument Neural Architectures

**A deep learning framework for sound-conditioned musical creativity**

---

## Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Pipeline & Modular Structure](#project-pipeline--modular-structure)
- [Generated Song Visualization Metrics](#generated-song-visualization-metrics)
- [RAG Audio Metrics (Top-K Retrieval Evaluation)](#rag-audio-metrics-top-k-retrieval-evaluation)
- [Key Takeaways and Future Outlook](#-key-takeaways-and-future-outlook)

---

## Overview

This project implements a multi-instrument audio-conditioned framework for expressive music augmentation. The system utilizes a specialist-model strategy to transform raw audio seeds into structured ensemble performances.

By pairing a Transformer-based arrangement module for global structural planning with **Neural-DSP (DDSP) decoders** for high-clarity sound synthesis, the pipeline captures instrument-specific nuances and hierarchical musical patterns.

💡 A core innovation is the integration of a **Music-Specific Retrieval-Augmented Generation (RAG)** architecture. This allows the system to analyze the structure of the input audio and bridge the gap between consecutive sounds by identifying the most compatible segments from a musical database.

By evaluating mathematical similarity and coherence across pitch and rhythm, the RAG logic ensures that any generated extension or accompaniment feels like a natural continuation of the original performance.

---

### System Components

The system integrates three key technical directions:

#### 🎧 Multi-Instrument Neural-DSP Pipeline (RNN + Transformer)

- Each instrument has its own Instrument RNN (NeuralSynthesizer)
- DDSP modules provide realistic harmonic & noise modeling
- Transformer Arranger predicts follower-instrument dynamics based on a leader signal
- Ensemble arrangement (bass + drums + guitar + etc.)

<p align="center">
  <img src="https://github.com/user-attachments/assets/e91e9aa9-8a7d-4168-b35e-d09d205021ed" width="750"/>
</p>

---

#### 🔁 Audio Retrieval-Augmented Generation (RAG)

- Large feature database of audio chunks
- Retrieval based on similarity (chroma, energy, coherence)
- Weighted scoring: relevance + seam smoothness
- Multi-instrument pipeline supports Top-K continuations
- Adaptive Multi-Scale Lookback using GRU embeddings across $7.0\text{s}$ to $8.5\text{s}$ windows

<p align="center">
  <img src="https://github.com/user-attachments/assets/e9cbfe7c-8dc5-4ca3-8e43-559121c288e3" width="400"/>
</p>

---

## Datasets

1. **Essen Folksong (KRN)**  
   Symbolic-only dataset for early pipeline validation.

2. **BabySlakh**  
   Small multitrack dataset for debugging and small-scale decoder training.

3. **LSX (~28GB)**  
   Large-scale multitrack dataset enabling:
   - Multi-instrument arrangement  
   - Temporal synchronization  
   - Realistic ensemble modeling  

4. **URMP**  
   Classical multi-instrument stems used for acoustic timbre modeling.

---

## Project Pipeline & Modular Structure

| Category | Script | Description |
|--------|--------|-------------|
| Inference | `src/models/perform_music.py` | Main entry point for end-to-end multi-instrument synthesis |
| Inference | `src/models/audio_RAG.py` | Retrieval-Augmented Generation for style-matched audio continuation |
| Orchestration | `src/models/decoder_conductor.py` | Transformer arranger for dynamic control of follower instruments |
| Orchestration | `src/models/train_conductor.py` | Training loop for learning musical conduction patterns |
| Synthesis | `src/models/decoder_instrument.py` | GRU-based instrument decoder with DDSP heads |
| Synthesis | `src/models/train_instrument.py` | Training environment for instrument-specific timbre |
| DSP Core | `src/models/signal_processing.py` | Harmonic additive synthesis + filtered noise engine |

---

## Generated Song Visualization Metrics

These metrics support interpretability, qualitative analysis, and comparison of generated audio.

| Metric | Musical Aspect | Description | Formula / Definition |
|------|---------------|-------------|----------------------|
| **Waveform (Texture)** | Timbre | Time-domain representation showing amplitude variation and transient structure | y(t): audio amplitude |
| **Melody (Pitch / Complexity)** | Pitch | Fundamental frequency extracted using pYIN; complexity via pitch variance | f₀(t): fundamental frequency |
| **Dynamics (Loudness / Energy)** | Energy | RMS-based loudness envelope capturing temporal variation | RMS(t) = √(1/N ∑ y²) |

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a9d9eaa-b1b1-4d78-8031-d66a088ce7e4" width="550"/>
</p>

---

## RAG Audio Metrics (Top-K Retrieval Evaluation)

### Evaluation Criteria

- **Rank** → Ordered by composite score  
- **IDX** → Start index of retrieved database chunk  
- **Composite Score** → Weighted combination of relevance and temporal coherence  

---

### Top-10 Retrieval Results

| Rank | Input Song 1: DB Chunk IDX | Input Song 1: Score | Input Song 2: DB Chunk IDX | Input Song 2: Score |
|------|------------------------|--------------------|---------------------|--------------------|
| 1 | **38773** | **1.3866** | 32772 | 1.3419 |
| 2 | 38772 | 1.3856 | 31330 | 1.3358 |
| 3 | 35602 | 1.3854 | 60316 | 1.3308 |
| 4 | 35601 | 1.3832 | 32531 | 1.3013 |
| 5 | 50781 | 1.3814 | 48190 | 1.2995 |
| 6 | 58783 | 1.3811 | 6401 | 1.2967 |
| 7 | 43069 | 1.3799 | 60221 | 1.2943 |
| 8 | 20349 | 1.3776 | 23115 | 1.2941 |
| 9 | 32140 | 1.3773 | **41021** | **1.2842** |
| 10 | 20713 | 1.3768 | 21927 | 1.2781 |

---

### Perceptual Evaluation (Human Listening)

- For *Input Song 1*, Rank-1 was consistently preferred  
- For *Input Song 2*, Rank-9 was preferred despite lower score  

👉 Highlights:
- Strong alignment between RAG ranking and perception  
- Human evaluation remains critical in music systems  

---

## 🌟 Key Takeaways and Future Outlook

### Key Takeaways

- **Scaling matters**  
  Increasing dataset size (KRN → LSX) and model capacity (~1.6M → 4.7M params) significantly improved audio fidelity.

- **Metrics ≠ musical quality**  
  Numerical scores (energy, dynamics, RAG score) are useful but insufficient.  
  Human listening remains the final benchmark.

- **No single optimal solution**  
  Music is subjective — multiple valid outputs exist.  
  This enables iterative refinement and creative exploration.

- **Specialized modeling improves performance**  
  Different instruments benefit from tailored architectures (e.g., drum modeling via loudness-based CNN).

---

### Future Directions

- **Expanded specialist-instrument modeling**  
  Extend architecture to more instruments with balanced datasets.

- **Multi-stage Audio RAG composition**  
  Chain multiple retrieved segments to generate longer, structured compositions.

---

## 🎯 Final Insight

This work reinforces that expressive music generation benefits from:

- scale  
- modular design  
- perceptual evaluation  
- retrieval-based grounding  

Together, these principles establish a strong foundation for **structured, high-quality audio generation systems**.
