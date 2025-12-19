# Expressive Audio Augmentation with Multi-Instrument Neural Architectures

**A deep learning framework for sound-conditioned musical creativity**

## 📑 Table of Contents

- [Overview](#overview)
- [Datasets](#-datasets)
- [Project Pipeline & Modular Structure](#project-pipeline--modular-structure)
- [Generated Song Visualization Metrics](#generated-song-visualization-metrics)
- [RAG Audio Metrics (Top-K Retrieval Evaluation)](#rag-audio-metrics-top-k-retrieval-evaluation)
- [Key Takeaways and Future Outlook](#-key-takeaways-and-future-outlook)
- [Future Directions](#-future-directions)

## Overview

This project implements a multi-instrument audio-conditioned framework for expressive music augmentation. The system utilizes a specialist-model strategy to transform raw audio seeds into structured ensemble performances. By pairing a Transformer-based arrangement module for global structural planning with **Neural-DSP (DDSP) decoders** for high-clarity sound synthesis, the pipeline captures instrument-specific nuances and hierarchical musical patterns. 

💡 A core innovation is the integration of a **Music-Specific Retrieval-Augmented Generation (RAG)** architecture. This allows the system to analyze the structure of the input audio and bridge the gap between consecutive sounds by identifying the most compatible segments from a musical database. By evaluating mathematical similarity and coherence across pitch and rhythm, the RAG logic ensures that any generated extension or accompaniment feels like a natural continuation of the original performance.

The system integrates three key technical directions:
- **Multi-Instrument Neural-DSP Pipeline** (audio-conditioned RNN + Transformer)
  - Each instrument has its own Instrument RNN (NeuralSynthesizer)
  - DDSP modules provide realistic harmonic & noise modeling
  - Transformer Arranger predicts follower-instrument dynamics based on a leader signal
  - Ensemble arrangement (bass + drums + guitar + etc.)
 

<p align="center">
  <img src="https://github.com/user-attachments/assets/e91e9aa9-8a7d-4168-b35e-d09d205021ed" width="750"/>
</p>
    

- **Audio Retrieval-Augmented Generation (RAG)** (for extension, coherence, and style grounding)
  - Large feature database of audio chunks
  - Retrieval based on similarity (chroma, energy, coherence)
  - Weighted scoring: relevance + seam smoothness
  - Entire multi-instrument pipeline can generate extended Top-K continuations
  - Adaptive Multi-Scale Lookback: Matches style by analyzing GRU embeddings across $7.0\text{s}$ to $8.5\text{s}$ windows.


  <p align="center">
  <img src="https://github.com/user-attachments/assets/e9cbfe7c-8dc5-4ca3-8e43-559121c288e3" width="400"/>
</p> 
  
<br/>

📂 **Datasets**

1. **Essen Folksong (KRN)**: Symbolic-only dataset for early pipeline validation.

2. **BabySlakh**: Small multitrack dataset for debugging and small-scale decoder training.

3. **LSX**: Large-scale (~28 GB) multitrack dataset enabling
  - Multi-instrument arrangement
  - Temporal synchronization
  - Realistic ensemble modeling

4. **URMP**: Classical multi-instrument stems; used for acoustic timbre modeling.

<br/>

## Project Pipeline & Modular Structure

| Category | Script | Description |
|--------|--------|-------------|
| Inference | `src/models/perform_music.py` | Main entry point. Handles end-to-end generation and multi-instrument ensemble synthesis |
| Inference | `src/models/audio_RAG.py` | Implements Retrieval-Augmented Generation to extend user audio using style-matched database segments |
| Orchestration | `src/models/decoder_conductor.py` | Transformer arranger logic that predicts dynamic loudness curves for follower instruments |
| Orchestration | `src/models/train_conductor.py` | Training loop for learning musical “conduction” patterns |
| Synthesis | `src/models/decoder_instrument.py` | Instrument decoder combining a 3-layer GRU with DDSP synthesis heads |
| Synthesis | `src/models/train_instrument.py` | Training environment for learning instrument-specific timbre |
| DSP Core | `src/models/signal_processing.py` | Core DSP engine for harmonic additive synthesis and filtered noise generation |

<br/>

## Generated Song Visualization Metrics

The following metrics are computed for each generated song to support qualitative analysis, comparative evaluation against prior arrangements, and improved interpretability while listening to the output audio. Such analysis can form as foundation for further applying transfer learning, advanced ML techniques for truly augmented audio.

| Metric | Musical Aspect | Description | Formula / Definition |
|------|---------------|-------------|----------------------|
| **Waveform (Texture)** | Timbral structure | Time-domain visualization of the raw audio signal, illustrating transient density, amplitude variation, and overall waveform texture. Useful for identifying articulation patterns and temporal density. | y(t): audio signal amplitude at time t |
| **Melody (Pitch / Complexity)** | Pitch content | Fundamental frequency trajectory extracted using pYIN. Melodic complexity is measured as normalized pitch variance across voiced frames. | f0(t): fundamental frequency estimated by pYIN |
| **Dynamics (Loudness / Energy)** | Expressive dynamics | Temporal loudness envelope computed using RMS over short time windows. Average energy reflects perceived loudness, while dynamics capture amplitude variation over time. | RMS(t) = sqrt( (1/N) * sum(y_i²) ) |

<br/>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a9d9eaa-b1b1-4d78-8031-d66a088ce7e4" width="550"/>
</p>

<br/>

## RAG Audio Metrics (Top-K Retrieval Evaluation)

The table below reports the **Top-10 Audio RAG retrieval results** for two input songs:

Each output is generated using **one retrieved database segment** selected by a composite score that balances *style relevance* and *temporal coherence*.

### Evaluation Criteria
- **Rank**: Ordering of candidates sorted by composite score (best first)
- **IDX**: Start index of the selected database audio chunk
- **Composite Score**: Weighted combination of relevance (style similarity) and coherence (smooth pitch and loudness transition)

### Top-10 Retrieval Results

| Rank | Input Song 1: DB Chunk IDX | Input Song 1: Composite Score | Input Song 2: DB Chunk IDX | Input Song 2: Composite Score |
|------|------------------------|---------------------------|---------------------|------------------------|
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

**Perceptual Evaluation (Human Listening):**
- For *Input Song 1*, Rank-1 was consistently rated as the best perceptual continuation.
- For *Input Song 2*, Rank-9 was preferred despite a lower composite score.

These findings indicate strong alignment between the automated RAG ranking and subjective musical continuity, while also highlighting the importance of human listening as a final filtering step in music generation workflows.

<br/>

## 🌟 Key Takeaways and Future Outlook

This project highlights several important observations about expressive audio generation and large-scale music modeling:

- **Scaling matters for audio generation.** Similar to trends observed in large language models, audio-to-audio deep learning performance improves significantly with both data scale and model capacity. Moving from symbolic KRN data to large-scale multitrack datasets such as LSX, and scaling models from approximately 1.6M to 4.7M parameters, resulted in noticeably higher audio fidelity. These results strongly suggest that continued scaling along both data and model dimensions will yield further improvements.

- **Musical quality cannot be fully captured by numbers.** Quantitative metrics such as Energy, Dynamics, or composite Audio RAG scores provide useful signals, but they do not reliably correlate with perceived musical quality. In practice, human listening remained indispensable, with outputs often accepted or rejected based on auditory judgment rather than numerical thresholds. This reinforces the role of perceptual evaluation in music generation systems.

- **There is no single “optimal” music output.** Unlike many optimization problems, music generation does not converge to a strict optimal solution. There is always room for perceptual refinement. This insight motivated architectural experimentation, including specialized handling of drums using loudness-only representations processed through 1D convolutions. These results suggest that different instruments may benefit from tailored modeling strategies rather than uniform architectures.

<br/>

### 🔮 Future Directions

- **Expanded specialist-instrument modeling.** Treating each instrument as a specialist model has proven effective in capturing expressive timbre and articulation. Future work will extend this approach to a broader range of instruments, with a focus on maintaining large, balanced datasets per instrument to avoid degradation due to data scarcity.

- **Multi-stage Audio RAG composition.** While the current Audio RAG system focuses on high-quality single-segment extensions, the framework naturally extends to multi-hop retrieval and stitching. By chaining multiple high-scoring segments using a bankable composite score (relevance + coherence), the system could generate longer, more structured, and musically coherent multi-instrument compositions.

Overall, this work reinforces the idea that expressive music generation benefits from a combination of scale, modular design, perceptual evaluation, and retrieval-based grounding. These principles position the system as a flexible foundation for future research in structured, high-quality audio generation.

