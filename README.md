# Data Science for Audio Augmentation

**Ever wondered how the New York skyline could sound? 🏙️**

What if patterns we see around us could be transformed into music?

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/input_sample/img1.png" width="750"/>
</p> 


## Contents

- [Motivation](#motivation)
- [Why Data Science (Not Just Deep Learning)?](#why-data-science-(not-just-deep-learning)?)
- [Pipeline Overview](#pipeline-overview)
- [Future Directions](#future-directions)

## Motivation

Data Science has true potential to create sounds never heard before.
To do that, we must first understand how music itself can be represented:
- Music is fundamentally a time-series signal
- It can be decomposed into interpretable components:
  - Pitch (melody)
  - Loudness (dynamics)
  - Timbre (harmonics + noise)

By isolating and controlling these components, we can move beyond imitation → toward true augmentation.

But what if we go one step further?

Instead of learning only from existing music,
we reverse-engineer patterns from the world around us:
- Skylines
- Ocean waves
- Natural curves
- Abstract visual patterns

…and convert them into musical structures.
  
<br/>

## Why Data Science (Not Just Deep Learning)?

Modern AI (Transformers, RNNs, diffusion models) has made:

- Text → Music
- Style transfer
- Long-range coherence possible.

However:

⚠️ These systems are still heavily dependent on historical data

⚠️ True novelty is limited by training distributions

⚠️ Music remains a subjective domain

<br/>

## 💡 This is where Data Science comes in

Instead of treating music as a black box, we:

Decompose signals into interpretable features
Engineer transformations explicitly
Control each component independently

This is analogous to real-world DS systems:

| Domain | Components | 
|--------|--------|
| Marketing	| price, promotion, placement |
| Finance	| risk, return, volatility |
| Music (this project) |	pitch, loudness, timbre |

👉 We model the system, not just the output

<br/>

## 🔁 Core Idea
- Convert visual structure → time-series → controllable audio features → synthesized sound

## Pipeline Overview

### Step 1: Image → Curve Extraction
- Extract a continuous curve from visual inputs (e.g., skyline)
- Convert pixel space → normalized signal
### Step 2: Curve → Time-Series Mapping
- X-axis → time
- Y-axis → signal amplitude / displacement
### Step 3: Feature Engineering
Transform raw curve into musical controls:
- Pitch (Hz) → derived from displacement
- Loudness → derived from height / energy
- Optional:
  - smoothing / denoising
  - resampling
  - scaling
### Step 4: Pitch Postprocessing
- Remove interpolation artifacts
- Enforce discrete musical transitions
- Convert continuous signals → step-like notes
### Step 5: Audio Synthesis
DSP-based synthesis
Neural synthesis
### Step 6: Reconstruction & Evaluation
- Rebuild audio from features
- Compare: waveform, pitch contours, perceptual quality
- final filtering step in music generation workflows

<br/>

### Future Directions

- **Scope for Continuous Improvement** Music is inherently subjective, and this pipeline leverages that flexibility. By learning patterns from high-quality audio and adapting them to pattern-driven generation, the system enables iterative refinement. This allows for multiple cycles of listening, tuning, and enhancement, ensuring there is always room for improvement.

- **Ability to Utilize Diverse Patterns** While skyline structures provide a strong starting point for melodic instruments such as guitar and piano, other patterns can unlock new possibilities. For example, spherical or cyclic patterns could be mapped to rhythmic structures like drums. This opens up exciting Edge AI use cases where composers and users can discover new tunes simply by interpreting patterns they observe in the world around them.

- **Multi-Instrumentation Pipeline** By isolating and visualizing key components such as pitch, loudness, and timbre, the framework can be extended to support multiple instruments simultaneously. This enables the generation of richer, multi-instrument compositions, significantly expanding the scope for innovation and creation of fresh, original music.

Overall, this work reinforces the idea that expressive music generation benefits from a combination of structured data science principles, modular design, and perceptual evaluation. These foundations position the system as a flexible and scalable framework for future research in novel, high-quality audio generation.
