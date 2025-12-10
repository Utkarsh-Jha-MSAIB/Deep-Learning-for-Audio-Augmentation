import numpy as np
import librosa
from pathlib import Path
import os
import sys

# ==============================================================================
# 🕵️ CLEAN MELODY DETECTOR (Output: CHUNK INDICES)
# ==============================================================================

PROJECT_NAME = 'project_111'
INSTRUMENT = 'Guitar'
TOP_K = 20
MIN_DISTANCE_CHUNKS = 50
REQUIRED_DURATION_CHUNKS = 5  # 20 Seconds


def find_clean_melodic_chunks():
    # 1. SETUP
    current_file = Path(__file__).resolve()
    project_root = None
    for parent in current_file.parents:
        if parent.name == PROJECT_NAME:
            project_root = parent
            break
    if project_root is None:
        project_root = Path("/u/erdos/csga/uj3/projects/project_111")

    data_path = project_root / 'data' / 'processed' / 'features'

    audio_file = data_path / f'audio_{INSTRUMENT}.npy'
    pitch_file = data_path / f'pitch_{INSTRUMENT}.npy'

    if not audio_file.exists():
        print(f"❌ Error: Missing audio file {audio_file}")
        return

    print(f"💿 Loading {INSTRUMENT} Audio & Pitch...", flush=True)
    a_data = np.load(audio_file)
    p_data = np.load(pitch_file)

    # --- THE FIX: FORCE 1D SHAPE ---
    print(f"   Original Shapes -> Audio: {a_data.shape}, Pitch: {p_data.shape}")
    a_data = a_data.reshape(-1)
    p_data = p_data.reshape(-1)
    print(f"   Flattened Shapes -> Audio: {a_data.shape}, Pitch: {p_data.shape}")

    # 2. VIRTUAL CHUNKING
    sr = 16000
    samples_per_chunk = 64000  # 4 seconds
    frames_per_chunk = 1000  # 4 seconds

    num_chunks = min(len(a_data) // samples_per_chunk, len(p_data) // frames_per_chunk)

    if num_chunks == 0:
        print("❌ Error: Still found 0 chunks. Dataset might be empty or too short.")
        return

    print(f"   📊 Analyzing {num_chunks} chunks for Tone Quality...", flush=True)

    chunk_scores = []

    for i in range(num_chunks):
        if i % 100 == 0: print(f"   ... Scanning chunk {i}/{num_chunks}", end='\r')

        # Slicing
        a_chunk = a_data[i * samples_per_chunk: (i + 1) * samples_per_chunk]
        p_chunk = p_data[i * frames_per_chunk: (i + 1) * frames_per_chunk]

        # --- METRIC A: SPECTRAL FLATNESS (Tone vs Noise) ---
        try:
            flatness = librosa.feature.spectral_flatness(y=a_chunk)[0]
            avg_flatness = np.mean(flatness)
        except:
            avg_flatness = 1.0

            # --- METRIC B: PITCH CONFIDENCE (Melody vs Silence) ---
        valid_notes = np.sum(p_chunk > 75)
        note_ratio = valid_notes / len(p_chunk)

        # --- METRIC C: ROOT MEAN SQUARE (Volume) ---
        rms = np.sqrt(np.mean(a_chunk ** 2))

        # --- SCORING FORMULA ---
        if rms < 0.02 or note_ratio < 0.4:
            score = 0
        else:
            # High Melody + Low Noise
            score = (note_ratio * 2.0) + (1.0 - avg_flatness)

        chunk_scores.append(score)

    print("\n   ✅ Analysis Complete.")

    # 3. SLIDING WINDOW (Find 20s Blocks)
    chunk_scores = np.array(chunk_scores)
    window_scores = np.convolve(chunk_scores, np.ones(REQUIRED_DURATION_CHUNKS), mode='valid')

    # 4. RANKING & FILTERING
    sorted_indices = np.argsort(window_scores)[::-1]
    selected_indices = []

    for idx in sorted_indices:
        if len(selected_indices) >= TOP_K: break

        is_far = True
        for existing in selected_indices:
            if abs(idx - existing) < MIN_DISTANCE_CHUNKS:
                is_far = False
                break

        if is_far:
            selected_indices.append(idx)

    # 5. OUTPUT
    print("\n" + "=" * 40)
    print(f"🎸 TOP {TOP_K} CLEAN & MELODIC SECTIONS (CHUNK INDICES)")
    print("=" * 40)
    print("Copy this list into your main script:")
    print("-" * 40)

    # OUTPUT: RAW CHUNK INDICES
    print(f"HOT_CHUNKS = {selected_indices}")
    print("-" * 40)

    print("\nDetailed Stats:")
    for i, idx in enumerate(selected_indices):
        print(f"Rank {i + 1:02d}: Chunk {idx:<6} | Score: {window_scores[idx]:.4f}")


if __name__ == "__main__":
    find_clean_melodic_chunks()