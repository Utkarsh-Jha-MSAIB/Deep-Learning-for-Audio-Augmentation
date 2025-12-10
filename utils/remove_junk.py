import numpy as np
import librosa
import soundfile as sf
import sys
from pathlib import Path
import os
import math

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================

# --- USER INPUT ---
INPUT_FILE_TO_TRIM = "Smart_Arrangement_RAG_DEBUG_FIXED_Unison_0.wav"
# ---

# Standard Feature Config
SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 64

# DYNAMIC THRESHOLD FACTOR (Only used for printing dynamic threshold, not for actual trim)
THRESHOLD_FACTOR = 0.20

# CRITICAL FIX: THRESHOLD for removing INTERNAL silent gaps (5% of normalized max amp)
INTERNAL_GAP_THRESHOLD = 0.02


# ==============================================================================
# Helper Function for Debugging
# ==============================================================================

def print_avg_loudness_per_second(y_audio, sr):
    # ... (same as before) ...
    segment_samples = sr
    total_samples = len(y_audio)
    num_seconds = total_samples // segment_samples

    loudness_data = []

    print("\n--- AVERAGE LOUDNESS (RMS) PER SECOND ---")
    for i in range(num_seconds):
        start_sample = i * segment_samples
        end_sample = (i + 1) * segment_samples

        segment = y_audio[start_sample:end_sample]

        if segment.size > 0:
            rms_value = np.sqrt(np.mean(segment ** 2))
            loudness_data.append(f"Sec {i:02d}: {rms_value:.4f}")

    # Print data in a structured column format
    columns = 4
    for i in range(0, len(loudness_data), columns):
        print("  " + " | ".join(loudness_data[i:i + columns]))
    print("-----------------------------------------")


# ==============================================================================
# Execution Logic
# ==============================================================================

def loudness_trim(project_root):
    # ... (path setup remains the same) ...
    INPUT_DIR = project_root / 'data' / 'raw' / 'user_uploads'
    OUTPUT_DIR = project_root / 'data' / 'raw' / 'user_uploads'

    input_path_obj = INPUT_DIR / INPUT_FILE_TO_TRIM
    output_filename = f"{input_path_obj.stem}_gap_fixed.wav"
    output_path_obj = OUTPUT_DIR / output_filename

    if not input_path_obj.exists():
        sys.exit(f"❌ FATAL ERROR: Input file not found at: {input_path_obj}")

    print(f"🚀 Starting Loudness-Based Trim for: {input_path_obj.name}")

    try:
        # 1. Load Audio
        y, sr = librosa.load(str(input_path_obj), sr=SAMPLE_RATE)
        original_len = len(y)

        # 2. DEBUG: Print Loudness per second
        print_avg_loudness_per_second(y, sr)

        # 3. Extract and Normalize RMS (Matching Plot Logic)
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        rms_norm = rms / (np.max(rms) + 1e-7)

        # 4. Calculate Dynamic Loudness Threshold (for printing only)
        avg_normalized_loudness = np.mean(rms_norm)
        trim_threshold = avg_normalized_loudness * THRESHOLD_FACTOR

        print(f"   Calculated Average Loudness (Normalized RMS): {avg_normalized_loudness:.4f}")
        print(f"   Dynamic Trim Threshold (20% of Avg): {trim_threshold:.4f}")

        # --- CRITICAL FIX: INTERNAL GAP REMOVAL ---

        # 5. Calculate the absolute amplitude threshold for the internal gap
        # We use a strict, fixed threshold relative to the max amplitude.
        raw_amplitude_threshold = INTERNAL_GAP_THRESHOLD * np.max(np.abs(y))

        # 6. Find all samples that are above the internal silence threshold
        abs_y = np.abs(y)
        active_samples_indices = np.where(abs_y >= raw_amplitude_threshold)[0]

        if active_samples_indices.size == 0:
            print("⚠️ Warning: Audio too quiet or completely silent.")
            y_fixed = y
        else:
            # 7. Identify the contiguous segments of active audio

            # Create a boolean mask where samples are active
            is_active = (abs_y >= raw_amplitude_threshold)

            # Find the starts and ends of contiguous active blocks
            active_starts = np.where(np.diff(is_active.astype(int)) == 1)[0] + 1
            active_ends = np.where(np.diff(is_active.astype(int)) == -1)[0] + 1

            # Handle edges (start/end of the file)
            if is_active[0]:
                active_starts = np.insert(active_starts, 0, 0)
            if is_active[-1]:
                active_ends = np.append(active_ends, len(y))

            # Match up starts and ends (handle cases where array size might differ)
            if active_starts.size > active_ends.size:
                active_starts = active_starts[:active_ends.size]
            elif active_ends.size > active_starts.size:
                active_ends = active_ends[:active_starts.size]

            # 8. Concatenate only the active segments, skipping the silent gaps
            active_segments = [y[start:end] for start, end in zip(active_starts, active_ends) if end > start]
            y_fixed = np.concatenate(active_segments) if active_segments else np.array([], dtype=y.dtype)

        # --- End Internal Gap Removal ---

        trimmed_len = len(y_fixed)

        # 9. Save Output
        if trimmed_len > 0:
            sf.write(output_path_obj, y_fixed, sr)
            removed_s = (original_len - trimmed_len) / sr
            print(f"   ✅ Total time removed (including internal gaps): {removed_s:.3f}s")
            print(f"   Output saved to: {output_path_obj}")
        else:
            print("   ❌ Trim resulted in no active audio.")

    except Exception as e:
        print(f"❌ Critical Error during trimming: {e}")


if __name__ == "__main__":
    # --- Dynamic Project Path Setup ---
    PROJECT_ROOT = Path(__file__).resolve()
    while PROJECT_ROOT.name != 'project_111' and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent
    if PROJECT_ROOT.name != 'project_111':
        sys.exit("❌ FATAL ERROR: Could not locate 'project_111' root directory.")
    # ----------------------------------

    loudness_trim(PROJECT_ROOT)