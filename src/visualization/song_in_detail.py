import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


# File path of .wav file here
FILE_TO_ANALYZE = "Masked_Path/outputs/gen_Mix_Piano1.0_shift2.wav"

# Config
FRAME_LENGTH = 2048
HOP_LENGTH = 64


def calculate_scores(f0, loudness):
    """Calculates objective 'Vibe Scores' from raw data"""
    avg_energy = np.mean(loudness)
    energy_score = int(avg_energy * 100)

    dynamic_range = np.max(loudness) - np.min(loudness)
    dynamics_score = int(dynamic_range * 100)

    active_pitch = f0[f0 > 0]
    if len(active_pitch) > 0:
        pitch_std = np.std(active_pitch)
        complexity_score = int(np.clip((pitch_std / 100) * 100, 0, 100))
    else:
        complexity_score = 0

    return {
        "Energy": energy_score,
        "Dynamics": dynamics_score,
        "Complexity": complexity_score
    }


def save_report_card(file_path):
    """
    Reusable function to generate a dashboard for ANY file
    Can be imported by other scripts.
    """
    path_obj = Path(file_path)

    if not path_obj.exists():
        print(f"Report Card Error: File not found at {file_path}")
        return

    print(f"Generating Dashboard for: {path_obj.name}...")

    # 1. Load & Extract
    try:
        y, sr = librosa.load(str(path_obj), sr=16000)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    duration = librosa.get_duration(y=y, sr=sr)

    # Extract Features
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_norm = rms / (np.max(rms) + 1e-7)

    # Pitch extraction (fast mode)
    f0, _, _ = librosa.pyin(
        y, fmin=40, fmax=2000, sr=sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    f0 = np.nan_to_num(f0)

    # 2. Calculate Scores
    scores = calculate_scores(f0, rms_norm)

    # 3. Create Plot
    plt.figure(figsize=(10, 12))
    plt.suptitle(f"Audio DNA: {path_obj.name}", fontsize=16, fontweight='bold')

    # A. Waveform
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6, color='#4a90e2')
    plt.title("Waveform (Texture)")
    plt.axis('off')

    # B. Pitch
    plt.subplot(4, 1, 2)
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)
    plt.plot(times, f0, color='#e24a90', linewidth=2)
    plt.title(f"Melody (Complexity: {scores['Complexity']}/100)")
    plt.ylabel("Hz")
    plt.grid(alpha=0.3)

    # C. Loudness
    plt.subplot(4, 1, 3)
    times_rms = librosa.times_like(rms_norm, sr=sr, hop_length=HOP_LENGTH)
    plt.fill_between(times_rms, rms_norm, color='#f5a623', alpha=0.5)
    plt.title(f"Dynamics (Energy: {scores['Energy']}/100)")
    plt.ylabel("Amp")
    plt.grid(alpha=0.3)

    # D. Scorecard Text
    plt.subplot(4, 1, 4)
    plt.axis('off')
    text_str = (
        f"DURATION:   {duration:.2f} sec\n\n"
        f"ENERGY:     {scores['Energy']} / 100\n"
        f"DYNAMICS:   {scores['Dynamics']} / 100\n"
        f"COMPLEXITY: {scores['Complexity']} / 100"
    )
    plt.text(0.5, 0.5, text_str, ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="black", alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 4. Save
    output_img = path_obj.with_suffix('.png')
    plt.savefig(output_img)
    plt.close()

    print(f"Report Card Saved: {output_img}")


if __name__ == "__main__":
    # If run directly, use the hardcoded path
    save_report_card(TEST_FILE)
