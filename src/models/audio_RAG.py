import numpy as np
import librosa
import torch
import sys
import math
from pathlib import Path
from scipy.stats import entropy
import random
import os
import shutil
import warnings
from tqdm import tqdm
from collections import defaultdict 

#warnings.filterwarnings("ignore", module="librosa")

# Import engine
try:
    import src.models.perform_music as neural_band
except ImportError as e:
    sys.exit(f"Cannot import neural_band. Check dependencies. Error: {e}")

try:
    import torchcrepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

# Input file location & other parameters to always update

INPUT_FILENAME = "data/raw/user_uploads/PAR_Instrumental.wav"
TARGET_DURATION = 30.0
BASE_CHUNK_DURATION = 2.0
TEMP_LEADER_NAME = 'RAG_SINGLE_STEP'

# For GRU similarity with Input & Trained results, ensure to add leader, can check other instruments also
GRU_MODEL_CHECKPOINT = "checkpoints/Guitar.ckpt"

SAMPLE_RATE = 16000
FRAME_RATE = 250
CROSSFADE_TIME_SAMPLES = int(0.01 * SAMPLE_RATE)

# We need to specify how much last n seconds of input song we want to include for RAG logic
LOOKBACK_OPTIONS = [8.5, 8, 7, 7.5]
SEAM_FRAME_COUNT = 100

# Coherence constants we want to start with
ALPHA = 0.8
BETA = 0.6

# Coherence & RMS threshold 
COHERENCE_MIN_THRESHOLD = 0.70
RMS_TOLERANCE_PCT = 0.20

# How many top final songs we want from RAG to listen
TOP_K_RESULTS = 10

# Path finder
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != 'project_111' and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.name != 'project_111':
    sys.exit("ERROR: Could not locate 'project_111' root directory")
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE_PATH = PROJECT_ROOT / INPUT_FILENAME
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'features'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'


# Start for the function

def audio_to_chroma(audio_waveform, sr=SAMPLE_RATE):
    chromagram = librosa.feature.chroma_cqt(y=audio_waveform, sr=sr, hop_length=512)
    chroma_vector = np.mean(chromagram, axis=1)
    norm = np.linalg.norm(chroma_vector)
    return (chroma_vector / norm + 1e-9) if norm > 0 else (chroma_vector + 1e-9)


def pitch_to_chroma(f0_curve):
    active_f0 = f0_curve[f0_curve > 40]
    if len(active_f0) == 0: return np.zeros(12) + 1e-9
    midi_notes = 69 + 12 * np.log2(active_f0 / 440.0)
    chroma_vals = np.mod(np.round(midi_notes), 12).astype(int)
    chroma_vector = np.bincount(chroma_vals, minlength=12)
    norm = np.linalg.norm(chroma_vector)
    return (chroma_vector / norm + 1e-9) if norm > 0 else (chroma_vector + 1e-9)


def gru_embedding_relevance(p_embed, q_embed):
    """Calculates cosine similarity between GRU embedding vectors"""
    p = p_embed.flatten()
    q = q_embed.flatten()

    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)

    if norm_p == 0 or norm_q == 0:
        return 0.0

    return np.dot(p, q) / (norm_p * norm_q)


def crossfade_audio(audio_out, audio_in, length_samples):
    if len(audio_out) < length_samples: return audio_out
    fade_out = np.linspace(1.0, 0.0, length_samples)
    fade_in = np.linspace(0.0, 1.0, length_samples)

    audio_out[-length_samples:] *= fade_out
    audio_in[:length_samples] *= fade_in
    return audio_out


def cleanup_temp_files(temp_name):
    temp_files = [
        DATA_DIR / f'pitch_{temp_name}.npy',
        DATA_DIR / f'loudness_{temp_name}.npy',
        DATA_DIR / f'audio_{temp_name}.npy'
    ]
    cleaned = 0
    for f in temp_files:
        if f.exists():
            os.remove(f)
            cleaned += 1
    if cleaned > 0:
        pass


def get_gru_embedding(audio_segment):
    checkpoint_path = Path(GRU_MODEL_CHECKPOINT)

    if not checkpoint_path.exists():
        chroma = audio_to_chroma(audio_segment)
        return chroma[:12]

    chroma = audio_to_chroma(audio_segment)
    return chroma[:12]


def run_audio_rag():
    print(f"Starting Audio RAG Pipeline (Target: {TARGET_DURATION}s)...")

    if not INPUT_FILE_PATH.exists():
        sys.exit(f"ERROR: Input audio file not found at: {INPUT_FILE_PATH}")

    try:
        # 1. Load user input
        y_user_input, sr = librosa.load(str(INPUT_FILE_PATH), sr=SAMPLE_RATE)
        input_duration = len(y_user_input) / SAMPLE_RATE

        if TARGET_DURATION <= input_duration:
            sys.exit(
                f"Warning: Target duration ({TARGET_DURATION}s) must be greater than input duration ({input_duration:.1f}s) for extension")

        SEARCH_DURATION = TARGET_DURATION - input_duration
        print(
            f"   Input duration: {input_duration:.1f}s. Searching for single {SEARCH_DURATION:.1f}s extension segment")

        # Calculate Reference RMS for Dynamic Energy Filter
        input_rms_frames = librosa.feature.rms(y=y_user_input, frame_length=2048, hop_length=64, center=True)[0]

        # Calculate Average and Standard Deviation of RMS
        avg_input_rms = np.mean(input_rms_frames)
        std_input_rms = np.std(input_rms_frames)

        # Define Tolerance Bands for both metrics
        rms_avg_min = avg_input_rms * (1 - RMS_TOLERANCE_PCT)
        rms_avg_max = avg_input_rms * (1 + RMS_TOLERANCE_PCT)

        rms_std_min = std_input_rms * (1 - RMS_TOLERANCE_PCT)
        rms_std_max = std_input_rms * (1 + RMS_TOLERANCE_PCT)

        print(f"   Avg RMS Range: [{rms_avg_min:.4f}, {rms_avg_max:.4f}]")
        print(f"   STD RMS Range: [{rms_std_min:.4f}, {rms_std_max:.4f}]")

        # Database Loading
        db_pitches = np.load(DATA_DIR / 'pitch_Guitar.npy').reshape(-1)
        db_loudness = np.load(DATA_DIR / 'loudness_Guitar.npy').reshape(-1)
        db_audio = np.load(DATA_DIR / 'audio_Guitar.npy').reshape(-1)

        # 2. Analyze user input
        input_len_frames = int(input_duration * FRAME_RATE)
        seam_start_frame = input_len_frames - SEAM_FRAME_COUNT

        # Seam Tracker (raw frames)
        last_pitch_end = db_pitches[seam_start_frame: input_len_frames]
        last_loudness_end = db_loudness[seam_start_frame: input_len_frames]

        # Multiple Relevance Queries (Generate GRU embeddings)
        user_gru_queries = {}
        for dur in LOOKBACK_OPTIONS:
            if dur <= input_duration:
                start_sample = len(y_user_input) - int(dur * SAMPLE_RATE)
                query_audio = y_user_input[start_sample:]
                user_gru_queries[dur] = get_gru_embedding(query_audio)

        if not user_gru_queries:
            sys.exit("ERROR: Input audio is too short for any lookback option")

        # 3. Single-Step Retrieval Search
        search_len_frames = int(SEARCH_DURATION * FRAME_RATE)
        search_len_samples = int(SEARCH_DURATION * SAMPLE_RATE)
        base_chunk_frames = int(BASE_CHUNK_DURATION * FRAME_RATE)

        num_base_chunks = int(SEARCH_DURATION / BASE_CHUNK_DURATION)
        num_db_base_chunks = len(db_pitches) // base_chunk_frames

        # Try Exhaustive Search
        sampling_range = num_db_base_chunks - num_base_chunks
        if sampling_range <= 0: sys.exit(f"ERROR: Database is too short to find a {SEARCH_DURATION:.1f}s segment")
        sample_indices = range(sampling_range) 

        # Initialize Top K Match Tracker
        top_k_matches = []
        best_composite_score = -1.0
        
        for attempt in range(2):

            if attempt == 0:
                current_alpha = ALPHA
                current_beta = BETA
                coherence_floor = COHERENCE_MIN_THRESHOLD
                desc_label = "Strict Search (GRU/Coherence)"
            else:
                current_alpha = 0.95
                current_beta = 0.05   #Since finder higher coherence is bigger issue
                coherence_floor = 0.0
                desc_label = "Fallback Search"
                print(f"      Strict search failed. Retrying with α={current_alpha}, β={current_beta}")

            search_iterator = tqdm(sample_indices, desc=desc_label)

            for base_idx in search_iterator:
                start_f_db = base_idx * base_chunk_frames
                start_s_db = base_idx * int(BASE_CHUNK_DURATION * SAMPLE_RATE)

                # Retrieve features and audio for the candidate segment
                db_segment_audio = db_audio[start_s_db: start_s_db + search_len_samples]
                db_segment_pitch = db_pitches[start_f_db: start_f_db + search_len_frames]
                db_segment_loudness = db_loudness[start_f_db: start_f_db + search_len_frames]

                # Dynamic Pattern Filter
                candidate_rms_frames = \
                    librosa.feature.rms(y=db_segment_audio, frame_length=2048, hop_length=64, center=True)[0]

                avg_candidate_rms = np.mean(candidate_rms_frames)
                std_candidate_rms = np.std(candidate_rms_frames)

                # Check 1: Average Loudness Consistency
                if not (rms_avg_min <= avg_candidate_rms <= rms_avg_max):
                    continue

                # Check 2: Dynamic Range Consistency (Pattern Variance)
                if not (rms_std_min <= std_candidate_rms <= rms_std_max):
                    continue

                # Calculating coherence score, dependent on both pitch & loudness
                candidate_pitch_start = db_segment_pitch[:SEAM_FRAME_COUNT]
                candidate_loudness_start = db_segment_loudness[:SEAM_FRAME_COUNT]

                pitch_penalty = np.mean((candidate_pitch_start - last_pitch_end) ** 2)
                loudness_penalty = np.mean((candidate_loudness_start - last_loudness_end) ** 2)

                pitch_coherence = np.exp(-pitch_penalty * 50)
                loudness_coherence = np.exp(-loudness_penalty * 30)

                coherence_score = (pitch_coherence + loudness_coherence) / 2

                # Coherence Floor Guard
                if coherence_score < coherence_floor:
                    continue

                # 2. Relevance Score - GRU Embedding Match (Also tried cosine similarity logic here)
                candidate_gru_embed = get_gru_embedding(db_segment_audio)

                max_relevance = -1.0
                best_lookback_dur = 0.0

                for dur, user_gru_embed in user_gru_queries.items():
                    relevance = gru_embedding_relevance(candidate_gru_embed, user_gru_embed)
                    if relevance > max_relevance:
                        max_relevance = relevance
                        best_lookback_dur = dur

                # 3. Calculating a composite score
                composite_score = (current_alpha * max_relevance) + (current_beta * coherence_score)

                # Track Top K Matches
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    search_iterator.set_postfix(score=f"{best_composite_score:.4f}")

                # Add the result to the list of top matches
                top_k_matches.append({
                    'score': composite_score,
                    'db_idx': base_idx,
                    'coherence': coherence_score,
                    'relevance': max_relevance,
                    'lookback': best_lookback_dur
                })

            # After each attempt, sort all matches found so far and prune to TOP_K
            top_k_matches.sort(key=lambda x: x['score'], reverse=True)
            top_k_matches = top_k_matches[:TOP_K_RESULTS]

            if top_k_matches and top_k_matches[0]['db_idx'] != -1:
                if attempt == 0: break

        # 4. Final concatenation & Loop through Top K results
        if not top_k_matches:
            sys.exit("FAILURE: Could not find any suitable extension segment after filtering")

        print(f"\nRetrieval Complete. Found Top {len(top_k_matches)} extension segments")

        # Loop through Top K matches for Arrangement (adding drums, bass, other trained instruments)
        for rank, match in enumerate(top_k_matches):
            best_idx = match['db_idx']
            start_s_db = best_idx * int(BASE_CHUNK_DURATION * SAMPLE_RATE)

            # Retrieve the specific audio segment for this index
            retrieved_audio = db_audio[start_s_db: start_s_db + search_len_samples]

            # Re-load clean input for stitching, ensures fresh copy for every run
            y_user_input_clean, _ = librosa.load(str(INPUT_FILE_PATH), sr=SAMPLE_RATE)

            # Stitch the Input and Retrieval
            input_audio_copy = y_user_input_clean.copy()
            retrieved_audio_copy = retrieved_audio.copy()
            crossfade_len = CROSSFADE_TIME_SAMPLES

            crossfade_audio(input_audio_copy, retrieved_audio_copy, crossfade_len)

            y_final_audio = np.concatenate([
                input_audio_copy[:-crossfade_len],
                input_audio_copy[-crossfade_len:] + retrieved_audio_copy[:crossfade_len],
                retrieved_audio_copy[crossfade_len:]
            ])

            final_audio_len_s = len(y_final_audio) // SAMPLE_RATE

            # Feature extraction
            final_pitch_arr, _, _ = librosa.pyin(y_final_audio, fmin=40, fmax=2000, sr=SAMPLE_RATE, hop_length=64)
            final_pitch_arr = np.nan_to_num(final_pitch_arr)

            final_loudness_arr = librosa.feature.rms(y=y_final_audio, frame_length=2048, hop_length=64, center=True)[0]
            final_loudness_arr = np.nan_to_num(final_loudness_arr)

            N_frames = min(len(final_pitch_arr), len(final_loudness_arr))
            N_chunks = 1
            final_len_samples = len(y_final_audio)

            # Save the unique features for this specific match index
            temp_name_rank = f"{TEMP_LEADER_NAME}_{rank + 1}"
            np.save(DATA_DIR / f'pitch_{temp_name_rank}.npy', final_pitch_arr[:N_frames].reshape(N_chunks, N_frames))
            np.save(DATA_DIR / f'loudness_{temp_name_rank}.npy',
                    final_loudness_arr[:N_frames].reshape(N_chunks, N_frames))
            np.save(DATA_DIR / f'audio_{temp_name_rank}.npy', y_final_audio.reshape(N_chunks, final_len_samples))

            # 5. Execute generation
            neural_band.LEADER = temp_name_rank
            neural_band.GENERATE_SECONDS = final_audio_len_s
            neural_band.MANUAL_START_INDEX = 0

            print(f"\n   -> RANK {rank + 1}: Index {best_idx} (Score: {match['score']:.4f})")
            print(f"      Running arrangement on {neural_band.LEADER}...")
            neural_band.run_complete_arrangement()

            # 6. Rename the file after saving
            default_out_name = f"Smart_Arrangement_{temp_name_rank}_Unison_0.wav"
            default_out_path = OUTPUT_DIR / default_out_name
            corrected_out_name = f"Rank{rank + 1}_IDX_{best_idx}_LEN_{final_audio_len_s}s_Score_{match['score']:.4f}.wav"
            corrected_out_path = OUTPUT_DIR / corrected_out_name

            if default_out_path.exists():
                shutil.move(default_out_path, corrected_out_path)
                default_report_path = default_out_path.with_suffix('.png')
                if default_report_path.exists():
                    shutil.move(default_report_path, corrected_out_path.with_suffix('.png'))
                print(f"   Renamed Output: {corrected_out_name}")

            # Clean up the specific temporary files used for this rank
            cleanup_temp_files(temp_name_rank)

        print("\nAudio RAG Pipeline Execution Complete (Top K Generated)")

    except Exception as e:
        print(f"Error during RAG execution: {e}")

    finally:
        
        pass

if __name__ == "__main__":
    run_audio_rag()
