import torch
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
import os
import math
import random
import sys
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import scipy.signal

from src.models.train_instrument import AudioSynthTrainer
from src.models.decoder_conductor import NeuralArranger
from src.models.signal_processing import harmonic_synthesis, noise_synthesis
from src.visualization.song_in_detail import save_report_card


# Mention the arranger checkpoint file location

#ARRANGER_CKPT = "Masked_Path/src/models/lightning_logs/version_6/checkpoints/Arranger-Guitar-epoch=93-train_loss=0.0667.ckpt"
#ARRANGER_CKPT = "Masked_Path/lightning_logs/version_4/checkpoints/Arranger-Guitar-epoch=27-train_loss=0.1695.ckpt"
ARRANGER_CKPT = "Masked_Path/lightning_logs/version_6/checkpoints/Arranger-Guitar-epoch=96-train_loss=0.1359.ckpt"

# Mention the individual instrument checkpoint file location

MODEL_REGISTRY = {
    'Bass': "Masked_Path/checkpoints/Bass.ckpt",
#    'Reed': "Masked_Path/src/models/lightning_logs/version_4/checkpoints/Reed-epoch=15-train_loss=0.4762.ckpt"
}

# Configuration for the instruments

LEADER = 'Guitar'
ARRANGER_FOLLOWERS = ['Bass']

# To separate Bass & Drums, since Drums possess no Pitch & require further processing

ARRANGER_MAP = {
    'Bass': 0,
    'Drums': 1
}

NUM_ARRANGER_CHANNELS = 2

# DRUM CONFIG

ADD_DRUMS = True
DRUM_VOLUME = 2

# Main parameters

MANUAL_START_INDEX = 4000
GENERATE_SECONDS = 20
NOISE_SCALE = 0.05
CONST_WINDOW = 4.0


# Function definition

def load_audio_model(ckpt_path):
    if not os.path.exists(ckpt_path): sys.exit(f"Missing Model: {ckpt_path}")
    model = AudioSynthTrainer.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def load_arranger_model(ckpt_path):
    if not os.path.exists(ckpt_path): sys.exit(f"Missing Arranger: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = NeuralArranger(num_followers=NUM_ARRANGER_CHANNELS)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def audio_to_chroma(audio_waveform):
    chromagram = librosa.feature.chroma_cqt(y=audio_waveform, sr=16000, hop_length=512)
    chroma_vector = np.mean(chromagram, axis=1)
    norm = np.linalg.norm(chroma_vector)
    return chroma_vector / norm if norm > 0 else chroma_vector


def pitch_to_chroma(f0_curve):
    active_f0 = f0_curve[f0_curve > 40]
    if len(active_f0) == 0: return np.zeros(12)
    midi_notes = 69 + 12 * np.log2(active_f0 / 440.0)
    chroma_vals = np.mod(np.round(midi_notes), 12).astype(int)
    chroma_vector = np.bincount(chroma_vals, minlength=12)
    norm = np.linalg.norm(chroma_vector)
    return chroma_vector / norm if norm > 0 else chroma_vector


def find_best_match_smart(leader_audio, candidate_pitches, candidate_loudness):
    leader_chroma = audio_to_chroma(leader_audio).reshape(1, -1)
    best_idx, best_score, best_shift = 0, -1.0, 0
    total = len(candidate_pitches)
    indices = random.sample(range(total), min(2000, total))
    for idx in indices:
        if np.mean(candidate_loudness[idx]) < 0.1: continue
        cand_chroma = pitch_to_chroma(candidate_pitches[idx])
        for shift in range(-6, 7):
            shifted = np.roll(cand_chroma, shift).reshape(1, -1)
            score = cosine_similarity(leader_chroma, shifted)[0][0]
            if score > best_score:
                best_score, best_idx, best_shift = score, idx, shift
    return best_idx, best_shift


def find_rhythmic_match(leader_loudness, drum_loudness_flat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Feature Engineering: Prioritize Rhythmic Onsets
    
    # Calculate the differentiated loudness curve
    # The derivative highlights where the energy *changes* rapidly
    lead_env_raw = leader_loudness.flatten()

    # Use padding to keep the size consistent, then calculate the difference
    # This acts like a high-pass filter on the energy envelope
    lead_env_diff = np.diff(lead_env_raw, prepend=lead_env_raw[0])

    # Remove DC offset and center the differential curve
    lead_env = lead_env_diff - np.mean(lead_env_diff)

    # We must also apply this to the drum input for a fair correlation
    drum_env_raw = drum_loudness_flat.flatten()
    drum_env_diff = np.diff(drum_env_raw, prepend=drum_env_raw[0])
    
    # Center the differential drum curve as well, we need to check this
    #drum_input_data = drum_env_diff - np.mean(drum_env_diff)
    
    drum_input_data = drum_env_diff

    # Convolution
    kernel = torch.from_numpy(lead_env).float().view(1, 1, -1).to(device)
    drum_input = torch.from_numpy(drum_input_data).float().view(1, 1, -1).to(device)

    with torch.no_grad():
        # F.conv1d now correlates the *rhythmic changes*
        correlation_map = F.conv1d(drum_input, kernel)

    best_idx = correlation_map.argmax().item()
    return best_idx

def boost_dynamics(curve, threshold=0.4, ratio=4.0):
    centered = curve - threshold
    expanded = centered * ratio
    result = expanded + threshold
    return torch.clamp(result, 0.0, 1.0)


def simplify_pitch(pitch_curve, factor=8):
    p_small = pitch_curve[:, ::factor, :]
    p_blocky = F.interpolate(p_small.permute(0, 2, 1), size=pitch_curve.shape[1], mode='nearest')
    return p_blocky.permute(0, 2, 1)


def force_octave(pitch_curve, target_hz=100.0):
    avg_pitch = torch.mean(pitch_curve[pitch_curve > 40])
    if torch.isnan(avg_pitch) or avg_pitch == 0: return pitch_curve
    num_octaves = torch.round(torch.log2(target_hz / avg_pitch))
    return pitch_curve * (2.0 ** num_octaves)


def run_complete_arrangement():
    print(f"Starting Complete AI Arranger (Leader: {LEADER})...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LOAD DATA
    project_root = Path(__file__).resolve().parents[2]
    if project_root.name != 'project_111': project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'processed' / 'features'

    datasets = {}
    load_list = [LEADER] + list(MODEL_REGISTRY.keys()) + (['Drums'] if ADD_DRUMS else [])

    for inst in load_list:
        print(f"Loading {inst}...", flush=True)
        try:
            datasets[inst] = {
                'pitch': np.load(data_path / f'pitch_{inst}.npy'),
                'loudness': np.load(data_path / f'loudness_{inst}.npy'),
                'audio': np.load(data_path / f'audio_{inst}.npy')
            }
        except FileNotFoundError:
            return

    # 2. PREPARE LEADER
    num_chunks = math.ceil(GENERATE_SECONDS / CONST_WINDOW)
    leader_len = len(datasets[LEADER]['pitch'])

    if MANUAL_START_INDEX is not None:
        start_idx = MANUAL_START_INDEX
    else:
        start_idx = random.randint(0, leader_len - num_chunks - 1)

    print("Normalizing Leader Audio...")
    a_lead = datasets[LEADER]['audio'][start_idx: start_idx + num_chunks].reshape(-1)
    max_val = np.max(np.abs(a_lead))
    if max_val > 0:
        a_lead = a_lead / max_val * 0.9

    p_lead_np = datasets[LEADER]['pitch'][start_idx: start_idx + num_chunks].reshape(-1)
    l_lead_np = datasets[LEADER]['loudness'][start_idx: start_idx + num_chunks].reshape(-1)

    final_mix_len = len(a_lead)
    final_mix = np.zeros(final_mix_len)
    final_mix += a_lead

    l_leader_t = torch.from_numpy(l_lead_np).unsqueeze(0).unsqueeze(-1).to(device)

    # 3. ARRANGER
    print("Generating Dynamics...", flush=True)
    arranger = load_arranger_model(ARRANGER_CKPT).to(device)
    with torch.no_grad():
        predicted_dynamics = arranger(l_leader_t)
        
    # Output the reference audio file        

    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    sf.write(output_dir / f"Reference_{LEADER}_idx{start_idx}.wav", a_lead, 16000)

    # 4. DRUMS
    if ADD_DRUMS:
        print("Recording Drums...", flush=True)
        drum_l_flat = datasets['Drums']['loudness'].reshape(-1)  #Pitch is not available for Drums
        drum_a_flat = datasets['Drums']['audio'].reshape(-1)

        # 1. CNN Match
        idx = find_rhythmic_match(l_lead_np, drum_l_flat)

        # 2. SLICE
        start_sample = idx * 64
        req_len = len(final_mix)

        if start_sample + req_len > len(drum_a_flat):
            part1 = drum_a_flat[start_sample:]
            remaining = req_len - len(part1)
            if remaining > len(drum_a_flat):
                part2 = np.tile(drum_a_flat, math.ceil(remaining / len(drum_a_flat)))[:remaining]
            else:
                part2 = drum_a_flat[:remaining]
            drum_track = np.concatenate([part1, part2])
        else:
            drum_track = drum_a_flat[start_sample: start_sample + req_len]

        # 3. APPLY ARRANGER
        if 'Drums' in ARRANGER_MAP:
            idx_map = ARRANGER_MAP['Drums']
            vol_curve = predicted_dynamics[0, :, idx_map].cpu().numpy()
            vol_audio = np.repeat(vol_curve, 64)

            if len(vol_audio) > len(drum_track):
                vol_audio = vol_audio[:len(drum_track)]
            elif len(vol_audio) < len(drum_track):
                vol_audio = np.pad(vol_audio, (0, len(drum_track) - len(vol_audio)), mode='edge')

            # Multiply and not mask
            vol_audio = np.clip(vol_audio * 1.5, 0.0, 1.0)

            # Force min volume if needed
            if vol_audio.mean() < 0.05: vol_audio = np.maximum(vol_audio, 0.2)

            drum_track = drum_track * vol_audio * DRUM_VOLUME

        final_mix += drum_track
        sf.write(output_dir / f"complete_stem_drums_{start_idx}.wav", drum_track, 16000)

    # Other INSTRUMENTS
    frames_per_chunk = int(CONST_WINDOW * 250)
    samples_per_chunk = int(CONST_WINDOW * 16000)

    # HELPER: FREQUENCY MAP
    # Maps Chroma Index (0=C, 1=C#, etc.) to Low Bass Frequencies
    # We aim for the octave between 40Hz and 80Hz
    BASS_FREQS = [
        32.70, 34.65, 36.71, 38.89, 41.20, 43.65,  # C  to F
        46.25, 49.00, 51.91, 55.00, 58.27, 61.74  # F# to B
    ]

    for inst_name in MODEL_REGISTRY.keys():
        print(f"Recording {inst_name}...", flush=True)

        model = load_audio_model(MODEL_REGISTRY[inst_name]).to(device)
        inst_audio_list = []

        # Dynamics Setup
        if inst_name in ARRANGER_MAP:
            idx = ARRANGER_MAP[inst_name]
            l_generated = predicted_dynamics[:, :, idx].unsqueeze(-1)
        else:
            l_generated = l_leader_t * 0.8
        l_generated = boost_dynamics(l_generated, threshold=0.4, ratio=4.0)

        for c in range(num_chunks):
            # SLICING
            start_s = c * samples_per_chunk
            end_s = start_s + samples_per_chunk
            if start_s >= len(a_lead): break
            leader_audio_chunk = a_lead[start_s:end_s]

            start_f = c * frames_per_chunk
            end_f = start_f + frames_per_chunk
            l_chunk = l_generated[:, start_f:end_f, :]  # Target Volume

            
            if inst_name == 'Bass':
                # Shadow the Guitar, as found while listening, this is better
                p_raw = p_lead_np[start_f:end_f]

                # Smoothening
                p_smooth = scipy.signal.medfilt(p_raw, kernel_size=15)

                # Force Sub-Bass
                p_t = torch.from_numpy(p_smooth).float().reshape(1, -1, 1).to(device)
                p_t = force_octave(p_t, target_hz=55.0)

                # We start with the Arranger's curve
                l_raw = l_chunk

                # Lock to Guitar Rhythm, for Bass this is better
                l_guitar_ref = l_lead_np[start_f:end_f]
                l_guitar_t = torch.from_numpy(l_guitar_ref).float().reshape(1, -1, 1).to(device)
                gate = (l_guitar_t > 0.05).float()
                l_raw = l_raw * gate

                # Normalize the chunk so the peak is exactly 1.0
                max_vol = torch.max(l_raw)
                if max_vol > 0:
                    l_raw = l_raw / max_vol

                # 0.6 is a safe "Shadow" level
                # Since we normalized to 1.0, we KNOW the output will peak at exactly 0.6
                l_chunk = l_raw * 0.6

                noise_scale = 0.001

            else:
                
                #Here we are matching the tune of other instruments with leader guitar using cosine similarity
                best_idx, shift = find_best_match_smart(leader_audio_chunk,
                                                        datasets[inst_name]['pitch'],
                                                        datasets[inst_name]['loudness'])

                # Handle Flat Indexing (from previous fix)
                p_flat = datasets[inst_name]['pitch'].reshape(-1)
                chunk_len_db = datasets[inst_name]['pitch'].shape[1] if len(
                    datasets[inst_name]['pitch'].shape) > 1 else 1000
                current_frame_idx = best_idx * chunk_len_db

                req_frames = end_f - start_f
                if current_frame_idx + req_frames > len(p_flat):
                    p_chunk = p_flat[current_frame_idx:]
                    p_chunk = np.concatenate([p_chunk, p_flat[:(req_frames - len(p_chunk))]])
                else:
                    p_chunk = p_flat[current_frame_idx: current_frame_idx + req_frames]

                p_chunk = p_chunk * (2 ** (shift / 12.0))
                p_t = torch.from_numpy(p_chunk).float().reshape(1, -1, 1).to(device)
                
                noise_scale = NOISE_SCALE

            # SYNTHESIS
            with torch.no_grad():
                controls = model.model(p_t, l_chunk)
                controls['noise'] *= noise_scale
                target_samples = p_t.shape[1] * 64
                harm = harmonic_synthesis(p_t, controls['amplitudes'], controls['harmonics'],
                                          n_samples=target_samples)
                nz = noise_synthesis(controls['noise'], n_samples=target_samples)
                chunk_out = (harm + nz).squeeze().cpu().numpy()
                inst_audio_list.append(chunk_out)

        track_audio = np.concatenate(inst_audio_list)
        min_len = min(len(final_mix), len(track_audio))
        final_mix[:min_len] += track_audio[:min_len]
        sf.write(output_dir / f"complete_stem_{inst_name}_{start_idx}.wav", track_audio, 16000)

    # 6. SAVE
    out_name = f"Smart_Arrangement_{LEADER}_Unison_{start_idx}.wav"
    final_path = output_dir / out_name
    sf.write(final_path, final_mix, 16000)
    print(f"\nDONE! Output: {final_path}")
    try:
        save_report_card(str(final_path))
    except:
        pass


if __name__ == "__main__":
    run_complete_arrangement()
