import os
import pretty_midi
import numpy as np
import librosa
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

# Try importing torchcrepe for GPU acceleration
try:
    import torchcrepe

    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

# Configuration
FRAME_RATE = 250
SAMPLE_RATE = 16000
WINDOW_SECONDS = 4.0
WINDOW_SIZE_CONTROL = int(FRAME_RATE * WINDOW_SECONDS)
WINDOW_SIZE_AUDIO = int(SAMPLE_RATE * WINDOW_SECONDS)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'features'

INSTRUMENT_MAP = {
    'Piano': (0, 7),
    'Guitar': (24, 31),
    'Bass': (32, 39),
    'Strings': (40, 55),
    'Brass': (56, 63),
    'Reed': (64, 71),
    'Pipe': (72, 79),
    'Drums': (-1, -1)
}


def get_slakh_info(track_folder, target_class):
    yaml_path = track_folder / 'metadata.yaml'
    if not yaml_path.exists(): return None, None
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)
    stems = metadata.get('stems', {})
    for stem_name, info in stems.items():
        if info.get('inst_class') == target_class:
            return track_folder / 'stems' / f"{stem_name}.wav", info.get('program_num')
    return None, None


def extract_features_from_audio(audio_path):
    """
    Hybrid Extractor:
    - Uses GPU (TorchCrepe) for Pitch if available (FAST)
    - Falls back to Librosa (CPU) if not
    """
    try:
        # 1. Load Audio
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)

        # 2. Extract Loudness (CPU is fast enough for this)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=64, center=True)[0]
        rms = rms / (np.max(rms) + 1e-7)

        # 3. Extract Pitch (The Bottleneck)
        if HAS_CREPE and torch.cuda.is_available():
            # GPU Mode
            device = 'cuda'
            # Convert to tensor
            audio_t = torch.tensor(audio).unsqueeze(0).to(device)

            # Crepe requires 16k audio. Hop length must be matching
            # Our target is 250Hz (16000 / 64 = 250)
            # Crepe hop_length is in samples
            hop_length = 64

            # Predict Pitch (f0) and Confidence
            f0, confidence = torchcrepe.predict(
                audio_t,
                sample_rate=SAMPLE_RATE,
                hop_length=hop_length,
                fmin=40,
                fmax=2000,
                model='tiny',
                batch_size=2048,
                device=device,
                return_periodicity=True
            )

            f0 = f0.squeeze().cpu().numpy()
            confidence = confidence.squeeze().cpu().numpy()

            # Filter out low confidence (silence/noise)
            f0[confidence < 0.4] = 0.0
            f0 = np.nan_to_num(f0)

        else:
            # CPU Mode
            f0, _, _ = librosa.pyin(
                audio, fmin=40, fmax=2000,
                sr=SAMPLE_RATE, frame_length=2048, hop_length=64, center=True
            )
            f0 = np.nan_to_num(f0)

        # Align lengths
        min_len = min(len(f0), len(rms))
        return f0[:min_len], rms[:min_len], audio
    except Exception as e:
        # print(f"Extraction Error: {e}")
        return None, None, None


def get_midi_features(midi_path, duration_sec, target_program_id):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except:
        return None, None

    target_instrument = None
    if target_program_id is not None:
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            if instrument.program == target_program_id:
                target_instrument = instrument;
                break
    else:
        for instrument in pm.instruments:
            if not instrument.is_drum:
                target_instrument = instrument;
                break

    if target_instrument is None: return None, None

    times = np.linspace(0, duration_sec, int(duration_sec * FRAME_RATE), endpoint=False)
    piano_roll = target_instrument.get_piano_roll(fs=FRAME_RATE)
    f0 = np.zeros_like(times)
    loudness = np.zeros_like(times)

    if piano_roll.shape[1] > 0:
        current_len = piano_roll.shape[1]
        target_len = len(times)
        if current_len > target_len:
            piano_roll = piano_roll[:, :target_len]
        elif current_len < target_len:
            padding = np.zeros((128, target_len - current_len))
            piano_roll = np.hstack([piano_roll, padding])

        active_notes = np.argmax(piano_roll, axis=0)
        velocities = np.max(piano_roll, axis=0)
        has_note = active_notes > 0
        if np.any(has_note):
            f0[has_note] = 440.0 * (2.0 ** ((active_notes[has_note] - 69.0) / 12.0))
        loudness = velocities / 127.0
    return f0, loudness


def process_instrument(target_instrument):
    print(f"\nProcessing: {target_instrument}")
    all_pitch, all_loudness, all_audio = [], [], []

    # SOURCE 1: SLakh LSX
    slakh_dir = PROJECT_ROOT / 'data' / 'raw' / 'slakh'
    if slakh_dir.exists():
        track_folders = [d for d in slakh_dir.iterdir() if
                         d.is_dir() and (d.name.startswith('Track') or d.name.startswith('Track_LSX'))]

        for track_folder in tqdm(track_folders, desc="Scanning Slakh"):
            audio_path, program_id = get_slakh_info(track_folder, target_instrument)
            if not audio_path or not audio_path.exists(): continue

            # Try MIDI first since faster
            f0, ld, audio_data = None, None, None

            if target_instrument != 'Drums':  #No pitch for drums
                midi_files = list(track_folder.glob('*.mid')) + list(track_folder.glob('MIDI/*.mid')) + list(
                    track_folder.glob('all_src.mid'))
                if midi_files:
                    try:
                        if midi_files[0].stat().st_size > 100:
                            temp_audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)
                            duration = len(temp_audio) / SAMPLE_RATE
                            f0, ld = get_midi_features(midi_files[0], duration, program_id)
                            if f0 is not None: audio_data = temp_audio
                    except:
                        pass

            # Fallback to Audio Extraction (GPU Accelerated)
            if f0 is None:
                if target_instrument == 'Drums':
                    try:
                        f0, ld, audio_data = extract_features_from_audio(audio_path)
                        f0 = np.zeros_like(ld)
                    except:
                        pass
                else:
                    f0, ld, audio_data = extract_features_from_audio(audio_path)

            if f0 is None or ld is None: continue

            # Windowing
            n_windows = len(f0) // WINDOW_SIZE_CONTROL
            for i in range(n_windows):
                start_c = i * WINDOW_SIZE_CONTROL
                start_a = i * WINDOW_SIZE_AUDIO
                if start_a + WINDOW_SIZE_AUDIO <= len(audio_data):
                    all_pitch.append(f0[start_c: start_c + WINDOW_SIZE_CONTROL])
                    all_loudness.append(ld[start_c: start_c + WINDOW_SIZE_CONTROL])
                    all_audio.append(audio_data[start_a: start_a + WINDOW_SIZE_AUDIO])

    # Source 2: URMP
    urmp_dir = PROJECT_ROOT / 'data' / 'raw' / 'urmp' / target_instrument
    if urmp_dir.exists():
        audio_files = list(urmp_dir.glob('*.wav'))
        for audio_path in tqdm(audio_files, desc="Scanning URMP"):
            f0, ld, audio_data = extract_features_from_audio(audio_path)
            if f0 is None: continue

            n_windows = len(f0) // WINDOW_SIZE_CONTROL
            for i in range(n_windows):
                start_c = i * WINDOW_SIZE_CONTROL
                start_a = i * WINDOW_SIZE_AUDIO
                if start_a + WINDOW_SIZE_AUDIO <= len(audio_data):
                    all_pitch.append(f0[start_c: start_c + WINDOW_SIZE_CONTROL])
                    all_loudness.append(ld[start_c: start_c + WINDOW_SIZE_CONTROL])
                    all_audio.append(audio_data[start_a: start_a + WINDOW_SIZE_AUDIO])

    # Save the files
    if len(all_pitch) > 0:
        X_p = np.array(all_pitch, dtype=np.float32)
        X_l = np.array(all_loudness, dtype=np.float32)
        X_a = np.array(all_audio, dtype=np.float32)

        np.save(PROCESSED_DIR / f'pitch_{target_instrument}.npy', X_p)
        np.save(PROCESSED_DIR / f'loudness_{target_instrument}.npy', X_l)
        np.save(PROCESSED_DIR / f'audio_{target_instrument}.npy', X_a)
        print(f"Saved {len(X_p)} examples")
    else:
        print(f"No data files found")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting Preprocessing...")

    if HAS_CREPE:
        print("   TorchCrepe detected! High-speed extraction enabled")
    else:
        print("   TorchCrepe not found. Falling back to slow CPU extraction")

    for inst_name in INSTRUMENT_MAP.keys():
        process_instrument(inst_name)


if __name__ == "__main__":
    main()