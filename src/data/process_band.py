import os
import pretty_midi
import numpy as np
import librosa
import yaml
import torch
import gc
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

# Configuration for the 1,700 LSX songs
REQUIRED_INSTRUMENTS = ['Guitar', 'Bass', 'Drums']

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'slakh'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'aligned_band'


def get_stem_info(track_folder, target_class):
    yaml_path = track_folder / 'metadata.yaml'
    if not yaml_path.exists(): return None, None
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    stems = metadata.get('stems', {})
    for stem_name, info in stems.items():
        if info.get('inst_class') == target_class:
            return f"{stem_name}.wav", info.get('program_num')
    return None, None


def extract_features_from_audio(audio, sr=16000, skip_pitch=False):
    """Extract Pitch/Loudness using GPU if available"""
    try:
        # Loudness
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=64, center=True)[0]
        rms = rms / (np.max(rms) + 1e-7)

        # Pitch
        if skip_pitch:
            # Optimization: Drums don't need pitch!
            f0 = np.zeros_like(rms)
        elif HAS_CREPE and torch.cuda.is_available():
            # GPU Mode
            device = 'cuda'
            audio_t = torch.tensor(audio).unsqueeze(0).to(device)
            f0, confidence = torchcrepe.predict(
                audio_t, sr, hop_length=64, fmin=40, fmax=2000, model='tiny', batch_size=2048, device=device,
                return_periodicity=True
            )
            f0 = f0.squeeze().cpu().numpy()
            f0[confidence.squeeze().cpu().numpy() < 0.4] = 0.0
            f0 = np.nan_to_num(f0)
        else:
            # CPU Mode
            f0, _, _ = librosa.pyin(audio, fmin=40, fmax=2000, sr=sr, frame_length=2048, hop_length=64)
            f0 = np.nan_to_num(f0)

        min_len = min(len(f0), len(rms))
        return f0[:min_len], rms[:min_len]
    except:
        return None, None


def get_midi_features(midi_path, duration_sec, target_program_id):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except:
        return None, None
    target_instrument = None
    for instrument in pm.instruments:
        if instrument.is_drum: continue
        if instrument.program == target_program_id:
            target_instrument = instrument
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


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting Aligned Preprocessing for: {REQUIRED_INSTRUMENTS}")

    track_folders = [d for d in RAW_DIR.iterdir() if d.is_dir() and (d.name.startswith('Track'))]
    track_folders.sort()

    aligned_data = {inst: {'pitch': [], 'loudness': [], 'audio': []} for inst in REQUIRED_INSTRUMENTS}
    valid_songs = 0

    for track_folder in tqdm(track_folders):
        song_files = {}
        missing_member = False
        for inst in REQUIRED_INSTRUMENTS:
            audio_name, prog_id = get_stem_info(track_folder, inst)
            if audio_name is None:
                missing_member = True;
                break
            song_files[inst] = (audio_name, prog_id)
        if missing_member: continue

        midi_files = list(track_folder.glob('*.mid')) + list(track_folder.glob('MIDI/*.mid')) + list(
            track_folder.glob('all_src.mid'))
        midi_path = midi_files[0] if midi_files else None

        # 1. Load Leader Audio
        leader_inst = REQUIRED_INSTRUMENTS[0]
        leader_file = song_files[leader_inst][0]
        try:
            leader_path = track_folder / 'stems' / leader_file
            leader_audio, _ = librosa.load(str(leader_path), sr=SAMPLE_RATE)
            song_duration = len(leader_audio) / SAMPLE_RATE
        except:
            continue

        song_chunks = {inst: {'p': [], 'l': [], 'a': []} for inst in REQUIRED_INSTRUMENTS}
        success = True

        for inst in REQUIRED_INSTRUMENTS:
            fname, pid = song_files[inst]
            try:
                # 2. Reuse Audio
                if inst == leader_inst:
                    audio = leader_audio
                else:
                    audio, _ = librosa.load(str(track_folder / 'stems' / fname), sr=SAMPLE_RATE)
                    if len(audio) != len(leader_audio):
                        audio = librosa.util.fix_length(audio, size=len(leader_audio))

                # 3. Try MIDI
                f0, ld = None, None
                if inst != 'Drums' and midi_path and midi_path.stat().st_size > 100:
                    f0, ld = get_midi_features(midi_path, song_duration, pid)

                # 4. Fallback
                if f0 is None:
                    f0, ld = extract_features_from_audio(audio, skip_pitch=(inst == 'Drums'))

                if f0 is None:
                    success = False;
                    break

                n_windows = len(f0) // WINDOW_SIZE_CONTROL
                for i in range(n_windows):
                    start_c = i * WINDOW_SIZE_CONTROL
                    start_a = i * WINDOW_SIZE_AUDIO
                    song_chunks[inst]['p'].append(f0[start_c: start_c + WINDOW_SIZE_CONTROL])
                    song_chunks[inst]['l'].append(ld[start_c: start_c + WINDOW_SIZE_CONTROL])
                    song_chunks[inst]['a'].append(audio[start_a: start_a + WINDOW_SIZE_AUDIO])
            except:
                success = False;
                break

        if success:
            valid_songs += 1
            for inst in REQUIRED_INSTRUMENTS:
                aligned_data[inst]['pitch'].extend(song_chunks[inst]['p'])
                aligned_data[inst]['loudness'].extend(song_chunks[inst]['l'])
                aligned_data[inst]['audio'].extend(song_chunks[inst]['a'])

        if valid_songs % 10 == 0: gc.collect()

    print(f"\nAlignment Complete! Found {valid_songs} valid songs")

    for inst in REQUIRED_INSTRUMENTS:
        X_p = np.array(aligned_data[inst]['pitch'], dtype=np.float32)
        X_l = np.array(aligned_data[inst]['loudness'], dtype=np.float32)
        X_a = np.array(aligned_data[inst]['audio'], dtype=np.float32)

        np.save(PROCESSED_DIR / f'pitch_{inst}.npy', X_p)
        np.save(PROCESSED_DIR / f'loudness_{inst}.npy', X_l)
        np.save(PROCESSED_DIR / f'audio_{inst}.npy', X_a)
        print(f"   Saved {inst}: {len(X_p)} chunks")


if __name__ == "__main__":
    main()
