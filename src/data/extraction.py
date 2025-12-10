# src/data/extraction.py
import pretty_midi
import numpy as np


def extract_ddsp_features(midi_path, frame_rate=250):
    """
    Extracts frame-wise pitch (f0) and loudness (L) from a MIDI file
    Returns:
        pitch_hz (np.array): Fundamental frequency in Hz
        loudness_env (np.array): Normalized loudness (0-1)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None, None

    if not midi_data.instruments:
        return None, None

    end_time = midi_data.get_end_time()
    n_frames = int(np.ceil(end_time * frame_rate))

    pitch_hz = np.zeros(n_frames, dtype=np.float32)
    loudness_env = np.zeros(n_frames, dtype=np.float32)

    instrument = midi_data.instruments[0]

    for note in instrument.notes:
        f0 = pretty_midi.note_number_to_hz(note.pitch)
        L_value = note.velocity / 127.0

        start_idx = int(np.floor(note.start * frame_rate))
        end_idx = int(np.ceil(note.end * frame_rate))

        # Boundary check
        end_idx = min(end_idx, n_frames)
        start_idx = min(start_idx, n_frames)

        pitch_hz[start_idx:end_idx] = f0
        loudness_env[start_idx:end_idx] = L_value

    return pitch_hz, loudness_env
