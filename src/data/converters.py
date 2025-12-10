# src/data/converters.py
import music21

def convert_kern_to_midi(krn_path, midi_path):
    """
    Converts a **kern file to a MIDI file using music21
    """
    try:
        score = music21.converter.parse(krn_path)
        score.write('midi', fp=midi_path)
        return True
    except Exception as e:
        print(f"Conversion failed for {krn_path}: {e}")
        return False