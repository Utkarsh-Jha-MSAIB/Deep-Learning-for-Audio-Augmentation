import os
import requests
import tarfile
import zipfile
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm


# Other options
# 'baby' = BabySlakh
# 'lsx'  = LSX Full

DATASET_MODE = 'lsx'


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'slakh'

DATASETS = {
    'baby': {
        'url': "https://zenodo.org/records/4603870/files/babyslakh_16k.tar.gz?download=1",
        'filename': "babyslakh_16k.tar.gz",
        'extract_folder': "babyslakh_16k",
        'type': 'tar'
    },
    'lsx': {
        # The MASTER file containing Train, Valid, and Test sets
        'url': "https://zenodo.org/records/7765140/files/lsx.zip?download=1",
        'filename': "lsx.zip",
        'extract_folder': "lsx",
        'type': 'zip'
    }
}


def download_file(url, dest_path):
    print(f"⬇️ Downloading {dest_path.name}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 * 1024  

            with open(dest_path, 'wb') as file, tqdm(
                    total=total_size, unit='iB', unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    bar.update(len(chunk))
                    file.write(chunk)
        print(" Download complete.")
        return True
    except Exception as e:
        print(f" Download Failed: {e}")
        return False


def create_dummy_metadata(track_path):
    """Generates metadata.yaml so preprocess.py works."""
    metadata = {
        'stems': {
            'Bass': {'inst_class': 'Bass', 'program_num': 33},
            'Guitar': {'inst_class': 'Guitar', 'program_num': 25},
            'Piano': {'inst_class': 'Piano', 'program_num': 0},
            'Drums': {'inst_class': 'Drums', 'program_num': 0}
        }
    }
    with open(track_path / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)


def extract_archive(archive_path, extract_to, archive_type):
    print("Extracting (This will take time)...")
    try:
        if archive_type == 'tar':
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_to)
        elif archive_type == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(path=extract_to)
        return True
    except Exception as e:
        print(f"Extraction Failed: {e}")
        return False


def organize_lsx(extract_root):
    print(f"   Converting LSX format to Slakh format...")

    # LSX extracts into 'lsx' folder with subfolders 'tr' (Train), 'cv' (Valid), 'tt' (Test)
    source_root = extract_root / 'lsx'
    if not source_root.exists():
        print("   Extracted 'lsx' folder not found, checking root...")
        source_root = extract_root

    found_tracks = 0

    # We look for the split folders
    splits = ['tr', 'cv', 'tt']

    # Collect all song folders from all splits
    song_folders = []
    for split in splits:
        split_path = source_root / split
        if split_path.exists():
            song_folders.extend(list(split_path.iterdir()))

    # Also check if they are just flat in root (fallback)
    if not song_folders:
        song_folders = list(source_root.iterdir())

    print(f"   Processing {len(song_folders)} potential tracks...")

    for song_folder in tqdm(song_folders):
        if not song_folder.is_dir(): continue
        # LSX folder names are numbers like '00023'
        if not song_folder.name.isdigit(): continue

        # 1. Create new Track ID
        track_id = f"Track_LSX_{song_folder.name}"
        new_track_path = extract_root / track_id
        new_stems_path = new_track_path / 'stems'

        if new_track_path.exists(): continue

        new_stems_path.mkdir(parents=True, exist_ok=True)

        # 2. Move & Rename Stems
        # LSX files are 'bass.wav'. We need 'Bass.wav'
        for audio_file in song_folder.glob('*.wav'):
            # Skip mix files
            if 'mix' in audio_file.name: continue

            clean_name = audio_file.stem.capitalize() + ".wav"
            shutil.move(str(audio_file), str(new_stems_path / clean_name))

        # 3. Metadata
        create_dummy_metadata(new_track_path)
        (new_track_path / 'all_src.mid').touch()
        found_tracks += 1

    # Cleanup the empty 'lsx' folder
    if (extract_root / 'lsx').exists():
        shutil.rmtree(extract_root / 'lsx')

    print(f"Organized {found_tracks} tracks into Slakh format.")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    config = DATASETS[DATASET_MODE]
    archive_path = RAW_DIR / config['filename']

    # 1. Download
    if not archive_path.exists():
        success = download_file(config['url'], archive_path)
        if not success: return
    else:
        print(f"Found existing archive: {archive_path}")

    # 2. Extract
    success = extract_archive(archive_path, RAW_DIR, config['type'])
    if not success: return

    # 3. Organize
    if DATASET_MODE == 'lsx':
        organize_lsx(RAW_DIR)
    elif DATASET_MODE == 'baby':
        src = RAW_DIR / config['extract_folder']
        if src.exists():
            for item in src.iterdir():
                if not (RAW_DIR / item.name).exists():
                    shutil.move(str(item), str(RAW_DIR / item.name))
            src.rmdir()

    print(f"\nSuccess! Data is ready in: {RAW_DIR}")
    print("   Next: Run 'python src/data/preprocess_band.py' to ingest this new data.")


if __name__ == "__main__":
    main()