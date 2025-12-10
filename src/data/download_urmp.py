import os
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'urmp'

# URMP 16kHz (Audio Only) - Hosted on Zenodo
URL = "https://zenodo.org/records/8021437/files/urmp_yourmt3_16k.tar.gz?download=1"
ARCHIVE_NAME = "urmp_yourmt3_16k.tar.gz"


def download_file(url, dest_path):
    print(f"Downloading URMP (521 MB)...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB

            with open(dest_path, 'wb') as file, tqdm(
                    total=total_size, unit='iB', unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    bar.update(len(chunk))
                    file.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download Failed: {e}")
        return False


def organize_urmp(extract_root):
    """
    Flattens the messy URMP structure into clean Instrument folders
    Original: urmp/01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter.wav
    Target:   urmp/Strings/01_Jupiter_Violin.wav
    """
    print("Organizing Stems...")

    # Map URMP codes to our Class Names
    inst_map = {
        'vn': 'Strings', 'va': 'Strings', 'vc': 'Strings', 'db': 'Strings',
        'fl': 'Pipe', 'ob': 'Reed', 'cl': 'Reed', 'sax': 'Reed',
        'tpt': 'Brass', 'tbn': 'Brass', 'hn': 'Brass', 'tba': 'Brass'
    }

    # Create folders
    for family in set(inst_map.values()):
        (extract_root / family).mkdir(parents=True, exist_ok=True)

    source_root = extract_root / 'urmp'
    if not source_root.exists(): source_root = extract_root

    count = 0
    # Checking through all the song folders
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith(".wav") and "AuSep" in file:
                # Filename format: AuSep_1_vn_01_Jupiter.wav
                parts = file.split('_')

                # The instrument code is usually index 2 (vn, vc, etc.)
                # But sometimes the numbering changes, so we search for the code
                found_code = None
                for p in parts:
                    if p in inst_map:
                        found_code = p
                        break

                if found_code:
                    family = inst_map[found_code]
                    # Clean name: Strings/Jupiter_vn.wav
                    new_name = f"{'_'.join(parts[3:])}"

                    # Move it
                    shutil.move(os.path.join(root, file), extract_root / family / new_name)
                    count += 1

    # Cleanup the empty extraction folder
    if (extract_root / 'urmp').exists():
        shutil.rmtree(extract_root / 'urmp')

    print(f"Organized {count} stems into {RAW_DIR}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / ARCHIVE_NAME

    # 1. Download
    if not archive_path.exists():
        success = download_file(URL, archive_path)
        if not success: return
    else:
        print(f"Found existing archive: {archive_path}")

    # 2. Extract
    print("Extracting...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=RAW_DIR)
    except Exception as e:
        print(f"Extraction Failed: {e}")
        return

    # 3. Organize
    organize_urmp(RAW_DIR)


if __name__ == "__main__":
    main()