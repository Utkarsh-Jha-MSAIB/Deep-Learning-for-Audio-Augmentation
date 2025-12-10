import os
import zipfile
import shutil
import yaml
import gc
import sys
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'slakh'
ZIP_FILE = RAW_DIR / "lsx.zip"

# LIMIT: Only extract this many files per run to prevent crashes
# If it crashes, lower this number (e.g., to 500)
FILES_PER_RUN = 2000


def create_fake_metadata(track_path):
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


def safe_extract_and_organize():
    print(f"🚀 Starting Surgical Extraction (Batch Limit: {FILES_PER_RUN})...")

    if not ZIP_FILE.exists():
        print(f"❌ Error: {ZIP_FILE} not found.")
        return

    extract_root = RAW_DIR

    # Counters
    extracted_this_run = 0
    skipped_existing = 0
    total_processed = 0

    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as z:
            all_files = z.infolist()
            print(f"   Total files in archive: {len(all_files)}")

            # Filter relevant files first to save time
            audio_targets = [
                f for f in all_files
                if f.filename.endswith('.wav')
                   and 'mix' not in f.filename
                   and '/' in f.filename  # Must be in a folder
            ]

            print(f"   Audio stems found: {len(audio_targets)}")

            for member in tqdm(audio_targets, desc="Processing"):
                filename = member.filename

                # --- PARSE FILENAME ---
                # Expect: lsx/tr/00023/bass.wav OR Track00023/bass.wav
                parts = filename.split('/')
                inst_name = parts[-1]  # bass.wav

                # Find the song ID (folder name that is digits)
                song_id = None
                for part in parts:
                    # Check for "00023" or "Track00023"
                    if part.isdigit():
                        song_id = part
                        break
                    elif part.startswith("Track") and part[5:].isdigit():
                        song_id = part[5:]
                        break

                if song_id is None: continue

                # --- SETUP PATHS ---
                track_id = f"Track_LSX_{song_id}"
                track_dir = extract_root / track_id
                stem_dir = track_dir / 'stems'

                clean_name = inst_name.capitalize()  # Bass.wav
                final_path = stem_dir / clean_name

                # --- CHECK IF DONE ---
                if final_path.exists():
                    skipped_existing += 1
                    total_processed += 1
                    continue  # Skip this file, move to next

                # --- EXTRACT ---
                # Create folders only when needed
                stem_dir.mkdir(parents=True, exist_ok=True)

                with z.open(member) as source, open(final_path, "wb") as target:
                    shutil.copyfileobj(source, target)

                # Create Metadata
                if not (track_dir / 'metadata.yaml').exists():
                    create_fake_metadata(track_dir)
                    (track_dir / 'all_src.mid').touch()

                extracted_this_run += 1
                total_processed += 1

                # --- BATCH LIMIT CHECK ---
                if extracted_this_run >= FILES_PER_RUN:
                    print(f"\n🛑 Batch limit ({FILES_PER_RUN}) reached. Stopping safely.")
                    print(f"   Skipped (Already Done): {skipped_existing}")
                    print(f"   Extracted (This Run):   {extracted_this_run}")
                    print("👉 RUN THIS SCRIPT AGAIN to continue!")
                    sys.exit(0)  # Exit cleanly

            # Cleanup
            if (extract_root / 'lsx').exists():
                shutil.rmtree(extract_root / 'lsx')

    except Exception as e:
        print(f"\n❌ Crash: {e}")
        return

    print("\n✅ ALL FILES EXTRACTED!")
    print(f"   Total processed: {total_processed}")


if __name__ == "__main__":
    # Force garbage collection to free memory
    gc.collect()
    safe_extract_and_organize()