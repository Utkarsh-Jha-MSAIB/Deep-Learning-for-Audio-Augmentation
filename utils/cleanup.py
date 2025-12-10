import shutil
from pathlib import Path


def clean_slakh_folder():
    project_root = Path(__file__).resolve().parents[1]
    slakh_dir = project_root / 'data' / 'raw' / 'slakh'

    print(f"Cleaning up duplicates in: {slakh_dir}")

    # 1. Identify BabySlakh (Keep these!)
    baby_tracks = {f"Track{i:05d}" for i in range(1, 21)}

    # 2. Scan directory
    count = 0
    for item in slakh_dir.iterdir():
        if not item.is_dir(): continue

        # Check if it's a "Number Only" track (TrackXXXXX)
        if item.name.startswith("Track") and item.name[5:].isdigit():

            # If it is NOT in the BabySlakh list (1-20), it is a duplicate/garbage
            if item.name not in baby_tracks:
                print(f"   Deleting redundant folder: {item.name}")
                shutil.rmtree(item)
                count += 1

    print(f"Cleanup Complete. Removed {count} duplicate folders.")
    print("   Your 'Track_LSX_...' folders and 'BabySlakh' (1-20) are safe.")


if __name__ == "__main__":
    clean_slakh_folder()
