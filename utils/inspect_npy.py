import numpy as np
from pathlib import Path
import sys
import os


def inspect_processed_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'processed' / 'features'

    print(f"🔍 Inspecting Data in: {data_path}")
    print(f"{'FILENAME':<30} | {'SIZE':<12} | {'SHAPE':<18} | {'RANGE':<20} | {'STATUS'}")
    print("-" * 115)

    files = sorted(list(data_path.glob('*.npy')))
    if not files:
        print("❌ No .npy files found.")
        return

    for f in files:
        try:
            # 1. Get File Size
            size_bytes = f.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"

            # 2. Lazy Load Data
            data = np.load(f, mmap_mode='r')
            shape_str = str(data.shape)

            # 3. Check Content (Fast Sample)
            if len(data) > 0:
                # Check first 1000 items
                sample = data[:1000]
                min_val = np.min(sample)
                max_val = np.max(sample)
                range_str = f"{min_val:.2f} - {max_val:.2f}"

                # Status Logic
                if max_val == 0 and "pitch" in f.name and "Drums" not in f.name:
                    status = "⚠️ SILENT?"
                elif "loudness" in f.name and max_val == 0:
                    status = "⚠️ SILENT?"
                else:
                    status = "✅ OK"
            else:
                range_str = "N/A"
                status = "❌ EMPTY"

            print(f"{f.name:<30} | {size_str:<12} | {shape_str:<18} | {range_str:<20} | {status}")

        except Exception as e:
            print(f"{f.name:<30} | ❌ ERROR: {e}")

    print("-" * 115)


if __name__ == "__main__":
    inspect_processed_data()