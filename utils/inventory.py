import numpy as np
from pathlib import Path
import os


def check_inventory():
    # 1. Locate Data
    # We are in tools/, so parent[1] is the project root
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'processed' / 'features'

    print(f"📊 Auditing Instrument Library...")
    print(f"   Location: {data_path}\n")

    if not data_path.exists():
        print("❌ Error: Data folder not found.")
        return

    # 2. Scan Files
    files = list(data_path.glob('loudness_*.npy'))

    if not files:
        print("❌ No processed data found. Did you run 'src/data/preprocess.py'?")
        return

    inventory = []

    for f in files:
        name = f.stem.replace('loudness_', '')

        try:
            data = np.load(f, mmap_mode='r')
            count = data.shape[0]
            duration_min = (count * 4.0) / 60.0

            inventory.append({
                'name': name,
                'chunks': count,
                'minutes': duration_min
            })
        except Exception as e:
            print(f"   ⚠️ Error reading {name}: {e}")

    # 3. Sort & Print
    inventory.sort(key=lambda x: x['chunks'], reverse=True)

    print(f"{'RANK':<5} | {'INSTRUMENT':<15} | {'CHUNKS':<10} | {'DURATION':<10}")
    print("-" * 50)

    for i, item in enumerate(inventory):
        rank = i + 1
        status = "✅" if item['chunks'] > 500 else "⚠️" if item['chunks'] > 100 else "❌"
        print(f"#{rank:<4} | {item['name']:<15} | {item['chunks']:<10} | {item['minutes']:<4.1f} min {status}")

    print("-" * 50)


if __name__ == "__main__":
    check_inventory()