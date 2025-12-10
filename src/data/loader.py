import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class SynthDataset(Dataset):
    def __init__(self, feature_dir, instrument):
        """
        Args:
            feature_dir (str or Path): Path to the processed folder
            instrument (str): REQUIRED. The instrument name (e.g., 'Bass', 'Strings')
                              Must match the suffix of your .npy files
        """
        self.feature_dir = Path(feature_dir)
        self.instrument = instrument

        # 1. Construct Filenames
        p_path = self.feature_dir / f'pitch_{instrument}.npy'
        l_path = self.feature_dir / f'loudness_{instrument}.npy'
        a_path = self.feature_dir / f'audio_{instrument}.npy'

        # 2. Check if they exist
        if not p_path.exists():
            raise FileNotFoundError(
                f"Error: Could not find data for '{instrument}' at {p_path}\n"
                f"   Did you run 'src/data/preprocess.py' with TARGET_INSTRUMENT='{instrument}'?"
            )

        # 3. Load Data
        print(f"Loading dataset for: {instrument}...")
        self.pitch = np.load(p_path).astype(np.float32)
        self.loudness = np.load(l_path).astype(np.float32)
        self.audio = np.load(a_path).astype(np.float32)

        # 4. Sanity Check (Lengths must match)
        if not (len(self.pitch) == len(self.loudness) == len(self.audio)):
            raise ValueError("Shape Mismatch: Pitch, Loudness, and Audio have different lengths!")

    def __len__(self):
        return len(self.pitch)

    def __getitem__(self, idx):
        # 1. Get Raw Arrays
        p = self.pitch[idx]  # (1000,)
        l = self.loudness[idx]  # (1000,)
        a = self.audio[idx]  # (64000,)

        # 2. Convert to Tensors & Add Dimensions, Inputs need (Time, 1)
        p_tensor = torch.from_numpy(p).unsqueeze(-1)
        l_tensor = torch.from_numpy(l).unsqueeze(-1)

        # Target Audio also needs (Time, 1)
        a_tensor = torch.from_numpy(a).unsqueeze(-1)

        return {
            'pitch': p_tensor,
            'loudness': l_tensor,
            'audio': a_tensor
        }


# --- TEST BLOCK ---
if __name__ == "__main__":

    # Configuration for test
    TEST_INSTRUMENT = 'Bass'  # Change this to test other files

    script_location = Path(__file__).resolve()
    project_root = script_location.parents[2]
    data_path = project_root / 'data' / 'processed' / 'features'

    print(f"Testing Loader in: {data_path}")

    if data_path.exists():
        try:
            # We must explicitly pass the instrument now
            ds = SynthDataset(data_path, instrument=TEST_INSTRUMENT)

            print(f"Loader Works! Found {len(ds)} examples")
            sample = ds[0]
            print(f"   Pitch Shape:    {sample['pitch'].shape}")
            print(f"   Loudness Shape: {sample['loudness'].shape}")
            print(f"   Audio Shape:    {sample['audio'].shape}")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Path not found: {data_path}")