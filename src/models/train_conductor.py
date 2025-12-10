import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pathlib import Path
import os

from src.models.decoder_conductor import NeuralArranger

# Parameters (Currently valid for LSX dataset with 3 instruments)
LEADER = 'Guitar'
FOLLOWERS = ['Bass', 'Drums']

BATCH_SIZE = 256
LR = 1e-4
MAX_EPOCHS = 100


class ArrangementDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_dir = Path(feature_dir)

        print(f"Loading Leader: {LEADER}...")
        lead_path = self.feature_dir / f'loudness_{LEADER}.npy'
        if not lead_path.exists():
            raise FileNotFoundError(f"Missing data in {self.feature_dir}. Did you run preprocess_band.py?")
        self.lead = np.load(lead_path).astype(np.float32)

        target_list = []
        min_len = len(self.lead)

        for inst in FOLLOWERS:
            path = self.feature_dir / f'loudness_{inst}.npy'
            if not path.exists():
                raise FileNotFoundError(f"Missing data for {inst}")

            data = np.load(path).astype(np.float32)
            print(f"   Loading {inst}: {len(data)} chunks")
            if len(data) < min_len: min_len = len(data)
            target_list.append(data)

        # If preprocess_band worked, min_len should equal len(lead)
        # But we keep this safety check just in case.
        if min_len < len(self.lead):
            print(f"Trimming to common length: {min_len}")

        self.lead = self.lead[:min_len]
        cropped_targets = [t[:min_len] for t in target_list]
        self.targets = np.stack(cropped_targets, axis=-1)

        print(f"Dataset Ready. Shape: {self.targets.shape}")

    def __len__(self):
        return len(self.lead)

    def __getitem__(self, idx):
        return torch.from_numpy(self.lead[idx]).unsqueeze(-1), torch.from_numpy(self.targets[idx])


class ArrangerTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = NeuralArranger(num_followers=len(FOLLOWERS))
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()  #We tried this also

    def forward(self, lead_loudness):
        return self.model(lead_loudness)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        prediction = self.model(src)
        loss = self.loss_fn(prediction, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=LR)

    def train_dataloader(self):
        current_file = Path(__file__).resolve()
        if 'src' in current_file.parts:
            src_index = current_file.parts.index('src')
            project_root = Path(*current_file.parts[:src_index])
        else:
            project_root = current_file.parents[2]

        data_path = project_root / 'data' / 'processed' / 'aligned_band'
        return DataLoader(ArrangementDataset(data_path), batch_size=BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    print(f"Training Neural Conductor...")

    filename_str = f"Arranger-{LEADER}-{{epoch:02d}}-{{train_loss:.4f}}"
    checkpoint_callback = ModelCheckpoint(
        filename=filename_str,
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    system = ArrangerTrainer()
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(system)
