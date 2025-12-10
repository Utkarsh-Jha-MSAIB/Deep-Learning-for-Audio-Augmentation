import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torchaudio
import os

from src.data.loader import SynthDataset
from src.models.decoder_instrument import NeuralSynthesizer
from src.models.signal_processing import harmonic_synthesis, noise_synthesis

# Paramters for model
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20

INSTRUMENT = 'Pipe'  #Enter a single instrument you want to train

class AudioSynthTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = NeuralSynthesizer()

        # Loss Function: L1 Loss (Mean Absolute Error) on Spectrograms
        self.loss_fn = nn.L1Loss()

    def forward(self, pitch, loudness):
        """
        Generates audio from pitch and loudness.
        This connects the 'Brain' to the 'Signal Processing'.
        """
        # Pitch/Loudness
        controls = self.model(pitch, loudness)

        # 2. Calculate Dynamic Length 
        # Input frames (250Hz) * 64 = Output samples (16000Hz)
        # Example: 1000 frames * 64 = 64000 samples (4s)
        # Example: 5000 frames * 64 = 320000 samples (20s)
        target_samples = pitch.shape[1] * 64

        # 3. Signal Processing: Controls: Audio
        harmonics = harmonic_synthesis(
            pitch,
            controls['amplitudes'],
            controls['harmonics'],
            n_samples=target_samples 
        )
        noise = noise_synthesis(
            controls['noise'],
            n_samples=target_samples 
        )

        # 4. Mix
        return harmonics + noise

    def training_step(self, batch, batch_idx):
        # A. Unpack Data
        pitch = batch['pitch']
        loudness = batch['loudness']
        target_audio = batch['audio']

        # B. Forward Pass (Generate Model Audio)
        generated_audio = self(pitch, loudness)

        # C. Calculate Loss
        real_spec = self.get_spectrogram(target_audio.squeeze(-1))
        model_spec = self.get_spectrogram(generated_audio.squeeze(-1))

        loss = self.loss_fn(model_spec, real_spec)

        # D. Log to Progress Bar
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def get_spectrogram(self, audio):
        """Turns audio into a Mel Spectrogram (Image)."""
        # We move the transform to the same device (CPU/GPU) as the audio
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, n_mels=80
        ).to(audio.device)
        return transform(audio)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train_dataloader(self):
        from pathlib import Path

        # 1. Find the file location
        current_file = Path(__file__).resolve()

        # 2. Walk up until we find the 'src' folder, then go one up to Root
        if 'src' in current_file.parts:
            src_index = current_file.parts.index('src')
            project_root = Path(*current_file.parts[:src_index])
        else:
            project_root = current_file.parents[2]

        # 3. Construct the correct path
        data_path = project_root / 'data' / 'processed' / 'features'

        print(f"Loading Data from: {data_path}")

        # Initialize
        dataset = SynthDataset(data_path, instrument=INSTRUMENT)

        num_workers = min(os.cpu_count() or 0, 4)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)



if __name__ == "__main__":
    print(f"Starting Training for {INSTRUMENT}...")

    # This forces the file to be named e.g. "Guitar-epoch=19-loss=0.05.ckpt"
    checkpoint_callback = ModelCheckpoint(
        filename=f"{INSTRUMENT}-{{epoch:02d}}-{{train_loss:.4f}}",
        save_top_k=1, 
        monitor="train_loss",  
        mode="min"  
    )

    system = AudioSynthTrainer()

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  
        devices=1,
        enable_checkpointing=True,
        callbacks = [checkpoint_callback] 
    )

    trainer.fit(system)