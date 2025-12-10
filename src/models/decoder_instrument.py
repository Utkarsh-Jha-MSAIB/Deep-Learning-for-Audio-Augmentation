import torch
import torch.nn as nn
import numpy as np


class NeuralSynthesizer(nn.Module):
    def __init__(self, hidden_size=512, n_harmonics=100, n_noise_bands=65):
        super().__init__()

        # Input Adapter with GELU
        self.mlp = nn.Sequential(
            nn.Linear(2, 256),
            nn.LayerNorm(256),
            nn.GELU(),  # First tried LeakyReLU
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # Upgraded to Stacked GRU (3 Layers)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=3,  
            batch_first=True,
            dropout=0.1  
        )

        # Post-GRU Normalization
        self.norm_out = nn.LayerNorm(hidden_size)

        # Output Knobs
        self.dense_amp = nn.Linear(hidden_size, 1)
        self.dense_harmonic = nn.Linear(hidden_size, n_harmonics)
        self.dense_noise = nn.Linear(hidden_size, n_noise_bands)

    def hz_to_midi_norm(self, hz):
        """
        Converts Hertz to a normalized 0-1 scale based on MIDI notes
        Formula: MIDI = 69 + 12 * log2(hz / 440)
        """
        # Avoid log(0) error
        hz = torch.clamp(hz, min=1e-5)

        # Convert to MIDI Note Number (e.g., Middle C = 60)
        midi = 69.0 + 12.0 * torch.log2(hz / 440.0)

        # Normalize (0 = Note 0, 1 = Note 127)
        return midi / 127.0

    def forward(self, pitch_hz, loudness):
        """
        Args:
            pitch_hz: (Batch, Time, 1) in Hertz
            loudness: (Batch, Time, 1) in Amplitude (0-1)
        """
        # We convert Hz to (0-1) range
        pitch_norm = self.hz_to_midi_norm(pitch_hz)

        # 1. Embed
        x = torch.cat([pitch_norm, loudness], dim=-1)
        x = self.mlp(x)

        # 2. Memory: GRU
        x, _ = self.gru(x)

        # 3. Stabilize
        x = self.norm_out(x)

        # 4. Control Output
        # Amp: Sigmoid (0-1)
        amp = torch.sigmoid(self.dense_amp(x))

        # Harmonics: Softmax (Distribution)
        harmonics = torch.softmax(self.dense_harmonic(x), dim=-1)

        # Noise: Sigmoid (0-1)
        noise = torch.sigmoid(self.dense_noise(x))

        return {
            'amplitudes': amp,
            'harmonics': harmonics,
            'noise': noise
        }


if __name__ == "__main__":
    print("Testing Pro Decoder...")
    model = NeuralSynthesizer()

    # Test Data (A4 Note = 440Hz)
    test_p = torch.ones(1, 200, 1) * 440.0
    test_l = torch.ones(1, 200, 1)

    # Verification
    norm_val = model.hz_to_midi_norm(test_p)[0, 0, 0]
    print(f"   Checking Math: 440Hz -> Normalized MIDI")
    print(f"   Expected: ~0.54 (Note 69 / 127)")
    print(f"   Actual:   {norm_val:.4f}")
    
    # Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model Size: {total_params:,} parameters")

    controls = model(test_p, test_l)
    print("Forward pass successful.")
