import torch
import torch.nn.functional as F
import numpy as np


def upsample(signal, n_samples):
    """
    Stretches a low-res signal (Batch, Time, Channels) to high-res (Batch, n_samples, Channels)
    """
    # Permute to (Batch, Channels, Time) for interpolate
    signal = signal.permute(0, 2, 1)
    # Linear Interpolation (Smoothly fills the gaps)
    signal = F.interpolate(signal, size=n_samples, mode='linear', align_corners=True)
    # Permute back to (Batch, Time, Channels)
    return signal.permute(0, 2, 1)


def harmonic_synthesis(pitch, amplitudes, harmonic_distribution, n_samples=64000, sample_rate=16000):
    """
    Synthesizes audio using Additive Synthesis (Stacking Sine Waves)
    """
    n_harmonics = harmonic_distribution.shape[-1]

    # 1. UPSAMPLE CONTROLS
    pitch = upsample(pitch, n_samples)
    amplitudes = upsample(amplitudes, n_samples)
    harmonic_distribution = upsample(harmonic_distribution, n_samples)

    # 2. Create Harmonic Frequencies
    harmonic_ratios = torch.linspace(1, n_harmonics, n_harmonics, device=pitch.device)
    harmonic_frequencies = pitch * harmonic_ratios

    # 3. Oscillator (Phase Accumulation)
    phase = torch.cumsum(harmonic_frequencies / sample_rate, dim=1)
    phase = 2 * np.pi * phase

    # 4. Generate Sine Waves
    sine_waves = torch.sin(phase)

    # 5. Apply Harmonic Volumes
    harmonic_sum = torch.sum(harmonic_distribution, dim=-1, keepdim=True)
    harmonic_distribution = harmonic_distribution / (harmonic_sum + 1e-7)

    signal = sine_waves * harmonic_distribution

    # 6. Mix Down
    harmonic_signal = torch.sum(signal, dim=-1, keepdim=True)

    return harmonic_signal * amplitudes


def noise_synthesis(noise_magnitudes, n_samples=64000):
    """
    Synthesizes filtered noise
    """
    batch_size = noise_magnitudes.shape[0]

    # 1. Upsample Noise Controls
    noise_envelope = upsample(noise_magnitudes, n_samples)

    # 2. Generate White Noise
    white_noise = torch.rand(batch_size, n_samples, 1, device=noise_magnitudes.device) * 2 - 1

    # 3. Apply Envelope
    global_noise_amp = torch.mean(noise_envelope, dim=-1, keepdim=True)

    return white_noise * global_noise_amp
