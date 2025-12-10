import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            x = x[:, :self.pe.size(1), :]
        return x + self.pe[:, :seq_len, :]


class NeuralArranger(nn.Module):
   
    def __init__(self, num_followers=2, d_model=256, nhead=8, num_layers=6):
        """
        Transformer for Music Arrangement
        """
        super().__init__()

        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_proj = nn.Linear(d_model, num_followers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lead_loudness):
        x = self.input_proj(lead_loudness)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return self.sigmoid(x)


if __name__ == "__main__":
    print("Testing Pro Arranger...")
    model = NeuralArranger(num_followers=2)

    # Check parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model Size: {params:,} parameters")  

    test_in = torch.rand(1, 1000, 1)
    test_out = model(test_in)
    print("Forward pass successful.")
    