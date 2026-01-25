import torch
import torch.nn as nn

class ClassicalAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        """
        Minimal Classical Autoencoder Baseline.
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64), 
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = encoded 
        reconstructed = self.decoder(latent)
        return reconstructed, latent
