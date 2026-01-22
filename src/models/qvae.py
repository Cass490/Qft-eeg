import torch
import torch.nn as nn
from .quantum_layers import QuantumEncoder

class HybridQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        
        # 1. Classical Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid() 
        )
        
        # 2. Quantum Encoder Layer (6 qubits)
        self.quantum_layer = QuantumEncoder(n_qubits=6, n_layers=6)
        
        # 3. Classical Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        compressed = self.encoder(x)
        latent_z = self.quantum_layer(compressed)
        reconstructed = self.decoder(latent_z.float())
        return reconstructed, latent_z

def hybrid_loss_function(recon_x, x, latent_z):
    return nn.functional.mse_loss(recon_x, x)
