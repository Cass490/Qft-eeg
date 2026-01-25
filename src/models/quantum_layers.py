import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumEncoder(nn.Module):
    def __init__(self, n_qubits=4, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Use default.qubit (stable and fast for small circuits)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights_rot, weights_rz):
            # 1. Amplitude Encoding (Time Domain)
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            
            # 2. Quantum Fourier Transform (QFT)
            qml.QFT(wires=range(n_qubits))
            
            # 3. Variational Layers (with both RY and RZ for more expressivity)
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(weights_rot[l, q], wires=q)
                    qml.RZ(weights_rz[l, q], wires=q)  # Add RZ gates!
                
                # Ring Entanglement
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
                    
            return qml.probs(wires=range(n_qubits))

        self.qnode = circuit
        
        # Moderate initialization
        scale = 0.05
        self.weights_rot = nn.Parameter(torch.randn(n_layers, n_qubits) * scale)
        self.weights_rz = nn.Parameter(torch.randn(n_layers, n_qubits) * scale)

    def forward(self, x):
        results = [self.qnode(x[i], self.weights_rot, self.weights_rz) for i in range(x.shape[0])]
        return torch.stack(results)
