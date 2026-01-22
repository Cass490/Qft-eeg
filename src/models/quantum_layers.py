import pennylane as qml
import torch
import torch.nn as nn

class QuantumEncoder(nn.Module):
    def __init__(self, n_qubits=6, n_layers=6):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights_rot):
            # 1. Amplitude Encoding (Time Domain)
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            
            # 2. Quantum Fourier Transform (QFT)
            qml.QFT(wires=range(n_qubits))
            
            # 3. Variational Layers
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(weights_rot[l, q], wires=q)
                
                # Ring Entanglement
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
                    
            return qml.probs(wires=range(n_qubits))

        self.qnode = circuit
        self.weights_rot = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.01)

    def forward(self, x):
        results = [self.qnode(x[i], self.weights_rot) for i in range(x.shape[0])]
        return torch.stack(results)
