"""
Generate quantum circuit diagram using Qiskit
This creates a visual representation of the 6-qubit parameterized quantum circuit
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import QFTGate
import matplotlib.pyplot as plt
import os
import numpy as np

def create_quantum_circuit_diagram():
    """
    Create a 6-qubit quantum circuit matching the PennyLane implementation
    """
    n_qubits = 6
    n_layers = 6
    
    qc = QuantumCircuit(n_qubits)
    
    # 1. Amplitude Encoding (represented as initialization)
    qc.barrier(label='Amplitude\nEncoding')
    
    # 2. Quantum Fourier Transform
    qc.barrier()
    qft_gate = QFTGate(n_qubits)
    qc.append(qft_gate, range(n_qubits))
    qc.barrier(label='QFT')
    
    # 3. Variational Layers (show first 2 layers for clarity)
    for layer in range(min(2, n_layers)):  # Show only 2 layers to keep diagram readable
        # RY and RZ rotations on each qubit
        for q in range(n_qubits):
            theta = Parameter(f'θ{layer}{q}')
            phi = Parameter(f'φ{layer}{q}')
            qc.ry(theta, q)
            qc.rz(phi, q)
        
        qc.barrier(label=f'Layer {layer+1}\nRotations')
        
        # Ring entanglement (CNOT in cyclic pattern)
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)
        
        qc.barrier(label='Ring\nEntangle')
    
    # Add ellipsis to indicate more layers
    qc.barrier(label='...\n(4 more\nlayers)')
    
    # 4. Measurement (Pauli-Z expectation)
    qc.barrier(label='Measure\nPauli-Z')
    
    return qc


def save_circuit_diagram():
    """
    Generate and save the quantum circuit diagram
    """
    os.makedirs('figures', exist_ok=True)
    
    qc = create_quantum_circuit_diagram()
    
    # Create figure with matplotlib circuit drawer
    fig = qc.draw(output='mpl', 
                  style={'backgroundcolor': '#FFFFFF',
                         'subfontsize': 9,
                         'fontsize': 11},
                  fold=False,
                  scale=1.0)
    
    # Save as PDF and PNG
    plt.savefig('figures/quantum_circuit_qiskit.pdf', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('figures/quantum_circuit_qiskit.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("Saved: quantum_circuit_qiskit.pdf/png")
    plt.close()


if __name__ == "__main__":
    print("Generating Qiskit quantum circuit diagram...")
    save_circuit_diagram()
    print("Done!")
