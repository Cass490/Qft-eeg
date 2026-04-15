import torch
import torch.nn as nn
from src.models.qvae import HybridQVAE
from src.models.classical_ae import ClassicalAE

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def detailed_count(name, model):
    print(f"\n===== {name} Parameter Breakdown =====")
    total = 0
    # For HybridQVAE
    if hasattr(model, 'encoder') and hasattr(model, 'quantum_layer'):
        enc = count_parameters(model.encoder)
        qnt = count_parameters(model.quantum_layer)
        dec = count_parameters(model.decoder)
        print(f"Classical Encoder: {enc:,}")
        print(f"Quantum Bottleneck: {qnt:,}")
        print(f"Classical Decoder: {dec:,}")
        total = enc + qnt + dec
    # For ClassicalAE
    else:
        enc = count_parameters(model.encoder)
        dec = count_parameters(model.decoder)
        print(f"Classical Encoder: {enc:,}")
        print(f"Classical Decoder: {dec:,}")
        total = enc + dec
    print(f"TOTAL Trainable Params: {total:,}")
    return total

if __name__ == "__main__":
    input_dim = 2558
    qvae = HybridQVAE(input_dim=input_dim)
    cae = ClassicalAE(input_dim=input_dim)
    
    q_total = detailed_count("Hybrid QVAE", qvae)
    c_total = detailed_count("Classical AE", cae)
    
    print("\n" + "="*40)
    # The true "Bottleneck Comparison":
    # QVAE core evolution is just the 72 quantum parameters.
    # Classical AE's final encoder layer is Linear(256, 64) -> 16,448 params.
    q_bottleneck = count_parameters(qvae.quantum_layer)
    c_bottleneck = sum(p.numel() for p in cae.encoder[-2].parameters() if p.requires_grad) # Linear(256,64)
    
    print(f"Quantum Bottleneck Params: {q_bottleneck}")
    print(f"Equivalent Classical Layer Params: {c_bottleneck}")
    print(f"Efficiency Factor: {c_bottleneck/q_bottleneck:.1f}x less parameters")
    print("="*40)
