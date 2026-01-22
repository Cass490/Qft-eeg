"""
Test the fixed, simplified loss function
"""
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.models.qvae import HybridQVAE
from src.utils.losses import HybridLoss

print("="*60)
print("Testing FIXED Loss Function (Back to What Worked)")
print("="*60)

# Load data
df = pd.read_csv('data/emotions.csv')
le = LabelEncoder()
labels = le.fit_transform(df['label'])
features = df.drop('label', axis=1).values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = torch.FloatTensor(features_scaled[:32])  # 1 batch
input_dim = features.shape[1]

print(f"\nInput shape: {X.shape}")
print(f"Input contains NaN: {torch.isnan(X).any()}")

# Create model and loss
model = HybridQVAE(input_dim=input_dim)
hybrid_criterion = HybridLoss(lambda1=0.5, lambda2=0.3, lambda3=1.0)

print("\n" + "="*60)
print("Test 1: Forward Pass")
print("="*60)
recon, z = model(X)
print(f"✓ Forward pass successful")
print(f"  Recon contains NaN: {torch.isnan(recon).any()}")
print(f"  Latent contains NaN: {torch.isnan(z).any()}")

print("\n" + "="*60)
print("Test 2: Loss Computation")
print("="*60)
loss = hybrid_criterion(recon, X, z)
print(f"  Loss value: {loss.item():.6f}")
print(f"  Loss is NaN: {torch.isnan(loss)}")
print(f"  Loss is Inf: {torch.isinf(loss)}")

if not torch.isnan(loss) and not torch.isinf(loss):
    print("\n✓✓✓ SUCCESS! Loss is valid!")
    
    print("\n" + "="*60)
    print("Test 3: Backward Pass")
    print("="*60)
    loss.backward()
    
    # Check gradients
    nan_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_grads.append(name)
    
    if nan_grads:
        print(f"✗ NaN gradients in: {nan_grads}")
    else:
        print(f"✓✓✓ All gradients are valid!")
        print("\n" + "="*60)
        print("CONCLUSION: Loss function is FIXED!")
        print("="*60)
        print("The model should now train properly.")
        print("Run: python train_comparison.py")
else:
    print("\n✗✗✗ FAILED: Loss is still NaN/Inf")
    print("Need to debug further...")
