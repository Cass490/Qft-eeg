import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.models.qvae import HybridQVAE
from src.utils.losses import HybridLoss

print("Loading data...")
df = pd.read_csv('data/emotions.csv')
le = LabelEncoder()
labels = le.fit_transform(df['label'])
features = df.drop('label', axis=1).values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = torch.FloatTensor(features_scaled[:32])  # Just 1 batch
input_dim = features.shape[1]

print(f"Input shape: {X.shape}")
print(f"Input contains NaN: {torch.isnan(X).any()}")
print(f"Input range: [{X.min():.4f}, {X.max():.4f}]")

# Create model
print("\nCreating model...")
model = HybridQVAE(input_dim=input_dim)
hybrid_criterion = HybridLoss(lambda1=0.1, lambda2=0.1, lambda3=1.0)

# Test forward pass
print("\nTesting forward pass...")
try:
    recon, z = model(X)
    print(f"✓ Forward pass successful!")
    print(f"  Recon shape: {recon.shape}, contains NaN: {torch.isnan(recon).any()}")
    print(f"  Latent shape: {z.shape}, contains NaN: {torch.isnan(z).any()}")
    print(f"  Latent range: [{z.min():.4f}, {z.max():.4f}]")
    
    # Test loss
    print("\nTesting loss computation...")
    loss = hybrid_criterion(recon, X, z)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss is NaN: {torch.isnan(loss)}")
    
    # Test backward
    print("\nTesting backward pass...")
    loss.backward()
    print(f"✓ Backward pass successful!")
    
    # Check gradients
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"  ✗ NaN gradient in: {name}")
            has_nan_grad = True
    
    if not has_nan_grad:
        print(f"✓ All gradients are valid!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
