"""
Quick test: Can the Quantum QVAE train with simple MSE loss?
This bypasses the hybrid loss to isolate the problem.
"""
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models.qvae import HybridQVAE
from tqdm import tqdm

print("Loading data...")
df = pd.read_csv('data/emotions.csv')
le = LabelEncoder()
labels = le.fit_transform(df['label'])
features = df.drop('label', axis=1).values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = torch.FloatTensor(features_scaled)
y = torch.LongTensor(labels)

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

input_dim = features.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Input dim: {input_dim}")

# Create model
model = HybridQVAE(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for 3 epochs with SIMPLE MSE LOSS ONLY
EPOCHS = 3
print(f"\nTraining for {EPOCHS} epochs with SIMPLE MSE LOSS...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_x = batch_x.to(device)
        
        optimizer.zero_grad()
        recon, z = model(batch_x)
        
        # SIMPLE MSE LOSS ONLY - no hybrid loss
        loss = torch.nn.functional.mse_loss(recon, batch_x)
        
        if torch.isnan(loss):
            print(f"\n!!! NaN loss at epoch {epoch+1} !!!")
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss = {avg_loss:.4f}")

print("\n✓ Training completed!")
print("\nIf you see valid loss values above (not NaN), then:")
print("  - The quantum model architecture works fine")
print("  - The problem is in the HybridLoss function")
print("  - Solution: Fix the entropy/QWD computation in losses.py")
