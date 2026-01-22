import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.qvae import HybridQVAE, hybrid_loss_function
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

def train():
    # 1. Load Data
    print("Loading EEG data from emotions.csv...")
    df = pd.read_csv('data/emotions.csv')
    
    # Preprocessing labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    # Split features and labels
    X = df.drop('label', axis=1).values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to Tensors
    tensor_x = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # Small batch for quantum simulation

    # 2. Initialize Model
    input_dim = X.shape[1]
    print(f"Input features: {input_dim}")
    model = HybridQVAE(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 5
    print(f"Starting Training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x_batch = batch[0]
            
            optimizer.zero_grad()
            recon, latent = model(x_batch)
            
            loss = hybrid_loss_function(recon, x_batch, latent)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}")

    # Save the model
    torch.save(model.state_dict(), "qvae_eeg_model.pth")
    print("Training complete. Model saved as qvae_eeg_model.pth")

if __name__ == "__main__":
    train()
