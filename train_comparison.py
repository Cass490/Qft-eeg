import pandas as pd
import torch
import numpy as np
import json
import random
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models.qvae import HybridQVAE
from src.models.classical_ae import ClassicalAE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def set_seed(seed=45):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PennyLane seed for quantum circuit randomness
    import pennylane as qml
    qml.numpy.random.seed(seed)

def train_and_evaluate():
    # Set seed for reproducibility
    set_seed(45)
    # --- 1. Data Prep ---
    print("Loading Multimodal EEG+ECG Data...")
    
    # Check if multimodal data exists, if not create it
    import os
    multimodal_path = 'data/multimodal_fused.csv'
    
    if not os.path.exists(multimodal_path):
        print("Creating multimodal EEG+ECG dataset...")
        from load_multimodal_data import save_multimodal_data
        save_multimodal_data(multimodal_path)
    
    df = pd.read_csv(multimodal_path)
    print(f"Loaded multimodal data: {df.shape}")
    
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
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    input_dim = features.shape[1]
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # --- 2. Setup Models ---
    models = {
        'Quantum_QVAE': HybridQVAE(input_dim=input_dim).to(device),
        'Classical_AE': ClassicalAE(input_dim=input_dim).to(device)
    }
    
    # Optimized learning rates
    optimizers = {
        'Quantum_QVAE': torch.optim.Adam(models['Quantum_QVAE'].parameters(), lr=0.0004),
        'Classical_AE': torch.optim.Adam(models['Classical_AE'].parameters(), lr=0.003)
    }
    
    history = {'Quantum_QVAE': [], 'Classical_AE': []}
    
    # --- 3. Training Loop ---
    EPOCHS = 150 # Full training: RZ gates, 6 layers, optimized LRs 
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.train()
        
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch_x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
                batch_x = batch_x.to(device)
                optimizers[name].zero_grad()
                recon, _ = model(batch_x)
                loss = torch.nn.functional.mse_loss(recon, batch_x)
                loss.backward()
                optimizers[name].step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history[name].append(avg_loss)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
    # --- 4. Latent Space Evaluation ---
    print("\n--- Evaluating Latent Representations ---")
    results = {}
    
    for name, model in models.items():
        model.eval()
        train_latents, train_labels_list = [], []
        test_latents, test_labels_list = [], []
        
        with torch.no_grad():
            for bx, by in train_loader:
                bx = bx.to(device)
                _, z = model(bx)
                train_latents.append(z.cpu().numpy())
                train_labels_list.append(by.numpy())
            for bx, by in test_loader:
                bx = bx.to(device)
                _, z = model(bx)
                test_latents.append(z.cpu().numpy())
                test_labels_list.append(by.numpy())
        
        X_train_latent = np.nan_to_num(np.vstack(train_latents))
        y_train = np.concatenate(train_labels_list)
        X_test_latent = np.nan_to_num(np.vstack(test_latents))
        y_test = np.concatenate(test_labels_list)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_latent, y_train)
        preds = clf.predict(X_test_latent)
        
        results[name] = {'Accuracy': accuracy_score(y_test, preds)}
        print(f"{name} Latent Accuracy: {results[name]['Accuracy']*100:.2f}%")

    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    train_and_evaluate()
