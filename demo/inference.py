import torch
import numpy as np
import os
import sys

# Add the project root to sys.path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.qvae import HybridQVAE

class DemoInference:
    def __init__(self, model_path, input_dim=2548, latent_dim=64):
        # Force CPU to avoid CUDA library conflicts (sufficient for demo)
        self.device = torch.device('cpu') 
        self.model = HybridQVAE(input_dim=input_dim, latent_dim=latent_dim)
        
        # Load weights
        print(f"Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.input_dim = input_dim

    def run(self, input_tensor):
        """
        Run inference on a single sample or batch
        input_tensor: (Batch, 2548)
        """
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.FloatTensor(input_tensor)
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            reconstructed, latent_z = self.model(input_tensor)
            
        return {
            "reconstructed": reconstructed.cpu().numpy(),
            "latent": latent_z.cpu().numpy(),
            "input": input_tensor.cpu().numpy()
        }

    def predict_emotion(self, latent_z):
        """
        Simple heuristic for demo if a classifier isn't provided
        In a real scenario, this would use the trained Logistic Regression
        """
        # For demo purposes, we'll simulate the 3-class prediction
        # based on the latent vector's mean/std or just random for now
        # until we find the actual classifier weights.
        classes = ["Negative", "Neutral", "Positive"]
        # Dummy logic for demo
        score = np.mean(latent_z)
        if score < 0.4: return classes[0]
        if score < 0.6: return classes[1]
        return classes[2]

if __name__ == "__main__":
    # Test loading
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../qvae_eeg_model.pth'))
    infer = DemoInference(model_path)
    print("Model loaded successfully.")
    
    # Dummy input
    dummy_input = np.random.rand(1, 2548)
    res = infer.run(dummy_input)
    print(f"Inference run complete. Latent shape: {res['latent'].shape}")
    print(f"Predicted Emotion: {infer.predict_emotion(res['latent'])}")
