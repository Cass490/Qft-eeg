import json
import matplotlib.pyplot as plt
import numpy as np

def plot_paper_results():
    # Load Data
    with open('training_history.json', 'r') as f:
        history = json.load(f)
        
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)

    # --- Plot 1: Reconstruction Loss Comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(history['Quantum_QVAE'], label='Fusion-Aware QVAE (Ours)', linewidth=2, marker='o')
    plt.plot(history['Classical_AE'], label='Classical AE (Baseline)', linewidth=2, linestyle='--')
    plt.title('Reconstruction Loss Comparison', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_comparison.png', dpi=300)
    print("Generated loss_comparison.png")

    # --- Plot 2: Latent Space Classification Accuracy ---
    models = list(results.keys())
    accuracies = [results[m]['Accuracy'] * 100 for m in models]
    
    plt.figure(figsize=(8, 6))
    colors = ['#4CAF50', '#FF5722'] # Green for Quantum, Orange for Classical
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
    
    plt.title('Latent Feature Classification Accuracy', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontsize=12, fontweight='bold')
        
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('accuracy_comparison.png', dpi=300)
    print("Generated accuracy_comparison.png")

if __name__ == "__main__":
    plot_paper_results()
