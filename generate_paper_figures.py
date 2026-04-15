import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, FancyArrow
import pandas as pd

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def plot_training_curves():
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs_q = range(1, len(history['Quantum_QVAE']) + 1)
    epochs_c = range(1, len(history['Classical_AE']) + 1)
    
    ax.plot(epochs_q, history['Quantum_QVAE'], label='Hybrid QVAE', linewidth=2, alpha=0.8)
    ax.plot(epochs_c, history['Classical_AE'], label='Classical AE', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=16, fontweight='bold')
    # Title removed per user preference
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xticks(fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: training_curves.pdf/png")
    plt.close()

def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=2)
    quantum_style = dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='black', linewidth=2)
    
    ax.text(1, 8, 'Input\n(2558-D)', ha='center', va='center', fontsize=10, bbox=box_style, fontweight='bold')
    
    ax.text(3.5, 8, 'Classical\nEncoder\n512→128→64', ha='center', va='center', fontsize=9, bbox=box_style)
    
    ax.text(7, 8, 'Quantum\nCircuit\n(6 qubits,\n6 layers)', ha='center', va='center', fontsize=9, bbox=quantum_style, fontweight='bold')
    
    ax.text(10.5, 8, 'Classical\nDecoder\n64→128→512', ha='center', va='center', fontsize=9, bbox=box_style)
    
    ax.text(13, 8, 'Output\n(2558-D)', ha='center', va='center', fontsize=10, bbox=box_style, fontweight='bold')
    
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    ax.annotate('', xy=(2.5, 8), xytext=(1.8, 8), arrowprops=arrow_props)
    ax.annotate('', xy=(5.5, 8), xytext=(4.3, 8), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 8), xytext=(8.2, 8), arrowprops=arrow_props)
    ax.annotate('', xy=(12, 8), xytext=(11.3, 8), arrowprops=arrow_props)
    
    ax.text(7, 5.5, 'Quantum Latent Space (64-D)', ha='center', va='center', 
            fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
            edgecolor='red', linewidth=2), fontweight='bold')
    
    ax.annotate('', xy=(7, 7.2), xytext=(7, 6.2), 
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    
    quantum_box = FancyBboxPatch((5.5, 2.5), 3, 2.5, boxstyle="round,pad=0.1", 
                                  edgecolor='darkred', facecolor='mistyrose', linewidth=2)
    ax.add_patch(quantum_box)
    
    ax.text(7, 4.3, 'Amplitude Encoding', ha='center', fontsize=8)
    ax.text(7, 3.9, 'QFT Layer', ha='center', fontsize=8)
    ax.text(7, 3.5, 'RY/RZ Rotations', ha='center', fontsize=8)
    ax.text(7, 3.1, 'Ring Entanglement (CNOT)', ha='center', fontsize=8)
    ax.text(7, 2.7, 'Pauli-Z Measurements', ha='center', fontsize=8)
    
    ax.text(7, 1.5, 'Hybrid Quantum-Classical Variational Autoencoder', 
            ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/architecture_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Saved: architecture_diagram.pdf/png")
    plt.close()

def plot_quantum_circuit_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    n_qubits = 6
    
    for i in range(n_qubits):
        y = 6 - i
        ax.plot([0.5, 11.5], [y, y], 'k-', linewidth=1.5)
        ax.text(0.2, y, f'$|q_{i}\\rangle$', ha='right', va='center', fontsize=11)
    
    x_pos = 1.5
    ax.text(x_pos, 7.5, 'Amplitude\nEncoding', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightblue'))
    for i in range(n_qubits):
        y = 6 - i
        rect = Rectangle((x_pos-0.2, y-0.2), 0.4, 0.4, facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
    
    x_pos = 3
    ax.text(x_pos, 7.5, 'QFT', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    for i in range(n_qubits):
        y = 6 - i
        rect = Rectangle((x_pos-0.3, y-0.25), 0.6, 0.5, facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x_pos, y, 'QFT', ha='center', va='center', fontsize=7)
    
    x_pos = 5
    ax.text(x_pos, 7.5, 'Layer 1-6\n(Repeated)', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    for i in range(n_qubits):
        y = 6 - i
        circle = Circle((x_pos, y), 0.15, facecolor='yellow', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x_pos, y, '$R_Y$', ha='center', va='center', fontsize=6)
    
    x_pos = 6
    for i in range(n_qubits):
        y = 6 - i
        circle = Circle((x_pos, y), 0.15, facecolor='orange', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x_pos, y, '$R_Z$', ha='center', va='center', fontsize=6)
    
    x_pos = 7.5
    ax.text(x_pos, 7.5, 'Ring\nEntanglement', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    for i in range(n_qubits):
        y1 = 6 - i
        y2 = 6 - ((i + 1) % n_qubits)
        
        circle = Circle((x_pos, y1), 0.1, facecolor='black')
        ax.add_patch(circle)
        
        if i < n_qubits - 1:
            ax.plot([x_pos, x_pos], [y1, y2], 'k-', linewidth=2)
            circle2 = Circle((x_pos, y2), 0.15, facecolor='white', edgecolor='black', linewidth=2)
            ax.add_patch(circle2)
            ax.plot([x_pos-0.1, x_pos+0.1], [y2, y2], 'k-', linewidth=1.5)
            ax.plot([x_pos, x_pos], [y2-0.1, y2+0.1], 'k-', linewidth=1.5)
        else:
            ax.plot([x_pos, x_pos, x_pos+0.5, x_pos+0.5], [y1, 0.3, 0.3, y2], 
                   'k--', linewidth=1.5, alpha=0.5)
            circle2 = Circle((x_pos+0.5, y2), 0.15, facecolor='white', edgecolor='black', linewidth=2)
            ax.add_patch(circle2)
            ax.plot([x_pos+0.4, x_pos+0.6], [y2, y2], 'k-', linewidth=1.5)
            ax.plot([x_pos+0.5, x_pos+0.5], [y2-0.1, y2+0.1], 'k-', linewidth=1.5)
    
    x_pos = 10
    ax.text(x_pos, 7.5, 'Measurement\n(Pauli-Z)', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lavender'))
    for i in range(n_qubits):
        y = 6 - i
        rect = Rectangle((x_pos-0.25, y-0.25), 0.5, 0.5, facecolor='lavender', 
                        edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos, y, 'Z', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.text(6, 0.3, '6-Qubit Parameterized Quantum Circuit', 
            ha='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/quantum_circuit.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/quantum_circuit.png', dpi=300, bbox_inches='tight')
    print("Saved: quantum_circuit.pdf/png")
    plt.close()

def plot_convergence_comparison():
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_q = range(1, len(history['Quantum_QVAE']) + 1)
    epochs_c = range(1, len(history['Classical_AE']) + 1)
    
    # Subplot 1: Hybrid QVAE
    ax1.plot(epochs_q, history['Quantum_QVAE'], linewidth=2.5, color='#D32F2F', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=history['Quantum_QVAE'][-1], color='r', linestyle='--', 
                label=f'Final: {history["Quantum_QVAE"][-1]:.4f}', alpha=0.6)
    ax1.legend(fontsize=12)
    
    # Subplot 2: Classical AE
    ax2.plot(epochs_c, history['Classical_AE'], linewidth=2.5, color='#1976D2', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Loss (MSE)', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=history['Classical_AE'][-1], color='b', linestyle='--', 
                label=f'Final: {history["Classical_AE"][-1]:.4f}', alpha=0.6)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/convergence_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: convergence_comparison.pdf/png")
    plt.close()

def plot_performance_comparison():
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Rename labels for multi-line display
    display_models = ['Hybrid\nQVAE', 'Classical\nAE']
    accuracies = [results['Quantum_QVAE']['Accuracy'] * 100, results['Classical_AE']['Accuracy'] * 100]
    
    latent_dims = [64, 32]
    
    # Create a compact standalone figure for accuracy ONLY
    fig, ax1 = plt.subplots(figsize=(4, 5.5)) # Narrower figure for compact look
    
    colors = ['#D32F2F', '#1976D2'] # Red for Quantum, Blue for Classical
    # Sleek bars (width=0.4)
    bars1 = ax1.bar(display_models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.4)
    
    # Enhanced typography for small figure
    ax1.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold', labelpad=12)
    # Title removed per user preference
    
    # Large, bold tick labels
    ax1.tick_params(axis='both', which='major', labelsize=15)
    plt.xticks(fontweight='bold')
    
    # Tight zoomed scale
    ax1.set_ylim(84.0, 86.5) 
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Large data labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=15, fontweight='bold', color='black')

    # (Quantum Advantage annotation removed per user preference)
    
    plt.tight_layout()
    plt.savefig('figures/performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: performance_comparison.pdf/png")
    plt.close()

def create_results_table():
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    data = {
        'Model': ['Hybrid QVAE', 'Classical AE'],
        'Latent Dim': [64, 32],
        'Accuracy (%)': [
            results['Quantum_QVAE']['Accuracy'] * 100,
            results['Classical_AE']['Accuracy'] * 100
        ],
        'Final Loss': [
            history['Quantum_QVAE'][-1],
            history['Classical_AE'][-1]
        ],
        'Epochs': [150, 150]
    }
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*60)
    print("RESULTS TABLE FOR PAPER")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")
    
    latex_table = df.to_latex(index=False, float_format="%.4f")
    
    with open('figures/results_table.tex', 'w') as f:
        f.write(latex_table)
    print("Saved: results_table.tex")
    
    return df

if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("Generating figures for research paper...")
    print("-" * 50)
    
    plot_training_curves()
    plot_architecture_diagram()
    plot_quantum_circuit_diagram()
    plot_convergence_comparison()
    plot_performance_comparison()
    create_results_table()
    
    print("-" * 50)
    print("All figures generated successfully!")
    print("Check the 'figures/' directory for outputs")
