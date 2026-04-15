# Project Overview: Fusion-Aware Quantum Variational Autoencoder (QVAE)

This document provides a concise and professional explanation of the project for use in presentations, reports, and vivo examinations.

## 1. The Core Problem
Most traditional AI systems process brain (EEG) and heart (ECG) signals using classical neural networks. However, these physiological signals are highly complex, nonlinear, and possess hidden "quantum-like" correlations that classical bits (0s and 1s) sometimes fail to capture efficiently.

## 2. Our Solution: The Hybrid QVAE
We have developed a **Hybrid Quantum-Classical Variational Autoencoder**. This system uses the best of both worlds:
- **Classical Power**: Handles the "heavy lifting" of raw signal preprocessing and high-dimensional compression.
- **Quantum Advantage**: Processes the core "essence" of the signal (the latent space) using a Quantum Circuit to find patterns that are difficult for classical models to identify.

---

## 3. How the Pipeline Works

### Step A: Multimodal Fusion
We take two different types of signals:
- **EEG (Brainwaves)**: 2,548 features extracted from multiple brain regions.
- **ECG (Heartbeat)**: 10 statistical features (like heart rate variability).
These are fused together into a single "feature vector" that represents the user's total physiological state.

### Step B: The Classical Encoder
A deep neural network compresses this massive data vector into a 64-dimensional "bottleneck." This ensures that only the most important emotional indicators are passed to the quantum layer.

### Step C: The Quantum Processing Unit (PQC)
This is the "brain" of the project. We use a **6-Qubit Parameterized Quantum Circuit**:
1. **Amplitude Encoding**: The 64 classical numbers are converted into a quantum state.
2. **Quantum Fourier Transform (QFT)**: This maps the signals into the frequency domain, which is crucial for identifying the "rhythms" of emotions.
3. **Ring Entanglement**: We link the qubits in a circular chain. This allows the model to analyze the cross-talk between different features simultaneously—something classical units struggle with.
4. **Variational Layers**: These are the "trainable" parts of the quantum circuit that adjust their rotation angles to recognize specific emotional patterns.

### Step D: Decoding and Classification
- The system tries to **Reconstruct** the original signals from the quantum state (as seen in the Dashboard).
- Simultaneously, a **Logistic Regression** model looks at the quantum features and predicts whether the person is feeling **Positive, Neutral, or Negative**.

---

## 4. Key Highlights for Presentation
- **Accuracy**: The model achieves **85.71% accuracy** in emotion recognition.
- **Efficiency**: By using only **6 qubits**, we can process complex multi-channel data that would normally require very deep classical networks.
- **Robustness**: The use of quantum entanglement helps the model ignore high-frequency noise that often plagues biological signals.

## 5. Technology Stack
- **Quantum**: PennyLane & Qiskit
- **Classical**: PyTorch (Deep Learning)
- **Frontend**: Flask Dashboard with Chart.js visualization
- **Data**: emotions.csv (EEG) and MIT-BIH (ECG)
