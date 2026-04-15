# Architecture Document: Fusion-Aware Quantum Variational Autoencoder (QVAE)

## 1. Introduction
This document outlines the architectural design of the Fusion-Aware Quantum Variational Autoencoder (QVAE) system for multimodal brain-heart signal processing, specifically designed to process Electroencephalogram (EEG) and Electrocardiogram (ECG) data for emotion classification.

## 2. System Overview
The architecture is designed as a hybrid quantum-classical pipeline. It accepts raw multimodal signals, fuses them at the feature level, compressing the representation using a classical encoder, processes the latent space representation using a parameterised quantum circuit (PQC), and decodes it classically. The latent representations are then evaluated using a classifier (Logistic Regression) for a 3-class emotion prediction task.

## 3. High-Level Architecture Components
The system consists of the following primary sub-systems:

### 3.1 Input & Preprocessing
- **EEG Features**: 2548-D features extracted from `emotions.csv` (14 channels). 
- **ECG Features**: 10-D statistical features extracted from the MIT-BIH Database using Pan-Tompkins processing logic.
- **Feature Fusion**: Concatenation of processed EEG and ECG features resulting in a 2558-D fused feature vector.

### 3.2 Classical Encoder
- Reduces dimensionality from the 2558-D fused input to a 64-D compressed representation.
- **Layers**: 
  - Dense (2558 → 512, ReLU)
  - Dense (512 → 128, ReLU)
  - Dense (128 → 64, Sigmoid)

### 3.3 Quantum Processing (QVAE Latent Layer)
- Maps the 64-D input to a quantum state using Amplitude Encoding (6 qubits).
- **QFT Layer**: Applies a Quantum Fourier Transform for domain mapping.
- **Variational Layers**: 6 identical layers of parameterized circuits:
  - **Rotations**: RY(θ) and RZ(φ) on each of the 6 qubits.
  - **Entanglement**: Ring (cyclic) CNOT entanglement pattern between adjacent qubits.
- **Measurement**: Pauli-Z Expectation values are measured to produce a 64-D quantum-processed latent state.

### 3.4 Classical Decoder
- Reconstructs the 2558-D original signal from the 64-D quantum latent state.
- **Layers**:
  - Dense (64 → 128, ReLU)
  - Dense (128 → 512, ReLU)
  - Dense (512 → 2558, Linear)

### 3.5 Emotion Classifier
- Accepts the 64-D latent vector as input.
- A Logistic Regression model provides predictions across 3 emotional state classes.

## 4. Performance Metrics
Based on the high-level visual architecture and initial simulation results:
- **Accuracy**: 85.71% (3-class emotion classification)
- **MSE Loss**: 0.2911 (Overall reconstruction stability)
- **Stability ($\sigma$)**: 0.0164 (Variance across test batches)
- **Latent Dimension**: 64-D (Optimal for 6-qubit encoding)

## 5. Technology Stack
- **Frameworks**: PyTorch (Classical Deep Learning), Qiskit/Pennylane context (Quantum Simulation).
- **Data Handling**: Pandas, NumPy, WFDB (for MIT-BIH).
