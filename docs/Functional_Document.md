# Functional Document (Module Description)

## 1. Introduction
This document describes the functional modules of the Fusion-Aware Quantum Variational Autoencoder (QVAE) project. It provides a detailed breakdown of the internal software modules and their operational purpose.

## 2. Module Descriptions

### 2.1 Data Loading and Preprocessing Module
**File(s):** `load_multimodal_data.py`, `src/preprocessing/eeg_loader.py`, `src/preprocessing/ecg_loader.py`
- **Functionality:** 
  - Loads EEG features from `emotions.csv` and ECG waveforms from the `MIT-BIH` database.
  - Extracts statistical features (mean, std, min, max, median, percentiles, variance, range, etc.) from ECG segments.
  - Fuses the modalities using feature concatenation (EEG + ECG).
  - Handles normalization using standard scalers.
  - Pads or truncates ECG data to match the length of the EEG dataset.

### 2.2 Quantum-Classical Hybrid Model Module
**File(s):** `src/models/qvae.py`
- **Functionality:** 
  - Implements the `HybridQVAE` class using PyTorch.
  - Controls the forward pass logic linking the classical encoder, the simulated quantum layers (parameterized circuits), and the classical decoder.
  - Calculates the hybrid loss function incorporating reconstruction loss for both EEG and ECG channels alongside the variational regularization terms.

### 2.3 Training and Execution Module
**File(s):** `main.py`, `train_comparison.py`, `verify_params_advantage.py`
- **Functionality:** 
  - Initializes model hyperparameters (batch size, epochs, learning rates).
  - Drives the end-to-end training loop for the autoencoder.
  - Handles the batching of input tensors, gradients zeroing, loss backpropagation, and optimizer steps.
  - Simulates training progress and generates comparative performance metrics between the Hybrid QVAE and classical counterparts.

### 2.4 Artifact and Visual Generation Module
**File(s):** `generate_paper_figures.py`, `generate_qiskit_circuit.py`, `plot_results.py`, `diag.py`, `diag_v2.py`
- **Functionality:** 
  - Generates visual artifacts like the system architecture diagrams (Mermaid format), training loss curves, and evaluation plots.
  - Extracts circuit architectures into visual formats for academic paper publication.
  - Produces premium-quality architecture diagrams (`arch_premium.png`) highlighting the input-process-output flow.

## 3. Core Project Workflow

1. **Wait for Input**: System receives synchronized EEG (14 Channels) and ECG (1 Channel) time-series data.
2. **Preprocessing**: 
   - EEG undergoes artifact removal (ICA) and bandpass filtering. 
   - ECG is processed via Pan-Tompkins for QRS detection and statistical feature extraction (10-D).
3. **Fusion**: Features are concatenated into a 2558-D vector and normalized.
4. **Encoding**: Classical Dense layers compress the vector to a 64-D latent feature set.
5. **Quantum Entanglement**: The 64-D features are mapped to a 6-qubit Hilbert space, processed via QFT and 6 layers of RY/RZ rotations with ring entanglement.
6. **Decoding / Prediction**:
   - The processed latent state is decoded back to the original 2558-D domain to calculate reconstruction loss.
   - Simultaneously, a Logistic Regression classifier predicts the emotional state (Positive, Neutral, Negative).
