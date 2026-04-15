# Project Diagram Guide: Fusion-Aware QVAE

To effectively document this hybrid quantum-classical project, you should use a combination of architectural and UML (Unified Modeling Language) diagrams.

## 1. Core Architectural Diagrams

### 1.1 High-Level System Architecture
- **Purpose**: Show the end-to-end flow from raw signals to emotion prediction.
- **Type**: Flowchart / Block Diagram.
- **Current Status**: Already exists as `architecture_diagram.mmd` and `figures/arch_premium.png`.

### 1.2 Quantum Circuit Diagram
- **Purpose**: Detail the specific quantum gates (Amplitude Encoding, QFT, RY/RZ, CNOT Ring Entanglement).
- **Type**: Quantum Circuit Schema.
- **Current Status**: Already generated via `generate_qiskit_circuit.py`.

---

## 2. Recommended UML Diagrams

### 2.1 UML Class Diagram
- **Purpose**: Visualize the object-oriented structure of the PyTorch/Qiskit implementation.
- **Mermaid Code**:
```mermaid
classDiagram
    class HybridQVAE {
        +Integer eeg_dim
        +Integer ecg_dim
        +Integer latent_dim
        +Encoder classical_encoder
        +QuantumLayer quantum_latent
        +Decoder classical_decoder
        +forward(eeg, ecg)
    }
    class EEGPreprocessor {
        +sample_rate
        +process(raw_eeg)
        -apply_ica()
        -bandpass_filter()
    }
    class ECGPreprocessor {
        +sample_rate
        +process(raw_ecg)
        -pan_tompkins()
        -extract_statistical_features()
    }
    class MultimodalDataLoader {
        +batch_size
        +load_sync_data()
    }

    HybridQVAE *-- EEGPreprocessor : uses
    HybridQVAE *-- ECGPreprocessor : uses
    MultimodalDataLoader --> HybridQVAE : feeds
```

### 2.2 UML Sequence Diagram
- **Purpose**: Show the synchronous interaction between components during a single training step.
- **Mermaid Code**:
```mermaid
sequenceDiagram
    participant D as DataLoader
    participant P as Preprocessors
    participant M as HybridQVAE Model
    participant Q as Quantum Simulator
    participant C as Classifier

    D->>P: Send Raw EEG/ECG
    P->>P: Apply ICA & Pan-Tompkins
    P->>M: Return Fused Features (2558-D)
    M->>M: Classical Encode (64-D)
    M->>Q: Amplitude Encoding (6 Qubits)
    Q->>Q: QFT + Variational Layers
    Q->>M: Measurement (64-D Latent)
    M->>C: Pass Latent Vector
    C->>M: Predict Emotion (3 Classes)
    M->>D: Return Reconstruction Loss
```

### 2.3 UML Use Case Diagram
- **Purpose**: Define how the user (Researcher) interacts with the system.
- **Components**:
  - **Actor**: Researcher / Data Scientist.
  - **Use Cases**: 
    - Ingest Multimodal Data.
    - Configure Hyperparameters.
    - Train Hybrid Model.
    - Evaluate Emotion Accuracy.
    - Generate Circuit Visualizations.

### 2.4 Activity Diagram
- **Purpose**: Model the specific logic of the Pan-Tompkins algorithm or the ICA artifact removal process.
- **Steps**: Raw ECG -> Derivative -> Squaring -> Integration -> Peak Finding -> Statistical Feature Extraction.

---

## 3. Visualization Tools
- **Mermaid.js**: Best for Class, Sequence, and Flowcharts (used in Markdown).
- **Qiskit/Pennylane**: Best for Quantum Circuit diagrams.
- **Matplotlib/Seaborn**: Best for Result Plots (Loss, Accuracy, Latent Space).
