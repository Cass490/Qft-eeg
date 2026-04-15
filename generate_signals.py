import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os

def generate_combined_signals():
    os.makedirs('figures', exist_ok=True)
    
    # 1. Load/Generate Data
    ecg_dir = 'data/mit-bih-arrhythmia-database-1.0.0'
    try:
        record = wfdb.rdrecord(os.path.join(ecg_dir, '100'), sampto=1000)
        ecg_signal = record.p_signal[:, 0]
    except Exception as e:
        t = np.linspace(0, 1, 1000)
        ecg_signal = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 3.0 * t)

    # EEG: Synthetic oscillatory signal
    fs = 500
    t = np.arange(0, len(ecg_signal)/fs, 1/fs)
    if len(t) > len(ecg_signal): t = t[:len(ecg_signal)]
    eeg_signal = 0.5 * np.sin(2 * np.pi * 7 * t) + 0.2 * np.sin(2 * np.pi * 12 * t)
    eeg_signal += 0.1 * np.random.normal(size=len(t))

    # 2. Plotting - Combined with Axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, dpi=200)
    
    # ECG - Red
    ax1.plot(ecg_signal, color='#D32F2F', lw=1.2, label='ECG')
    ax1.set_ylabel('Amplitude (mV)', fontsize=9)
    ax1.set_title('Multimodal Physiological Input', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # EEG - Black
    ax2.plot(eeg_signal, color='black', lw=1.0, label='EEG')
    ax2.set_xlabel('Time Samples', fontsize=9)
    ax2.set_ylabel('Amplitude (uV)', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/combined_signals.png', bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()
    print("Generated figures/combined_signals.png (Combined EEG/ECG with Axes)")

if __name__ == "__main__":
    generate_combined_signals()
