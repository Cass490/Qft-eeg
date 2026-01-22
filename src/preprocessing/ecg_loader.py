import numpy as np
import scipy.signal as signal

class ECGPreprocessor:
    def __init__(self, sample_rate=360):
        """
        Initializes the ECG Preprocessor.
        
        Args:
            sample_rate (int): Sampling rate of the ECG data (MIT-BIH is 360 Hz).
        """
        self.fs = sample_rate

    def bandpass_filter(self, data, lowcut=0.5, highcut=45.0, order=3):
        """
        Standard bandpass to remove baseline wander and high freq noise.
        Paper mentions 0.5-45 Hz (Fig 3 caption).
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def pan_tompkins_detect(self, ecg_signal):
        """
        Implements the Pan-Tompkins algorithm steps for QRS enhancement (Section 2.3.2).
        Returns the processed signal suitable for peak detection.
        """
        # 1. Bandpass (5-15Hz is standard for QRS, though paper implies general filtering first)
        # We assume 'ecg_signal' is already pre-filtered by the general bandpass above.
        
        # 2. Differentiation (Eq 5 implies finding high slopes)
        # H(z) = (1/8T)(-z^-2 - 2z^-1 + 2z^1 + z^2) - approximated by diff
        diff_signal = np.diff(ecg_signal)
        # Pad to keep length same
        diff_signal = np.append(diff_signal, 0) 

        # 3. Squaring (Eq 5)
        squared_signal = diff_signal ** 2

        # 4. Moving Window Integration
        # Window width roughly 150ms (0.150 * fs)
        window_size = int(0.150 * self.fs)
        integrated_signal = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')

        return integrated_signal

    def get_r_peaks(self, integrated_signal, threshold=None):
        """
        Basic peak detection on the Pan-Tompkins integrated signal.
        """
        if threshold is None:
            threshold = np.mean(integrated_signal) * 2
        
        peaks, _ = signal.find_peaks(integrated_signal, height=threshold, distance=self.fs*0.4)
        return peaks

    def process(self, ecg_data):
        """
        Full pipeline: Bandpass -> Pan-Tompkins Enhancement
        """
        # 1. Baseline Wander & Noise Removal
        filtered_ecg = self.bandpass_filter(ecg_data)
        
        # 2. Pan-Tompkins steps (for Feature Extraction/QRS detection)
        # The paper uses this to enhance QRS for detection.
        enhanced_ecg = self.pan_tompkins_detect(filtered_ecg)
        
        return filtered_ecg, enhanced_ecg
