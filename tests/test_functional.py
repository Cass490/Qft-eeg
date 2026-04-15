import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.eeg_loader import EEGPreprocessor
from src.preprocessing.ecg_loader import ECGPreprocessor
from load_multimodal_data import MultimodalDataLoader
from src.models.qvae import HybridQVAE

class TestFunctionalQVAE(unittest.TestCase):
    
    def test_eeg_preprocessing(self):
        """TC-01: Verify EEG preprocessor handles dummy data"""
        eeg_proc = EEGPreprocessor(sample_rate=128)
        dummy_eeg = np.random.randn(10, 128) # eeg loader applies filtfilt over axis=-1
        clean_eeg = eeg_proc.process(dummy_eeg)
        self.assertEqual(clean_eeg.shape, (10, 128))
        
    def test_ecg_preprocessing(self):
        """TC-02: Verify ECG preprocessor handles dummy data"""
        ecg_proc = ECGPreprocessor(sample_rate=360)
        dummy_ecg = np.random.randn(360) # ecg loader expects 1D signal for pan-tompkins
        clean_ecg, enhanced_ecg = ecg_proc.process(dummy_ecg)
        self.assertEqual(clean_ecg.shape, (360,))
        self.assertEqual(enhanced_ecg.shape, (360,))
        
    def test_multimodal_data_loader(self):
        """TC-03: Validate DataLoader logic (mocking actual paths)"""
        try:
            loader = MultimodalDataLoader()
            self.assertTrue(hasattr(loader, 'load_eeg_data'))
            self.assertTrue(hasattr(loader, 'load_ecg_data'))
        except Exception as e:
            self.fail(f"DataLoader initialization failed: {e}")
            
    def test_qvae_encoder_decoder_dimension(self):
        """TC-04/05: Verify Hybrid QVAE dimension integrity"""
        model = HybridQVAE(input_dim=2558) 
        batch = torch.randn(4, 2558)
        
        recon, latent = model(batch)
        
        self.assertEqual(recon.shape, (4, 2558))
        self.assertEqual(latent.shape, (4, 64))

if __name__ == '__main__':
    unittest.main()
