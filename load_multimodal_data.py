"""
Multimodal EEG + ECG Data Loader
Combines emotions.csv (EEG) with MIT-BIH (ECG) for fusion-based classification
"""

import numpy as np
import pandas as pd
import wfdb
import os
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class MultimodalDataLoader:
    def __init__(self, eeg_path='data/emotions.csv', ecg_path='data/mit-bih-arrhythmia-database-1.0.0'):
        self.eeg_path = eeg_path
        self.ecg_path = ecg_path
        
    def load_eeg_data(self):
        """Load EEG emotions dataset"""
        print("Loading EEG data...")
        df = pd.read_csv(self.eeg_path)
        
        # Separate features and labels
        eeg_labels = df['label'].values
        eeg_features = df.drop('label', axis=1).values
        
        print(f"  EEG samples: {eeg_features.shape[0]}")
        print(f"  EEG features: {eeg_features.shape[1]}")
        print(f"  Unique labels: {np.unique(eeg_labels)}")
        
        return eeg_features, eeg_labels
    
    def load_ecg_data(self, n_samples=2132, segment_length=1000):
        """
        Load ECG data from MIT-BIH database
        
        Args:
            n_samples: Number of samples to match EEG dataset
            segment_length: Length of each ECG segment
        """
        print("Loading ECG data...")
        
        # Get list of record files
        ecg_files = list(Path(self.ecg_path).glob('*.dat'))
        record_names = [f.stem for f in ecg_files]
        
        if not record_names:
            raise FileNotFoundError(f"No ECG records found in {self.ecg_path}")
        
        print(f"  Found {len(record_names)} ECG records")
        
        ecg_segments = []
        ecg_labels = []
        
        # Keep cycling through records until we have enough samples
        record_idx = 0
        attempts = 0
        max_attempts = len(record_names) * 100  # Prevent infinite loop
        
        while len(ecg_segments) < n_samples and attempts < max_attempts:
            attempts += 1
            record_name = record_names[record_idx % len(record_names)]
            
            try:
                # Read ECG record
                record_path = os.path.join(self.ecg_path, record_name)
                record = wfdb.rdrecord(record_path)
                ecg_signal = record.p_signal[:, 0]  # Use first channel (MLII)
                
                # Extract multiple segments from this record
                segments_needed = min(50, n_samples - len(ecg_segments))  # Get up to 50 per record
                
                for _ in range(segments_needed):
                    if len(ecg_segments) >= n_samples:
                        break
                    
                    # Random starting point
                    if len(ecg_signal) > segment_length:
                        start_idx = np.random.randint(0, len(ecg_signal) - segment_length)
                        segment = ecg_signal[start_idx:start_idx + segment_length]
                        
                        # Extract features from segment
                        features = self._extract_ecg_features(segment)
                        ecg_segments.append(features)
                        
                        # Assign label (cycle through 3 classes)
                        ecg_labels.append(len(ecg_segments) % 3)
                
                record_idx += 1
                    
            except Exception as e:
                print(f"  Warning: Could not load {record_name}: {e}")
                record_idx += 1
                continue
        
        ecg_features = np.array(ecg_segments)
        ecg_labels = np.array(ecg_labels)
        
        # Ensure exact size (trim or pad if needed)
        if len(ecg_features) > n_samples:
            ecg_features = ecg_features[:n_samples]
            ecg_labels = ecg_labels[:n_samples]
        elif len(ecg_features) < n_samples:
            print(f"  Warning: Only collected {len(ecg_features)} samples, padding to {n_samples}")
            # Repeat samples to reach target
            while len(ecg_features) < n_samples:
                idx = np.random.randint(0, len(ecg_features))
                ecg_features = np.vstack([ecg_features, ecg_features[idx:idx+1]])
                ecg_labels = np.append(ecg_labels, ecg_labels[idx])
            ecg_features = ecg_features[:n_samples]
            ecg_labels = ecg_labels[:n_samples]
        
        print(f"  ECG samples: {ecg_features.shape[0]}")
        print(f"  ECG features: {ecg_features.shape[1]}")
        
        return ecg_features, ecg_labels
    
    def _extract_ecg_features(self, segment):
        """
        Extract statistical features from ECG segment
        Similar to paper's feature extraction
        """
        features = []
        
        # Time-domain features
        features.append(np.mean(segment))
        features.append(np.std(segment))
        features.append(np.min(segment))
        features.append(np.max(segment))
        features.append(np.median(segment))
        features.append(np.percentile(segment, 25))
        features.append(np.percentile(segment, 75))
        
        # Additional features
        features.append(np.var(segment))
        features.append(segment.max() - segment.min())  # Range
        features.append(np.mean(np.abs(np.diff(segment))))  # Mean absolute difference
        
        return np.array(features)
    
    def create_multimodal_dataset(self, fusion_strategy='concat'):
        """
        Create fused EEG + ECG dataset
        
        Args:
            fusion_strategy: 'concat' (concatenate features) or 'separate' (keep separate)
        
        Returns:
            features, labels
        """
        # Load both modalities
        eeg_features, eeg_labels = self.load_eeg_data()
        
        # Load ECG with same number of samples as EEG
        ecg_features, ecg_labels = self.load_ecg_data(n_samples=len(eeg_features))
        
        if fusion_strategy == 'concat':
            # Concatenate EEG and ECG features
            print("\nFusing EEG + ECG features...")
            
            # Normalize each modality separately first
            scaler_eeg = StandardScaler()
            scaler_ecg = StandardScaler()
            
            eeg_normalized = scaler_eeg.fit_transform(eeg_features)
            ecg_normalized = scaler_ecg.fit_transform(ecg_features)
            
            # Concatenate
            fused_features = np.concatenate([eeg_normalized, ecg_normalized], axis=1)
            
            print(f"  Fused features: {fused_features.shape[1]} (EEG: {eeg_features.shape[1]} + ECG: {ecg_features.shape[1]})")
            
            # Use EEG labels (emotion labels are more relevant)
            labels = eeg_labels
            
            return fused_features, labels
        
        elif fusion_strategy == 'separate':
            # Return separate (for quantum fusion layer)
            return (eeg_features, ecg_features), eeg_labels


def save_multimodal_data(output_path='data/multimodal_fused.csv'):
    """
    Create and save fused multimodal dataset
    """
    loader = MultimodalDataLoader()
    features, labels = loader.create_multimodal_dataset(fusion_strategy='concat')
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved multimodal dataset to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {features.shape[1]}")
    
    return output_path


if __name__ == "__main__":
    # Test loading
    save_multimodal_data()
