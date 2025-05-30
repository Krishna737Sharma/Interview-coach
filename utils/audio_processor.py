"""
Audio processing utility for real-time audio handling
"""

import numpy as np
import librosa
from scipy import signal
import threading
import time

class AudioProcessor:
    def __init__(self, sample_rate=22050, buffer_size=2048):
        """
        Initialize audio processor
        
        Args:
            sample_rate (int): Audio sample rate
            buffer_size (int): Buffer size for processing
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_processing = False
        
        # Audio filters
        self.high_pass_filter = None
        self.low_pass_filter = None
        self.setup_filters()
        
        print("🎵 Audio processor initialized")
    
    def setup_filters(self):
        """Setup audio filters for noise reduction"""
        # High-pass filter to remove low-frequency noise
        nyquist = self.sample_rate * 0.5
        high_cutoff = 300 / nyquist  # 300 Hz high-pass
        self.high_pass_filter = signal.butter(4, high_cutoff, btype='high')
        
        # Low-pass filter to remove high-frequency noise
        low_cutoff = 8000 / nyquist  # 8 kHz low-pass
        self.low_pass_filter = signal.butter(4, low_cutoff, btype='low')
    
    def preprocess_audio(self, audio_data):
        """
        Preprocess audio data for emotion detection
        
        Args:
            audio_data (np.array): Raw audio data
            
        Returns:
            np.array: Preprocessed audio data
        """
        try:
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Apply high-pass filter
            if len(audio_data) > 10:  # Minimum length for filtering
                audio_data = signal.filtfilt(self.high_pass_filter[0], self.high_pass_filter[1], audio_data)
                audio_data = signal.filtfilt(self.low_pass_filter[0], self.low_pass_filter[1], audio_data)
            
            # Remove silence (simple voice activity detection)
            audio_data = self.remove_silence(audio_data)
            
            return audio_data
            
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
            return audio_data
    
    def remove_silence(self, audio_data, threshold=0.01):
        """
        Remove silent parts from audio
        
        Args:
            audio_data (np.array): Input audio
            threshold (float): Silence threshold
            
        Returns:
            np.array: Audio with silence removed
        """
        try:
            # Calculate energy in sliding windows
            window_size = 1024
            hop_length = 512
            
            if len(audio_data) < window_size:
                return audio_data
            
            # Calculate RMS energy
            energy = []
            for i in range(0, len(audio_data) - window_size, hop_length):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window**2))
                energy.append(rms)
            
            energy = np.array(energy)
            
            # Find non-silent regions
            non_silent = energy > threshold
            
            if np.any(non_silent):
                # Keep audio segments above threshold
                result = []
                for i, is_voice in enumerate(non_silent):
                    if is_voice:
                        start = i * hop_length
                        end = min(start + window_size, len(audio_data))
                        result.extend(audio_data[start:end])
                
                return np.array(result) if result else audio_data
            else:
                return audio_data
                
        except Exception as e:
            print(f"Error removing silence: {str(e)}")
            return audio_data
    
    def extract_prosodic_features(self, audio_data):
        """
        Extract prosodic features (pitch, rhythm, etc.)
        
        Args:
            audio_data (np.array): Input audio
            
        Returns:
            dict: Prosodic features
        """
        features = {}
        
        try:
            # Pitch extraction
            pitches, magnitudes = librosa.core.piptrack(
                y=audio_data, 
                sr=self.sample_rate, 
                threshold=0.1
            )
            
            # Extract fundamental frequency
            f0 = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0.append(pitch)
            
            if f0:
                features['pitch_mean'] = np.mean(f0)
                features['pitch_std'] = np.std(f0)
                features['pitch_range'] = np.max(f0) - np.min(f0)
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
            
            # Rhythm features
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, 
                sr=self.sample_rate
            )
            features['rhythm_density'] = len(onset_frames) / (len(audio_data) / self.sample_rate)
            
            # Energy features
            rms = librosa.feature.rms(y=audio_data)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # Speaking rate (approximate)
            features['speaking_rate'] = len(onset_frames) / max(1, len(audio_data) / self.sample_rate)
            
            return features
            
        except Exception as e:
            print(f"Error extracting prosodic features: {str(e)}")
            return {
                'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0,
                'rhythm_density': 0.0, 'energy_mean': 0.0, 'energy_std': 0.0,
                'speaking_rate': 0.0
            }
    
    def calculate_audio_quality(self, audio_data):
        """
        Calculate audio quality metrics
        
        Args:
            audio_data (np.array): Input audio
            
        Returns:
            dict: Audio quality metrics
        """
        quality = {}
        
        try:
            # Signal-to-noise ratio estimate
            signal_power = np.mean(audio_data**2)
            noise_power = np.var(audio_data - np.mean(audio_data))
            
            if noise_power > 0:
                quality['snr_estimate'] = 10 * np.log10(signal_power / noise_power)
            else:
                quality['snr_estimate'] = float('inf')
            
            # Dynamic range
            quality['dynamic_range'] = np.max(audio_data) - np.min(audio_data)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            quality['zero_crossing_rate'] = np.mean(zcr)
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            quality['spectral_centroid'] = np.mean(spectral_centroids)
            
            return quality
            
        except Exception as e:
            print(f"Error calculating audio quality: {str(e)}")
            return {
                'snr_estimate': 0.0, 'dynamic_range': 0.0,
                'zero_crossing_rate': 0.0, 'spectral_centroid': 0.0
            }
    
    def is_speech_present(self, audio_data, threshold=0.02):
        """
        Simple voice activity detection
        
        Args:
            audio_data (np.array): Input audio
            threshold (float): Energy threshold
            
        Returns:
            bool: True if speech is detected
        """
        try:
            # Calculate short-term energy
            energy = np.mean(audio_data**2)
            
            # Check if energy is above threshold
            return energy > threshold
            
        except:
            return False
    
    def get_audio_level_db(self, audio_data):
        """
        Get audio level in dB
        
        Args:
            audio_data (np.array): Input audio
            
        Returns:
            float: Audio level in dB
        """
        try:
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                db = 20 * np.log10(rms)
                return max(db, -60)  # Minimum -60 dB
            else:
                return -60
        except:
            return -60