"""
Audio-based emotion detection using lightweight models
Optimized for edge deployment
"""

import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pyaudio
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class AudioEmotionDetector:
    def __init__(self, model_path=None):
        """
        Initialize audio emotion detector
        
        Args:
            model_path (str): Path to pre-trained model (optional)
        """
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.model = None
        self.scaler = StandardScaler()
        self.is_recording = False
        self.audio_buffer = deque(maxlen=100)  # Store last 100 audio chunks
        self.sample_rate = 22050
        self.chunk_size = 1024
        
        # Audio recording parameters
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize model
        self._initialize_model(model_path)
        self._start_audio_capture()
    
    def _initialize_model(self, model_path=None):
        """Initialize or load the emotion detection model"""
        if model_path:
            try:
                # Load pre-trained model
                self.model = joblib.load(model_path)
                print("✅ Loaded pre-trained audio emotion model")
            except:
                print("⚠️  Could not load pre-trained model, creating new one")
                self._create_simple_model()
        else:
            self._create_simple_model()
    
    def _create_simple_model(self):
        """Create a simple model for demo purposes"""
        # This is a placeholder model for demonstration
        # In a real implementation, you would train this on actual data
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Create dummy training data for demo
        dummy_features = np.random.rand(1000, 40)  # 40 MFCC features
        dummy_labels = np.random.choice(len(self.emotions), 1000)
        
        # Fit scaler and model
        dummy_features_scaled = self.scaler.fit_transform(dummy_features)
        self.model.fit(dummy_features_scaled, dummy_labels)
        
        print("✅ Created demo audio emotion model")
    
    def _start_audio_capture(self):
        """Start continuous audio capture"""
        try:
            self.stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.is_recording = True
            print("🎙️  Audio capture started")
        except Exception as e:
            print(f"❌ Error starting audio capture: {str(e)}")
            self.is_recording = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.append(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def extract_audio_features(self, audio_data):
        """
        Extract audio features for emotion detection
        
        Args:
            audio_data (np.array): Raw audio data
            
        Returns:
            np.array: Extracted features
        """
        try:
            # Resample if necessary
            if len(audio_data) < self.sample_rate:
                audio_data = np.pad(audio_data, (0, self.sample_rate - len(audio_data)))
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [np.mean(spectral_centroids)],
                [np.mean(spectral_rolloff)],
                [np.mean(zero_crossing_rate)],
                np.mean(chroma, axis=1)
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return np.zeros(40)  # Return zero features if extraction fails
    
    def predict_emotion(self):
        """
        Predict emotion from current audio buffer
        
        Returns:
            dict: Emotion prediction with confidence
        """
        if not self.is_recording or len(self.audio_buffer) < 10:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        try:
            # Combine recent audio chunks
            recent_audio = np.concatenate(list(self.audio_buffer)[-10:])
            
            # Extract features
            features = self.extract_audio_features(recent_audio)
            
            # Predict emotion
            if self.model is not None:
                features_scaled = self.scaler.transform([features])
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                
                emotion = self.emotions[prediction]
                confidence = float(np.max(probabilities))
                
                return {
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': {emotion: float(prob) for emotion, prob in zip(self.emotions, probabilities)}
                }
            else:
                return {'emotion': 'neutral', 'confidence': 0.0}
                
        except Exception as e:
            print(f"Error predicting emotion: {str(e)}")
            return {'emotion': 'neutral', 'confidence': 0.0}
    
    def get_audio_level(self):
        """Get current audio input level"""
        if len(self.audio_buffer) > 0:
            recent_audio = self.audio_buffer[-1]
            return float(np.sqrt(np.mean(recent_audio**2)))
        return 0.0
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("🔇 Audio recording stopped")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_recording()
        except:
            pass