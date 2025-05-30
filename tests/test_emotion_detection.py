import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.audio_emotion_detector import AudioEmotionDetector
from models.video_emotion_detector import VideoEmotionDetector

class TestEmotionDetection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.audio_detector = AudioEmotionDetector()
        self.video_detector = VideoEmotionDetector()
    
    def test_audio_detector_initialization(self):
        """Test if audio detector initializes correctly"""
        self.assertIsNotNone(self.audio_detector.model)
        self.assertEqual(len(self.audio_detector.emotion_labels), 8)
        self.assertIn('happy', self.audio_detector.emotion_labels)
        self.assertIn('sad', self.audio_detector.emotion_labels)
    
    def test_video_detector_initialization(self):
        """Test if video detector initializes correctly"""
        self.assertIsNotNone(self.video_detector.model)
        self.assertEqual(len(self.video_detector.emotion_labels), 7)
        self.assertIn('happy', self.video_detector.emotion_labels)
        self.assertIn('neutral', self.video_detector.emotion_labels)
    
    def test_audio_preprocessing(self):
        """Test audio preprocessing functionality"""
        # Create dummy audio data
        dummy_audio = np.random.rand(16000)  # 1 second of audio at 16kHz
        
        try:
            features = self.audio_detector.preprocess_audio(dummy_audio)
            self.assertIsNotNone(features)
            self.assertIsInstance(features, np.ndarray)
        except Exception as e:
            self.fail(f"Audio preprocessing failed: {e}")
    
    def test_video_preprocessing(self):
        """Test video preprocessing functionality"""
        # Create dummy image data (48x48 grayscale)
        dummy_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        
        try:
            processed_image = self.video_detector.preprocess_image(dummy_image)
            self.assertIsNotNone(processed_image)
            self.assertEqual(processed_image.shape, (1, 48, 48, 1))
        except Exception as e:
            self.fail(f"Video preprocessing failed: {e}")
    
    def test_audio_emotion_prediction(self):
        """Test audio emotion prediction"""
        # Create dummy audio features
        dummy_features = np.random.rand(40)  # MFCC features
        
        try:
            emotion, confidence = self.audio_detector.predict_emotion(dummy_features)
            self.assertIsInstance(emotion, str)
            self.assertIn(emotion, self.audio_detector.emotion_labels)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
        except Exception as e:
            self.fail(f"Audio emotion prediction failed: {e}")
    
    def test_video_emotion_prediction(self):
        """Test video emotion prediction"""
        # Create dummy image (48x48 grayscale)
        dummy_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        
        try:
            emotion, confidence = self.video_detector.predict_emotion(dummy_image)
            self.assertIsInstance(emotion, str)
            self.assertIn(emotion, self.video_detector.emotion_labels)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
        except Exception as e:
            self.fail(f"Video emotion prediction failed: {e}")
    
    def test_face_detection(self):
        """Test face detection functionality"""
        # Create dummy color image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        try:
            faces = self.video_detector.detect_faces(dummy_image)
            self.assertIsInstance(faces, list)
            # Note: faces might be empty for random image, which is expected
        except Exception as e:
            self.fail(f"Face detection failed: {e}")
    
    def test_emotion_labels_consistency(self):
        """Test that emotion labels are consistent"""
        audio_emotions = set(self.audio_detector.emotion_labels)
        video_emotions = set(self.video_detector.emotion_labels)
        
        # Check for common emotions
        common_emotions = audio_emotions.intersection(video_emotions)
        self.assertGreater(len(common_emotions), 0, "Should have some common emotions")
        
        # Check for expected emotions
        expected_emotions = {'happy', 'sad', 'angry', 'neutral'}
        for emotion in expected_emotions:
            self.assertIn(emotion, audio_emotions, f"{emotion} should be in audio emotions")
            self.assertIn(emotion, video_emotions, f"{emotion} should be in video emotions")

class TestCoachingFeedback(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.audio_detector = AudioEmotionDetector()
        self.video_detector = VideoEmotionDetector()
    
    def test_coaching_message_generation(self):
        """Test coaching message generation"""
        test_emotions = ['happy', 'sad', 'angry', 'neutral']
        
        for emotion in test_emotions:
            if emotion in self.audio_detector.emotion_labels:
                message = self.audio_detector.get_coaching_message(emotion)
                self.assertIsInstance(message, str)
                self.assertGreater(len(message), 0)
    
    def test_confidence_threshold(self):
        """Test confidence threshold functionality"""
        # Test with low confidence
        low_confidence_emotion = "happy"
        low_confidence = 0.3
        
        # Should return a generic message for low confidence
        message = self.audio_detector.get_coaching_message(
            low_confidence_emotion, low_confidence
        )
        self.assertIsInstance(message, str)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)