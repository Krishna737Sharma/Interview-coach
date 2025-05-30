"""
Video-based emotion detection using lightweight models
Optimized for edge deployment with OpenCV and face detection
"""

import cv2
import numpy as np
from collections import deque
import time
import threading

class VideoEmotionDetector:
    def __init__(self, model_path=None):
        """
        Initialize video emotion detector
        
        Args:
            model_path (str): Path to pre-trained model (optional)
        """
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.emotion_weights = {
            'angry': [0.8, 0.1, 0.3, 0.0, 0.0, 0.2, 0.1],
            'disgust': [0.2, 0.8, 0.2, 0.0, 0.1, 0.3, 0.0],
            'fear': [0.3, 0.1, 0.8, 0.0, 0.1, 0.4, 0.3],
            'happy': [0.0, 0.0, 0.0, 0.9, 0.2, 0.0, 0.1],
            'neutral': [0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.0],
            'sad': [0.2, 0.1, 0.3, 0.0, 0.1, 0.8, 0.0],
            'surprised': [0.1, 0.0, 0.2, 0.2, 0.0, 0.0, 0.9]
        }
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Frame processing
        self.frame_buffer = deque(maxlen=30)  # Store last 30 frames
        self.current_frame = None
        self.last_prediction = {'emotion': 'neutral', 'confidence': 0.0}
        
        # Feature tracking
        self.facial_landmarks = []
        self.emotion_history = deque(maxlen=10)
        
        print("✅ Video emotion detector initialized")
    
    def detect_faces(self, frame):
        """
        Detect faces in the frame
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            list: List of face rectangles
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def extract_facial_features(self, face_region):
        """
        Extract facial features for emotion analysis
        
        Args:
            face_region (np.array): Cropped face region
            
        Returns:
            dict: Extracted facial features
        """
        features = {}
        
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Basic geometric features
            h, w = gray_face.shape
            features['aspect_ratio'] = w / h if h > 0 else 1.0
            
            # Eye detection
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
            features['num_eyes'] = len(eyes)
            features['eye_area_ratio'] = 0.0
            
            if len(eyes) >= 2:
                # Calculate eye separation and area
                eye_areas = [eye[2] * eye[3] for eye in eyes]
                features['eye_area_ratio'] = sum(eye_areas) / (w * h)
                
                # Eye positions for expression analysis
                eye_centers = [(eye[0] + eye[2]//2, eye[1] + eye[3]//2) for eye in eyes[:2]]
                if len(eye_centers) == 2:
                    features['eye_distance'] = np.sqrt(
                        (eye_centers[0][0] - eye_centers[1][0])**2 + 
                        (eye_centers[0][1] - eye_centers[1][1])**2
                    ) / w
            
            # Intensity analysis
            features['mean_intensity'] = np.mean(gray_face)
            features['intensity_std'] = np.std(gray_face)
            
            # Edge density (for expression lines)
            edges = cv2.Canny(gray_face, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (w * h)
            
            # Histogram features
            hist = cv2.calcHist([gray_face], [0], None, [16], [0, 256])
            features['hist_peak'] = np.argmax(hist)
            features['hist_variance'] = np.var(hist)
            
            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['num_contours'] = len(contours)
            
            # Mouth region analysis (bottom half of face)
            mouth_region = gray_face[h//2:, :]
            features['mouth_intensity'] = np.mean(mouth_region)
            features['mouth_contrast'] = np.std(mouth_region)
            
            return features
            
        except Exception as e:
            print(f"Error extracting facial features: {str(e)}")
            return {
                'aspect_ratio': 1.0, 'num_eyes': 0, 'eye_area_ratio': 0.0,
                'eye_distance': 0.0, 'mean_intensity': 128, 'intensity_std': 0.0,
                'edge_density': 0.0, 'hist_peak': 8, 'hist_variance': 0.0,
                'num_contours': 0, 'mouth_intensity': 128, 'mouth_contrast': 0.0
            }
    
    def analyze_emotion_from_features(self, features):
        """
        Analyze emotion based on extracted features
        
        Args:
            features (dict): Facial features
            
        Returns:
            dict: Emotion prediction
        """
        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        
        try:
            # Simple rule-based emotion detection
            # These rules are simplified for demo purposes
            
            # Happy detection
            if features['mouth_intensity'] > 140 and features['eye_area_ratio'] > 0.05:
                emotion_scores['happy'] += 0.6
            
            # Sad detection
            if features['mouth_intensity'] < 100 and features['intensity_std'] < 30:
                emotion_scores['sad'] += 0.5
            
            # Angry detection
            if features['edge_density'] > 0.3 and features['eye_distance'] < 0.3:
                emotion_scores['angry'] += 0.4
            
            # Surprised detection
            if features['eye_area_ratio'] > 0.08 and features['num_contours'] > 10:
                emotion_scores['surprised'] += 0.5
            
            # Fear detection
            if features['intensity_std'] > 50 and features['eye_area_ratio'] > 0.06:
                emotion_scores['fear'] += 0.4
            
            # Neutral baseline
            emotion_scores['neutral'] += 0.3
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            return {
                'emotion': dominant_emotion,
                'confidence': float(confidence),
                'scores': emotion_scores
            }
            
        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")
            return {'emotion': 'neutral', 'confidence': 0.0, 'scores': {}}
    
    def process_frame(self, frame):
        """
        Process a single frame for emotion detection
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            np.array: Processed frame with annotations
        """
        self.current_frame = frame.copy()
        self.frame_buffer.append(frame)
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        processed_frame = frame.copy()
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size > 0:
                # Extract features and predict emotion
                features = self.extract_facial_features(face_region)
                emotion_result = self.analyze_emotion_from_features(features)
                
                # Update last prediction
                self.last_prediction = emotion_result
                self.emotion_history.append(emotion_result)
                
                # Draw emotion text
                emotion_text = f"{emotion_result['emotion']}: {emotion_result['confidence']:.2f}"
                cv2.putText(processed_frame, emotion_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return processed_frame
    
    def predict_emotion(self):
        """
        Get current emotion prediction
        
        Returns:
            dict: Current emotion prediction
        """
        if len(self.emotion_history) > 0:
            # Return smoothed prediction from recent history
            recent_emotions = list(self.emotion_history)[-5:]  # Last 5 predictions
            
            # Average the confidence scores
            emotion_avg = {}
            for emotion in self.emotions:
                scores = [pred.get('scores', {}).get(emotion, 0.0) for pred in recent_emotions if 'scores' in pred]
                emotion_avg[emotion] = np.mean(scores) if scores else 0.0
            
            if any(emotion_avg.values()):
                dominant_emotion = max(emotion_avg, key=emotion_avg.get)
                confidence = emotion_avg[dominant_emotion]
                
                return {
                    'emotion': dominant_emotion,
                    'confidence': float(confidence),
                    'all_scores': emotion_avg
                }
        
        return self.last_prediction
    
    def get_frame_count(self):
        """Get number of processed frames"""
        return len(self.frame_buffer)
    
    def reset(self):
        """Reset the detector state"""
        self.frame_buffer.clear()
        self.emotion_history.clear()
        self.last_prediction = {'emotion': 'neutral', 'confidence': 0.0}
        print("🔄 Video emotion detector reset")