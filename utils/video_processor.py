"""
Video processing utility for real-time video handling and face analysis
"""

import cv2
import numpy as np
from collections import deque
import time

class VideoProcessor:
    def __init__(self, input_size=(640, 480)):
        """
        Initialize video processor
        
        Args:
            input_size (tuple): Expected input frame size
        """
        self.input_size = input_size
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Frame processing parameters
        self.brightness_adjustment = 0
        self.contrast_adjustment = 1.0
        
        print("📹 Video processor initialized")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for better emotion detection
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            np.array: Preprocessed frame
        """
        try:
            # Resize frame if needed
            if frame.shape[:2] != self.input_size:
                frame = cv2.resize(frame, self.input_size)
            
            # Auto-adjust brightness and contrast
            frame = self.auto_adjust_lighting(frame)
            
            # Reduce noise
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            
            return frame
            
        except Exception as e:
            print(f"Error preprocessing frame: {str(e)}")
            return frame
    
    def auto_adjust_lighting(self, frame):
        """
        Automatically adjust lighting for better face detection
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            np.array: Adjusted frame
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return frame
            
        except Exception as e:
            print(f"Error adjusting lighting: {str(e)}")
            return frame
    
    def detect_and_analyze_faces(self, frame):
        """
        Detect faces and analyze facial features
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            list: List of face analysis results
        """
        results = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                face_analysis = {
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': w * h,
                    'features': {}
                }
                
                # Extract face region
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
                
                # Analyze facial features
                face_analysis['features'] = self.analyze_facial_features(face_gray, face_color)
                
                # Detect eyes
                eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
                face_analysis['eyes'] = eyes
                face_analysis['num_eyes'] = len(eyes)
                
                # Detect smile
                smiles = self.smile_cascade.detectMultiScale(face_gray, 1.8, 20)
                face_analysis['smiles'] = smiles
                face_analysis['is_smiling'] = len(smiles) > 0
                
                results.append(face_analysis)
            
            return results
            
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return []
    
    def analyze_facial_features(self, face_gray, face_color):
        """
        Analyze facial features for emotion detection
        
        Args:
            face_gray (np.array): Grayscale face region
            face_color (np.array): Color face region
            
        Returns:
            dict: Facial feature analysis
        """
        features = {}
        
        try:
            h, w = face_gray.shape
            
            # Basic geometric features
            features['width'] = w
            features['height'] = h
            features['aspect_ratio'] = w / h if h > 0 else 1.0
            
            # Intensity analysis
            features['mean_intensity'] = np.mean(face_gray)
            features['intensity_std'] = np.std(face_gray)
            features['intensity_range'] = np.max(face_gray) - np.min(face_gray)
            
            # Texture analysis
            features['contrast'] = self.calculate_contrast(face_gray)
            features['homogeneity'] = self.calculate_homogeneity(face_gray)
            features['energy'] = self.calculate_energy(face_gray)
            
            # Edge analysis
            edges = cv2.Canny(face_gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (w * h)
            features['edge_mean'] = np.mean(edges)
            
            # Region-based analysis
            features.update(self.analyze_face_regions(face_gray))
            
            # Color analysis
            if len(face_color.shape) == 3:
                features.update(self.analyze_color_features(face_color))
            
            # Symmetry analysis
            features['symmetry'] = self.calculate_face_symmetry(face_gray)
            
            return features
            
        except Exception as e:
            print(f"Error analyzing facial features: {str(e)}")
            return {}
    
    def analyze_face_regions(self, face_gray):
        """
        Analyze different regions of the face
        
        Args:
            face_gray (np.array): Grayscale face region
            
        Returns:
            dict: Region-based features
        """
        features = {}
        h, w = face_gray.shape
        
        try:
            # Divide face into regions
            # Upper region (forehead and eyes)
            upper_region = face_gray[:h//3, :]
            features['upper_mean'] = np.mean(upper_region)
            features['upper_std'] = np.std(upper_region)
            
            # Middle region (nose and cheeks)
            middle_region = face_gray[h//3:2*h//3, :]
            features['middle_mean'] = np.mean(middle_region)
            features['middle_std'] = np.std(middle_region)
            
            # Lower region (mouth and chin)
            lower_region = face_gray[2*h//3:, :]
            features['lower_mean'] = np.mean(lower_region)
            features['lower_std'] = np.std(lower_region)
            
            # Eye region analysis
            eye_region = face_gray[h//6:h//2, :]
            features['eye_region_mean'] = np.mean(eye_region)
            features['eye_region_contrast'] = np.std(eye_region)
            
            # Mouth region analysis
            mouth_region = face_gray[2*h//3:5*h//6, w//4:3*w//4]
            if mouth_region.size > 0:
                features['mouth_region_mean'] = np.mean(mouth_region)
                features['mouth_region_contrast'] = np.std(mouth_region)
            else:
                features['mouth_region_mean'] = 0.0
                features['mouth_region_contrast'] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Error analyzing face regions: {str(e)}")
            return {}
    
    def analyze_color_features(self, face_color):
        """
        Analyze color-based features
        
        Args:
            face_color (np.array): Color face region
            
        Returns:
            dict: Color features
        """
        features = {}
        
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_color, cv2.COLOR_BGR2LAB)
            
            # HSV analysis
            h, s, v = cv2.split(hsv)
            features['hue_mean'] = np.mean(h)
            features['saturation_mean'] = np.mean(s)
            features['value_mean'] = np.mean(v)
            
            # LAB analysis
            l, a, b = cv2.split(lab)
            features['lightness_mean'] = np.mean(l)
            features['a_channel_mean'] = np.mean(a)
            features['b_channel_mean'] = np.mean(b)
            
            # RGB analysis
            b_ch, g_ch, r_ch = cv2.split(face_color)
            features['red_mean'] = np.mean(r_ch)
            features['green_mean'] = np.mean(g_ch)
            features['blue_mean'] = np.mean(b_ch)
            
            return features
            
        except Exception as e:
            print(f"Error analyzing color features: {str(e)}")
            return {}
    
    def calculate_contrast(self, image):
        """Calculate image contrast using standard deviation"""
        return np.std(image) / 255.0 if np.std(image) > 0 else 0.0
    
    def calculate_homogeneity(self, image):
        """Calculate image homogeneity"""
        try:
            # Simple homogeneity measure based on local variance
            kernel = np.ones((3,3), np.float32) / 9
            mean_img = cv2.filter2D(image.astype(np.float32), -1, kernel)
            variance = np.mean((image.astype(np.float32) - mean_img)**2)
            return 1.0 / (1.0 + variance/1000.0)  # Normalize
        except:
            return 0.5
    
    def calculate_energy(self, image):
        """Calculate image energy"""
        try:
            # Energy based on squared intensity
            normalized = image.astype(np.float32) / 255.0
            return np.mean(normalized**2)
        except:
            return 0.0
    
    def calculate_face_symmetry(self, face_gray):
        """
        Calculate face symmetry score
        
        Args:
            face_gray (np.array): Grayscale face region
            
        Returns:
            float: Symmetry score (0-1)
        """
        try:
            h, w = face_gray.shape
            
            # Split face vertically
            left_half = face_gray[:, :w//2]
            right_half = face_gray[:, w//2:]
            
            # Flip right half
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate similarity
            diff = np.abs(left_half.astype(np.float32) - right_half_flipped.astype(np.float32))
            symmetry = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0.0, min(1.0, symmetry))
            
        except Exception as e:
            print(f"Error calculating symmetry: {str(e)}")
            return 0.5
    
    def process_frame(self, frame):
        """
        Main frame processing function
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            np.array: Processed frame with annotations
        """
        self.frame_count += 1
        
        # Update FPS counter
        current_time = time.time()
        self.fps_counter.append(1.0 / max(0.001, current_time - self.last_time))
        self.last_time = current_time
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame.copy())
        
        # Detect and analyze faces
        face_results = self.detect_and_analyze_faces(processed_frame)
        
        # Draw annotations
        annotated_frame = self.draw_annotations(processed_frame, face_results)
        
        return annotated_frame
    
    def draw_annotations(self, frame, face_results):
        """
        Draw annotations on the frame
        
        Args:
            frame (np.array): Input frame
            face_results (list): Face analysis results
            
        Returns:
            np.array: Annotated frame
        """
        annotated = frame.copy()
        
        try:
            for face in face_results:
                x, y, w, h = face['bbox']
                
                # Draw face rectangle
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw eyes
                for (ex, ey, ew, eh) in face['eyes']:
                    cv2.rectangle(annotated, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
                
                # Draw smile indicator
                if face['is_smiling']:
                    cv2.putText(annotated, 'SMILING', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw face center
                center = face['center']
                cv2.circle(annotated, center, 3, (0, 0, 255), -1)
            
            # Draw FPS
            if self.fps_counter:
                fps = np.mean(self.fps_counter)
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"Error drawing annotations: {str(e)}")
            return frame
    
    def get_fps(self):
        """Get current FPS"""
        return np.mean(self.fps_counter) if self.fps_counter else 0.0
    
    def reset_counters(self):
        """Reset frame counters"""
        self.frame_count = 0
        self.fps_counter.clear()