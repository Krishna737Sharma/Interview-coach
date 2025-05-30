#!/usr/bin/env python3
"""
Real-Time Emotion Coach using Edge AI
Main Flask application for emotion detection and coaching
"""

import os
import cv2
import numpy as np
import threading
import time
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import json
from datetime import datetime

# Import our custom modules
from models.audio_emotion_detector import AudioEmotionDetector
from models.video_emotion_detector import VideoEmotionDetector
from utils.audio_processor import AudioProcessor
from utils.video_processor import VideoProcessor

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion_coach_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
emotion_detector_audio = None
emotion_detector_video = None
audio_processor = None
video_processor = None
current_emotions = {
    'audio': {'emotion': 'neutral', 'confidence': 0.0},
    'video': {'emotion': 'neutral', 'confidence': 0.0},
    'combined': {'emotion': 'neutral', 'confidence': 0.0}
}
coaching_feedback = []

def initialize_models():
    """Initialize all AI models for emotion detection"""
    global emotion_detector_audio, emotion_detector_video, audio_processor, video_processor
    
    print("Initializing emotion detection models...")
    
    try:
        # Initialize audio components
        audio_processor = AudioProcessor()
        emotion_detector_audio = AudioEmotionDetector()
        
        # Initialize video components
        video_processor = VideoProcessor()
        emotion_detector_video = VideoEmotionDetector()
        
        print("✅ All models initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing models: {str(e)}")
        return False

def generate_coaching_feedback(emotion_data):
    """Generate coaching feedback based on detected emotions"""
    
    # Load coaching messages
    coaching_messages = {
        'angry': [
            "Take a deep breath and try to lower your voice tone",
            "Consider pausing for a moment to collect your thoughts",
            "Your expression shows tension - try relaxing your facial muscles"
        ],
        'sad': [
            "Try to speak with more energy and enthusiasm",
            "Consider sitting up straighter to project confidence",
            "Your tone sounds low - try speaking a bit more clearly"
        ],
        'happy': [
            "Great energy! Your positive emotions are coming through clearly",
            "Maintain this enthusiastic tone - it's very engaging",
            "Your facial expressions show warmth and approachability"
        ],
        'surprised': [
            "Your reaction seems genuine - that's good for authenticity",
            "Try to maintain steady eye contact",
            "Consider moderating your expressions for professional settings"
        ],
        'fear': [
            "Take a moment to breathe and center yourself",
            "Your nervousness is showing - try slowing down your speech",
            "Remember to make eye contact to build confidence"
        ],
        'neutral': [
            "Consider adding more expression to engage your audience",
            "Your tone is steady - try varying it for more impact",
            "Good baseline - you can build energy from here"
        ]
    }
    
    combined_emotion = emotion_data['combined']['emotion']
    confidence = emotion_data['combined']['confidence']
    
    if confidence > 0.6:
        if combined_emotion in coaching_messages:
            return np.random.choice(coaching_messages[combined_emotion])
    
    return "Continue expressing yourself naturally"

def emotion_analysis_thread():
    """Background thread for continuous emotion analysis"""
    global current_emotions, coaching_feedback
    
    while True:
        try:
            # Process audio emotions
            if audio_processor and emotion_detector_audio:
                audio_emotion = emotion_detector_audio.predict_emotion()
                if audio_emotion:
                    current_emotions['audio'] = audio_emotion
            
            # Process video emotions  
            if video_processor and emotion_detector_video:
                video_emotion = emotion_detector_video.predict_emotion()
                if video_emotion:
                    current_emotions['video'] = video_emotion
            
            # Combine emotions (weighted average)
            audio_conf = current_emotions['audio']['confidence']
            video_conf = current_emotions['video']['confidence']
            
            if audio_conf > 0 or video_conf > 0:
                # Weight video slightly higher as facial expressions are often more reliable
                total_weight = (audio_conf * 0.4) + (video_conf * 0.6)
                
                if total_weight > 0:
                    if video_conf > audio_conf:
                        combined_emotion = current_emotions['video']['emotion']
                    else:
                        combined_emotion = current_emotions['audio']['emotion']
                    
                    current_emotions['combined'] = {
                        'emotion': combined_emotion,
                        'confidence': total_weight,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Generate coaching feedback
            feedback = generate_coaching_feedback(current_emotions)
            if feedback and len(coaching_feedback) < 5:  # Keep last 5 feedbacks
                coaching_feedback.append({
                    'message': feedback,
                    'timestamp': datetime.now().isoformat(),
                    'emotion': current_emotions['combined']['emotion']
                })
                
                # Emit real-time update to frontend
                socketio.emit('emotion_update', {
                    'emotions': current_emotions,
                    'feedback': coaching_feedback[-1]
                })
            
            time.sleep(0.5)  # Update every 500ms
            
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            time.sleep(1)

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/api/emotions')
def get_emotions():
    """API endpoint to get current emotion data"""
    return jsonify({
        'emotions': current_emotions,
        'feedback': coaching_feedback[-1] if coaching_feedback else None,
        'status': 'active'
    })

@app.route('/api/start_session')
def start_session():
    """Start a new emotion coaching session"""
    global coaching_feedback
    coaching_feedback = []
    
    return jsonify({
        'status': 'started',
        'message': 'Emotion coaching session started',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to Emotion Coach'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def generate_video_feed():
    """Generate video feed for streaming"""
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process frame for emotion detection
            if video_processor:
                processed_frame = video_processor.process_frame(frame)
                
                # Add emotion overlay
                emotion = current_emotions['video']['emotion']
                confidence = current_emotions['video']['confidence']
                
                cv2.putText(processed_frame, f'Emotion: {emotion}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f'Confidence: {confidence:.2f}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 Starting Real-Time Emotion Coach...")
    
    # Initialize models
    if initialize_models():
        # Start emotion analysis thread
        emotion_thread = threading.Thread(target=emotion_analysis_thread, daemon=True)
        emotion_thread.start()
        
        print("🎯 Starting web server...")
        print("📱 Open http://localhost:5000 in your browser")
        
        # Run the Flask app
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize models. Please check your setup.")