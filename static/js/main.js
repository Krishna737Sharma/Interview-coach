class EmotionCoach {
    constructor() {
        this.isActive = false;
        this.videoElement = null;
        this.mediaStream = null;
        this.socket = null;
        this.emotionData = {};
        this.feedbackHistory = [];
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
    }

    initializeElements() {
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.videoElement = document.getElementById('videoElement');
        this.emotionContainer = document.getElementById('emotionData');
        this.feedbackContainer = document.getElementById('feedbackList');
        this.statusIndicators = document.querySelectorAll('.status-indicator');
    }

    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startDetection());
        this.stopBtn.addEventListener('click', () => this.stopDetection());
        
        // Handle page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('WebSocket connected');
        };
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.socket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('WebSocket disconnected');
            // Try to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('WebSocket error');
        };
    }

    handleWebSocketMessage(data) {
        switch(data.type) {
            case 'emotion_update':
                this.updateEmotionDisplay(data.emotions);
                break;
            case 'feedback':
                this.addFeedback(data.message, data.emotion_type);
                break;
            case 'error':
                this.showError(data.message);
                break;
            case 'status':
                this.updateStatus(data.message);
                break;
        }
    }

    async startDetection() {
        try {
            this.updateStatus('Starting camera...');
            
            // Request camera and microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: true
            });
            
            this.videoElement.srcObject = this.mediaStream;
            this.videoElement.play();
            
            this.isActive = true;
            this.updateButtonStates();
            this.updateStatusIndicators(true);
            
            // Start sending frames for processing
            this.startFrameCapture();
            
            this.updateStatus('Detection started');
            this.showSuccess('Emotion detection started successfully!');
            
        } catch (error) {
            console.error('Error starting detection:', error);
            this.showError('Failed to start camera: ' + error.message);
            this.updateStatus('Failed to start detection');
        }
    }

    stopDetection() {
        this.isActive = false;
        this.updateButtonStates();
        this.updateStatusIndicators(false);
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        
        // Send stop signal to server
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ type: 'stop_detection' }));
        }
        
        this.updateStatus('Detection stopped');
        this.showSuccess('Emotion detection stopped.');
    }

    startFrameCapture() {
        if (!this.isActive) return;
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        const captureFrame = () => {
            if (!this.isActive || !this.videoElement.videoWidth) {
                setTimeout(captureFrame, 100);
                return;
            }
            
            canvas.width = this.videoElement.videoWidth;
            canvas.height = this.videoElement.videoHeight;
            ctx.drawImage(this.videoElement, 0, 0);
            
            // Convert to base64 and send to server
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this.socket.send(JSON.stringify({
                    type: 'frame_data',
                    image: imageData.split(',')[1], // Remove data:image/jpeg;base64, prefix
                    timestamp: Date.now()
                }));
            }
            
            // Capture next frame
            setTimeout(captureFrame, 200); // 5 FPS
        };
        
        captureFrame();
    }

    updateEmotionDisplay(emotions) {
        this.emotionData = emotions;
        
        const emotionElements = {
            'happy': document.getElementById('emotion-happy'),
            'sad': document.getElementById('emotion-sad'),
            'angry': document.getElementById('emotion-angry'),
            'surprise': document.getElementById('emotion-surprise'),
            'fear': document.getElementById('emotion-fear'),
            'neutral': document.getElementById('emotion-neutral')
        };
        
        Object.keys(emotions).forEach(emotion => {
            const element = emotionElements[emotion];
            if (element) {
                const value = Math.round(emotions[emotion] * 100);
                const fillElement = element.querySelector('.emotion-fill');
                const valueElement = element.querySelector('.emotion-value');
                
                if (fillElement && valueElement) {
                    fillElement.style.width = `${value}%`;
                    valueElement.textContent = `${value}%`;
                }
            }
        });
    }

    addFeedback(message, emotionType = null) {
        const feedbackItem = document.createElement('div');
        feedbackItem.className = 'feedback-item';
        
        const timestamp = new Date().toLocaleTimeString();
        
        feedbackItem.innerHTML = `
            <div class="feedback-timestamp">${timestamp}</div>
            <div class="feedback-text">${message}</div>
        `;
        
        this.feedbackContainer.insertBefore(feedbackItem, this.feedbackContainer.firstChild);
        
        // Keep only last 10 feedback items
        while (this.feedbackContainer.children.length > 10) {
            this.feedbackContainer.removeChild(this.feedbackContainer.lastChild);
        }
        
        // Store in history
        this.feedbackHistory.unshift({
            message,
            timestamp: new Date(),
            emotionType
        });
        
        if (this.feedbackHistory.length > 50) {
            this.feedbackHistory.pop();
        }
    }

    updateButtonStates() {
        this.startBtn.disabled = this.isActive;
        this.stopBtn.disabled = !this.isActive;
        
        if (this.isActive) {
            this.startBtn.innerHTML = 'Detection Active <span class="loading"></span>';
        } else {
            this.startBtn.innerHTML = 'Start Detection';
        }
    }

    updateStatusIndicators(active) {
        this.statusIndicators.forEach(indicator => {
            if (active) {
                indicator.classList.remove('status-inactive');
                indicator.classList.add('status-active');
            } else {
                indicator.classList.remove('status-active');
                indicator.classList.add('status-inactive');
            }
        });
    }

    updateStatus(message) {
        console.log('Status:', message);
        // You can add a status display element if needed
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        messageDiv.textContent = message;
        
        const container = document.querySelector('.main-content');
        container.insertBefore(messageDiv, container.firstChild);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.parentNode.removeChild(messageDiv);
            }
        }, 5000);
    }

    cleanup() {
        if (this.isActive) {
            this.stopDetection();
        }
        
        if (this.socket) {
            this.socket.close();
        }
    }

    // Export functionality for debugging
    exportFeedbackHistory() {
        const data = {
            emotions: this.emotionData,
            feedback: this.feedbackHistory,
            exportTime: new Date()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `emotion-coach-session-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new EmotionCoach();
    
    // Make app accessible globally for debugging
    window.emotionCoach = app;
    
    // Add export button functionality if exists
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => app.exportFeedbackHistory());
    }
});