# Setup Guide - Real-Time Emotion Coach

This guide will help you set up the Real-Time Emotion Coach application on your local machine.

## Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.8 or higher** - [Download Python](https://python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads/)
- **Webcam and Microphone** - For real-time emotion detection
- **VS Code** (recommended) - [Download VS Code](https://code.visualstudio.com/)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/real-time-emotion-coach.git
cd real-time-emotion-coach
```

### 2. Create Virtual Environment

**On Windows:**
```bash
python -m venv emotion_coach_env
emotion_coach_env\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv emotion_coach_env
source emotion_coach_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with some packages, try installing them individually:

```bash
# For audio processing
pip install pyaudio
# If pyaudio fails, on Ubuntu/Debian:
sudo apt-get install portaudio19-dev python3-pyaudio

# For macOS with Homebrew:
brew install portaudio
pip install pyaudio
```

### 4. Download Pre-trained Models

The application will automatically download lightweight models on first run. However, you can manually download them:

```bash
python -c "
import tensorflow as tf
# This will download MobileNetV2 for facial emotion detection
tf.keras.applications.MobileNetV2(weights='imagenet')
"
```

### 5. Test Your Setup

```bash
# Run tests to verify everything is working
python -m pytest tests/ -v

# Or run individual test
python tests/test_emotion_detection.py
```

### 6. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Troubleshooting

### Common Issues and Solutions

#### 1. Camera/Microphone Access Issues

**Problem:** Browser doesn't access camera/microphone
**Solution:** 
- Make sure to allow camera/microphone permissions in your browser
- Use `https://localhost:5000` instead of `http://` if needed
- Check if other applications are using the camera/microphone

#### 2. PyAudio Installation Issues

**Problem:** `pip install pyaudio` fails
**Solution:**

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

#### 3. TensorFlow Installation Issues

**Problem:** TensorFlow installation fails or compatibility issues
**Solution:**
```bash
# Uninstall and reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.13.0

# For Apple Silicon Macs:
pip install tensorflow-macos tensorflow-metal
```

#### 4. OpenCV Issues

**Problem:** OpenCV not working properly
**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.0.74
```

#### 5. Socket.IO Connection Issues

**Problem:** Real-time updates not working
**Solution:**
- Check firewall settings
- Ensure port 5000 is not blocked
- Try running on a different port: `python app.py --port 5001`

## Performance Optimization

### For Better Performance:

1. **Close unnecessary applications** that might use camera/microphone
2. **Use a good quality webcam** for better face detection
3. **Ensure good lighting** for facial emotion detection
4. **Use a quiet environment** for better audio emotion detection

### Hardware Requirements:

- **Minimum:** 4GB RAM, dual-core processor
- **Recommended:** 8GB RAM, quad-core processor
- **Camera:** 720p or higher resolution
- **Microphone:** Built-in or external microphone

## Development Setup

If you want to contribute or modify the code:

### 1. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

### 2. Code Formatting

```bash
# Format code with Black
black .

# Check code style
flake8 .

# Type checking
mypy .
```

### 3. Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=models --cov=utils
```

## Deployment Notes

### Running on Different Devices:

1. **Raspberry Pi:** Works with some performance limitations
2. **Laptop/Desktop:** Best performance
3. **Cloud:** Can be deployed but edge AI benefits are reduced

### Environment Variables:

Create a `.env` file for configuration:
```bash
FLASK_ENV=development
MODEL_PATH=./models/
LOG_LEVEL=INFO
```

## Getting Help

If you encounter issues:

1. Check the [Issues](https://github.com/your-username/real-time-emotion-coach/issues) page
2. Create a new issue with detailed error messages
3. Join our community discussions

## Next Steps

After successful setup:

1. Open `http://localhost:5000` in your browser
2. Allow camera and microphone permissions
3. Click "Start Coaching" to begin emotion detection
4. Practice your communication skills with real-time feedback!

---

**Note:** This application runs entirely on your device (Edge AI) for privacy and performance. No data is sent to external servers.