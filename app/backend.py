# app.py
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import threading
import time
import streamlink
import m3u8
import requests
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Global variables for live stream handling
live_stream = None
live_stream_url = None
stop_live_stream = False
current_prediction = "No prediction yet"

# Path configurations - Update these paths according to your setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, r'C:\Users\mutha\Desktop\Crime_Detection\data')
MODEL_PATH = os.path.join(BASE_DIR, r'C:\Users\mutha\Desktop\Crime_Detection\models\crime_detection_model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, r'C:\Users\mutha\Desktop\Crime_Detection\uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, r'C:\Users\mutha\Desktop\Crime_Detection\app\static')

# Create necessary directories
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Labels for prediction
LABELS = ["crime", "non_crime"]

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def get_direct_stream_url(webpage_url):
    """Extract direct video stream URL from webpage URL."""
    try:
        # Try using streamlink first
        streams = streamlink.streams(webpage_url)
        if streams:
            logger.info("Stream URL found using streamlink")
            return streams['best'].url
    except Exception as e:
        logger.warning(f"Streamlink extraction failed: {str(e)}")
    
    try:
        # Parse the URL
        parsed_url = urlparse(webpage_url)
        
        # For EarthCam specifically
        if 'earthcam.com' in webpage_url:
            cam_id = webpage_url.split('cam=')[1].split('&')[0]
            api_url = f"https://api.earthcam.com/cameras/public/{cam_id}/stream"
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                stream_url = data.get('url')
                if stream_url:
                    logger.info("Stream URL found using EarthCam API")
                    return stream_url
    except Exception as e:
        logger.warning(f"EarthCam API extraction failed: {str(e)}")
    
    # If all else fails, return the original URL
    return webpage_url

def load_video_frames(video_path, max_frames=30):
    """Load and preprocess video frames."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (64, 64))
            frame = img_to_array(frame)
            frame = frame.astype("float") / 255.0
            frames.append(frame)
            count += 1

        cap.release()
        
        if len(frames) < max_frames:
            logger.warning(f"Only {len(frames)} frames found in video")
            
        return np.array(frames)
    except Exception as e:
        logger.error(f"Error loading video frames: {str(e)}")
        return np.array([])

def predict_crime(frames):
    
    if len(frames) < 30:
        return "Insufficient frames for prediction."
    
    try:
        frames = np.expand_dims(frames, axis=0)
        predictions = model.predict(frames)[0]
        average_prediction = np.mean(predictions)
        
        result = "No crime detected." if average_prediction >= 0.30 else "Crime detected!"
        logger.info(f"Prediction made: {result}")
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return "Error making prediction"
    

def process_live_stream():
    
    global live_stream, stop_live_stream, current_prediction, live_stream_url
    
    frames_buffer = []
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    
    while not stop_live_stream:
        try:
            if live_stream is None or not live_stream.isOpened():
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached")
                    break
                
                if live_stream_url:
                    logger.info("Attempting to reconnect to stream...")
                    live_stream = cv2.VideoCapture(live_stream_url)
                    reconnect_attempts += 1
                
                time.sleep(1)
                continue

            ret, frame = live_stream.read()
            if not ret:
                continue

            # Process frame for prediction
            processed_frame = cv2.resize(frame, (64, 64))
            processed_frame = img_to_array(processed_frame)
            processed_frame = processed_frame.astype("float") / 255.0
            
            frames_buffer.append(processed_frame)
            if len(frames_buffer) >= 30:
                prediction = predict_crime(np.array(frames_buffer))
                current_prediction = prediction
                frames_buffer = frames_buffer[15:]
                
            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Reset reconnect attempts on successful frame processing
            reconnect_attempts = 0

        except Exception as e:
            logger.error(f"Error in live stream processing: {str(e)}")
            time.sleep(1)
            continue

    if live_stream is not None and live_stream.isOpened():
        live_stream.release()

# Flask routes
@app.route('/')
def index():
   
    return render_template('frontend1.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    
    video = request.files['file']
    if not video.filename:
        return jsonify({'error': 'No file selected.'}), 400
    
    try:
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)
        
        frames = load_video_frames(video_path)
        if len(frames) == 0:
            return jsonify({'error': 'Failed to load video frames.'}), 400
        
        result = predict_crime(frames)
        
        # Clean up
        os.remove(video_path)
        
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': 'Error processing video.'}), 500

@app.route('/start_live_stream', methods=['POST'])
def start_live_stream():
    
    global live_stream, live_stream_url, stop_live_stream, current_prediction
    
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        if live_stream is not None and live_stream.isOpened():
            live_stream.release()
        
        stop_live_stream = False
        
        # Get direct stream URL
        direct_url = get_direct_stream_url(url)
        if not direct_url:
            return jsonify({'error': 'Could not find video stream URL'}), 400
            
        live_stream = cv2.VideoCapture(direct_url)
        live_stream_url = direct_url
        current_prediction = "No prediction yet"
        
        if not live_stream.isOpened():
            return jsonify({'error': 'Failed to connect to stream'}), 400
            
        logger.info("Stream started successfully")
        return jsonify({'message': 'Stream started successfully'})
    
    except Exception as e:
        logger.error(f"Error starting stream: {str(e)}")
        return jsonify({'error': f'Error starting stream: {str(e)}'}), 500

@app.route('/stop_live_stream', methods=['POST'])
def stop_live_stream():
    
    global live_stream, stop_live_stream
    
    stop_live_stream = True
    if live_stream is not None and live_stream.isOpened():
        live_stream.release()
    live_stream = None
    
    logger.info("Stream stopped successfully")
    return jsonify({'message': 'Stream stopped successfully'})

@app.route('/video_feed')
def video_feed():
    
    return Response(process_live_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_current_prediction')
def get_current_prediction():
   
    global current_prediction
    return jsonify({'prediction': current_prediction})

@app.route('/static/<path:filename>')
def serve_static(filename):
    
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)