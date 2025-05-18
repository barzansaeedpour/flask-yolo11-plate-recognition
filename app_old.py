from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
from my_yolo_11.yolov11 import plate_detection
import numpy as np
from ultralytics import YOLO
import base64
import cv2

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the YOLO11 model (replace this line with the appropriate YOLO11 loading code)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='./my_yolo_11/models/plate-detector.pt', force_reload=True).autoshape()  # Change to YOLO11 equivalent
# model = torch.load('./my_yolo_11/models/plate-detector.pt')
# اگر کش موجود نیست، مدل‌ها را از فایل بارگذاری کنید
# model_plate_detection = YOLO('./my_yolo_11/models/plate-detector.pt')
model_plate_detection = YOLO('./my_yolo_11/models/best_pose_detection.pt')
model_character_detection = YOLO('./my_yolo_11/models/character-detector.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read and process the uploaded image
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    frame = np.array(image)
    # frame = frame[:,:,[2,1,0]]
    frame = frame[:,:,[0,1,2]]
    
    detected_plate_txt, detected_plate_image, detected_chars_image = plate_detection(frame, model_plate_detection, model_character_detection, save_dir='', save=False)

    if detected_plate_txt!='':
        cv2.imwrite('./detected_plate_image.png', detected_plate_image)
        cv2.imwrite('./detected_chars_image.png', detected_chars_image)
        # Save images to in-memory files to send back as response 
        detected_chars_img_io = io.BytesIO() 
        Image.fromarray(detected_chars_image).save(detected_chars_img_io, 'PNG') 
        detected_chars_img_io.seek(0) 
        detected_plate_img_io = io.BytesIO() 
        Image.fromarray(detected_plate_image).save(detected_plate_img_io, 'PNG') 
        detected_plate_img_io.seek(0)
    
        # Perform detection
        # results = model(image)
        # detections = results.pandas().xyxy[0].to_dict(orient='records')  # List of detection dictionaries

        # # return jsonify({'detections': "detected_plate_txt"})
        # detected_plate_txt = detected_plate_txt[:-2] + ' ' + detected_plate_txt[-2:]
        return jsonify({ 'detected_plate_txt': detected_plate_txt, 'detected_chars_image': 'data:image/png;base64,' + base64.b64encode(detected_chars_img_io.getvalue()).decode(), 'detected_plate_image': 'data:image/png;base64,' + base64.b64encode(detected_plate_img_io.getvalue()).decode() })
    else:
        return jsonify({ 'detected_plate_txt': 'عدم تشخیص', 'detected_chars_image': '', 'detected_plate_image': '' })
        
        
        
###################### Detect from video ######################

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import base64


# shared list of processed-frame thumbnails (base64 JPEG)
thumbnails = []
thumb_lock = threading.Lock()
MAX_THUMBS = 20

def gen_frames():
    """Camera capture + processing generator yielding MJPEG frames."""
    cap = cv2.VideoCapture("G:/16_01_R_20250425150000.avi")  # or your RTSP URL
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Simple processing: Canny edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # encode original frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_jpeg = buffer.tobytes()

        # encode processed frame thumbnail to base64
        ret2, buffer2 = cv2.imencode('.jpg', edges)
        thumb_b64 = base64.b64encode(buffer2).decode('utf-8')

        # store thumbnail (thread-safe)
        with thumb_lock:
            thumbnails.append(thumb_b64)
            if len(thumbnails) > MAX_THUMBS:
                thumbnails.pop(0)

        # yield MJPEG chunk
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')

    cap.release()


@app.route('/detect_video')
def detect_video():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thumbnails')
def get_thumbnails():
    with thumb_lock:
        # return newest-first
        data = list(reversed(thumbnails))
    return jsonify(data)




        
if __name__ == '__main__':
    app.run(debug=True)
