import os
import threading
import base64
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
import cv2
from werkzeug.utils import secure_filename
from my_yolo_11.yolov11 import plate_detection
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret"  # required for flash()

# Static login credentials
USERNAME = 'admin'
PASSWORD = 'admin'

# where uploaded videos go
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# shared state
video_source = None
thumbnails = []
thumb_lock = threading.Lock()
MAX_THUMBS = 2000

model_plate_detection = YOLO("./my_yolo_11/models/best_pose_detection.pt")
model_character_detection = YOLO("./my_yolo_11/models/character-detector.pt")

the_last_image = []
the_last_plate_text = ''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_new_name():
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the datetime as desired (e.g., YYYYMMDD-HHMMSS)
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")

    # Create a unique filename
    filename = f"my_file_{formatted_datetime}"
    # print(f"Unique filename: {filename}")
    return filename

def save_frame(frame, frame_count, current_frame_number, fps=None):
    """Save the current frame as an image file."""
    if frame_count <= 0 or current_frame_number <= 0:
        return
    save_dir = f"{UPLOAD_FOLDER.replace('\\','/')}/frames"
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Save the frame as an image file
    filename = f"frame_{current_frame_number:04d}.jpg"
    filepath = f"{save_dir}/{filename}"
    
    cv2.imwrite(filepath, frame)
    print(f"Saved frame {current_frame_number} to {filepath}")
    if frame_count==current_frame_number:
        

        # Parameters
        frames_dir = save_dir          # Directory with your image frames
        output_path = f"{UPLOAD_FOLDER}/{get_new_name()}.mp4"
        if not fps:
            fps = 30  # Frames per second                       # Frames per second
        image_extension = '.jpg'        # or '.png', depending on your files

        # Get sorted list of frame filenames
        frame_files = sorted([
            f for f in os.listdir(frames_dir) if f.endswith(image_extension)
        ])

        # Read the first frame to get frame size
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width, _ = first_frame.shape
        size = (width, height)

        # Define the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'XVID' for .avi
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        # Write frames to video
        for filename in frame_files:
            frame_path = os.path.join(frames_dir, filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Skipping {filename}, couldn't read.")
                continue
            out.write(frame)

        out.release()
        print("Video saved as:", output_path)


        
def gen_frames(camera=False):
    """Read from the selected video_source and yield MJPEG frames."""
    global video_source
    global the_last_image
    global the_last_plate_text
    global frame_counter
    if video_source is None:
        return

    # cap = cv2.VideoCapture(video_source)
    if camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS:", fps)
        print("Total number of frames:", frame_count)
        
    
    current_frame_number = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        current_frame_number += 1
        print(f"Processing frame {current_frame_number}/{frame_count}")
        detected_plate_txt, frame_with_plate, detected_plate_image = plate_detection(frame, model_plate_detection, model_character_detection, save_dir='', save=False)

        if len(detected_plate_txt) >= 8:
            processed = detected_plate_image
            frame = frame_with_plate
            
            the_last_image = detected_plate_image
            the_last_plate_text = detected_plate_txt
        else:
            # processed = frame
            detected_plate_txt = ''
            if len(the_last_image)==0:
                processed = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                the_last_plate_text = ''
            else:
                processed = the_last_image
                detected_plate_txt = the_last_plate_text
        
        # if not camera: 
        #     save_frame(frame, frame_count, current_frame_number, fps)   
        # encode processed thumbnail (grayscale)
        ret2, buffer2 = cv2.imencode('.jpg', processed)
        b64 = base64.b64encode(buffer2).decode('utf-8')
        
        # store thumbnail
        with thumb_lock:
            thumbnails.append([b64, detected_plate_txt])
            if len(thumbnails) > MAX_THUMBS:
                thumbnails.pop(0)
                
        # encode original frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_jpeg = buffer.tobytes()

        # yield MJPEG chunk
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')

    cap.release()

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template('video.html', context={"camera":True})
    
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        
        # handle file upload
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(dest)
            # set as current source
            global video_source, thumbnails
            video_source = dest
            with thumb_lock:
                thumbnails = []
            return redirect(url_for('detect_video'))
        else:
            flash('Invalid file type')
            return redirect(request.url)
        
    # camera = request.args.get('camera', 'false').lower() == 'true'
    return render_template('upload_video.html')

@app.route('/detect_video')
def detect_video():
    # camera = request.args.get('camera', 'false').lower() == 'true'
    if video_source is None:
        flash('اول باید یک ویدیو آپلود کنید')
        return redirect(url_for('index'))
    global the_last_image
    the_last_image = []
    global the_last_plate_text
    the_last_plate_text = ''
    return render_template('video.html', context={"camera":False})

@app.route('/upload_image')
def upload_image():    
    return render_template('image.html')

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
        
        




@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(camera=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed')
def camera_feed():
    # camera = request.args.get('camera', 'false').lower() == 'true'
    return Response(gen_frames(camera=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/thumbnails')
def get_thumbnails():
    with thumb_lock:
        data = list(reversed(thumbnails))
    return jsonify(data)

########################################### Upload Video, Download Result
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_file, url_for

import uuid
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import os, uuid, threading, cv2

processing_progress = {}  # Dictionary to track progress by file_id

@app.route('/upload-video-progress', methods=['GET', 'POST'])
def upload_video_progress():
    if request.method == 'GET':
        return render_template('upload_process_video.html')

    if request.method == 'POST':
        video = request.files['video']
        filename = secure_filename(video.filename)
        file_id = str(uuid.uuid4())
        saved_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        result_path = os.path.join(RESULT_FOLDER, f"{file_id}_result.mp4")

        video.save(saved_path)
        processing_progress[file_id] = 0

        # Start processing in a background thread
        threading.Thread(target=process_video, args=(saved_path, result_path, file_id)).start()

        return jsonify({'file_id': file_id})

@app.route('/progress/<file_id>')
def check_progress(file_id):
    progress = processing_progress.get(file_id, 0)
    return jsonify({'progress': progress})


@app.route('/download/<filename>')
def download_processed_video(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

import cv2


def frame_process(frame):
    detected_plate_txt, frame_with_plate, detected_plate_image = plate_detection(frame, model_plate_detection, model_character_detection, save_dir='', save=False)

    if len(detected_plate_txt) >= 8:
        return frame_with_plate
    else:
        return frame
    
def process_video(input_path, output_path, file_id):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame_process(frame)
        if frame is not None:
            out.write(frame)

        count += 1
        processing_progress[file_id] = int((count / total_frames) * 100)

    cap.release()
    out.release()

    processing_progress[file_id] = 100  # Done



############################# Login Route #############################
from flask import Flask, render_template, request, redirect, url_for, session, flash

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == USERNAME and request.form['password'] == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('نام کاربری یا رمز عبور اشتباه است.')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        flash('ویدیو آپلود شد.')  # Just a placeholder
    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
