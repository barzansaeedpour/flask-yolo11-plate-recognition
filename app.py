import os
import threading
import base64
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
import cv2
from werkzeug.utils import secure_filename
from my_yolo_11.yolov11 import plate_detection
from ultralytics import YOLO

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
MAX_THUMBS = 20

model_plate_detection = YOLO("./my_yolo_11/models/best_pose_detection.pt")
model_character_detection = YOLO("./my_yolo_11/models/character-detector.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen_frames(camera=False):
    """Read from the selected video_source and yield MJPEG frames."""
    global video_source
    if video_source is None:
        return

    # cap = cv2.VideoCapture(video_source)
    if camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)
            
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        detected_plate_txt, detected_chars_image, detected_plate_image = plate_detection(frame, model_plate_detection, model_character_detection, save_dir='', save=False)

        if len(detected_plate_txt) >= 8:
            processed = detected_plate_image
        else:
            processed = frame
            
        # encode processed thumbnail (grayscale)
        ret2, buffer2 = cv2.imencode('.jpg', processed)
        b64 = base64.b64encode(buffer2).decode('utf-8')
        
        # store thumbnail
        with thumb_lock:
            thumbnails.append(b64)
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
    return render_template('video.html', context={"camera":False})


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
