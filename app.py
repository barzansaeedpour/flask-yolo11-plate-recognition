import os
import threading
import base64
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret"  # required for flash()

# where uploaded videos go
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# shared state
video_source = None
thumbnails = []
thumb_lock = threading.Lock()
MAX_THUMBS = 80

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen_frames():
    """Read from the selected video_source and yield MJPEG frames."""
    global video_source
    if video_source is None:
        return

    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Simple processing: Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = gray

        # encode original frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_jpeg = buffer.tobytes()

        # encode processed thumbnail (grayscale)
        ret2, buffer2 = cv2.imencode('.jpg', processed)
        b64 = base64.b64encode(buffer2).decode('utf-8')

        # store thumbnail
        with thumb_lock:
            thumbnails.append(b64)
            if len(thumbnails) > MAX_THUMBS:
                thumbnails.pop(0)

        # yield MJPEG chunk
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')

    cap.release()


@app.route('/', methods=['GET', 'POST'])
def index():
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

    return render_template('index.html')


@app.route('/detect_video')
def detect_video():
    if video_source is None:
        flash('اول باید یک ویدیو آپلود کنید')
        return redirect(url_for('index'))
    return render_template('video.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/thumbnails')
def get_thumbnails():
    with thumb_lock:
        data = list(reversed(thumbnails))
    return jsonify(data)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
