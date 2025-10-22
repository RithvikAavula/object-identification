import os
import cv2
from flask import Flask, Response, request, jsonify, send_from_directory, render_template_string
from ultralytics import YOLO
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = YOLO("yolov8m.pt")  # YOLOv8 medium

UPLOAD_FOLDER = "static/uploads"
SAVED_FRAMES = "static/saved_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVED_FRAMES, exist_ok=True)

confidence_threshold = 0.3
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------------- Live Detection State ----------------
camera = None
camera_active = False
last_metrics = {"fps": 0, "confidence": confidence_threshold, "object_count": 0, "detections": {}}

# ---------------- Frontend HTML ----------------
FRONTEND_HTML = """ 
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Object Detection Dashboard</title>
<style>
:root {--primary:#6366f1;--dark-light:#1e293b;--text:#f1f5f9;--border:#94a3b81a;}
body{font-family:'Inter',sans-serif;background:#0f172a;color:var(--text);margin:0;padding:0;}
h1{margin:0;color:var(--primary);}
button{cursor:pointer;transition:0.2s;outline:none;}
button:hover{opacity:0.85;transform:scale(1.05);}
.dashboard{display:grid;grid-template-columns:2fr 1fr;gap:1rem;padding:1rem;}
@media(max-width:768px){.dashboard{grid-template-columns:1fr;}}
.video-container{position:relative;overflow:hidden;border-radius:10px;}
#video_feed{width:100%;border-radius:10px;transition:0.3s;}
.controls,.metrics,.saved-frames{margin-top:1rem;background:var(--dark-light);padding:1rem;border-radius:10px;transition:0.3s;}
.controls button{margin:0.25rem;padding:0.5rem 1rem;border:none;border-radius:5px;background:var(--primary);color:white;}
input[type=range]{width:100%;}
.saved-frames img,.uploaded-frames img{width:100px;margin:0.25rem;border-radius:5px;cursor:pointer;border:2px solid var(--border);transition:0.2s;}
.saved-frames img:hover,.uploaded-frames img:hover{transform:scale(1.1);}
.object-list{margin-top:0.5rem;max-height:150px;overflow-y:auto;}
.modal { display:none;position:fixed;z-index:1000;left:0;top:0;width:100%;height:100%; background: rgba(0,0,0,0.9); justify-content:center;align-items:center;}
.modal img {max-width:90%; max-height:90%; border-radius:10px;}
.modal.active {display:flex;}
</style>
</head>
<body>
<div class="header">
    <h1>AI Object Detection</h1>
    <div class="status">
        Confidence: <span id="conf_val">0.3</span>
    </div>
</div>

<div class="dashboard">
    <div>
        <div class="video-container">
            <img id="video_feed">
        </div>
        <div class="controls">
            <button onclick="startCamera()">▶ Start</button>
            <button onclick="stopCamera()">⏹ Stop</button>
            <button onclick="captureFrame()">💾 Detect & Save Frame</button>
            <label>Confidence:</label>
            <input type="range" min="0.1" max="1" step="0.05" id="confidence" value="0.3" oninput="updateConfidence(this.value)">
            <br><br>
            <form id="uploadForm">
                <input type="file" name="image" accept="image/*">
                <button type="submit">Upload & Detect</button>
            </form>
        </div>
        <div class="metrics">
            <h3>Object Detection History</h3>
            <ul id="object_list" class="object-list"></ul>
        </div>
    </div>

    <div class="saved-frames">
        <h3>Saved Camera Frames</h3>
        <div id="saved_container" class="saved-frames"></div>
        <h3>Uploaded Images</h3>
        <div id="upload_container" class="uploaded-frames"></div>
    </div>
</div>

<div class="modal" id="modal">
    <img id="modal_img">
</div>

<script>
let metricsInterval;

async function startCamera(){ 
    await fetch('/start_camera');
    document.getElementById('video_feed').src='/video_feed';
    metricsInterval = setInterval(loadMetrics, 1000);
}
async function stopCamera(){ 
    await fetch('/stop_camera'); 
    document.getElementById('video_feed').src='';
    clearInterval(metricsInterval);
}
async function captureFrame(){ 
    const res=await fetch('/save-frame',{method:'POST'}); 
    const data=await res.json();
    if(data.status=='saved'){ loadSavedFrames(); alert('Saved frame: '+data.filename); }
}
async function updateConfidence(val){
    document.getElementById('conf_val').innerText=val;
    await fetch('/confidence',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confidence:val})});
}
async function loadSavedFrames(){
    const res=await fetch('/saved-frames');
    const data=await res.json();
    const container=document.getElementById('saved_container'); container.innerHTML='';
    data.frames.forEach(f=>{ let img=document.createElement('img'); img.src='/saved_frames/'+f; img.onclick=()=>viewModal(img.src); container.appendChild(img); });
}
async function loadUploadedImages(){
    const res=await fetch('/uploaded-frames');
    const data=await res.json();
    const container=document.getElementById('upload_container'); container.innerHTML='';
    data.frames.forEach(f=>{ let img=document.createElement('img'); img.src='/uploads/'+f; img.onclick=()=>viewModal(img.src); container.appendChild(img); });
}
async function loadMetrics(){
    const res = await fetch('/metrics');
    const data = await res.json();
    const list = document.getElementById('object_list');
    list.innerHTML = '';
    for(let obj in data.detections){
        let li = document.createElement('li');
        li.textContent = `${obj}: ${data.detections[obj]}`;
        list.appendChild(li);
    }
}
function viewModal(src){ const modal=document.getElementById('modal'); document.getElementById('modal_img').src=src; modal.classList.add('active'); }
document.getElementById('modal').onclick=function(e){if(e.target.id==='modal') this.classList.remove('active');}
document.getElementById('uploadForm').onsubmit=async (e)=>{
    e.preventDefault(); 
    const formData=new FormData(e.target);
    const res=await fetch('/upload-detect',{method:'POST',body:formData});
    const data=await res.json();
    if(data.status=='ok'){ loadUploadedImages(); alert('Detection done: '+data.filename); }
}
loadSavedFrames(); loadUploadedImages();
</script>
</body>
</html>
"""

# ---------------- Backend ----------------

@app.route('/')
def index():
    return render_template_string(FRONTEND_HTML)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Camera Control ----------
@app.route('/start_camera')
def start_camera():
    global camera, camera_active
    if camera is None:
        camera = cv2.VideoCapture(0)
    camera_active = True
    return jsonify({"status":"started"})

@app.route('/video_feed')
def video_feed():
    global camera, camera_active, last_metrics
    if camera is None or not camera_active:
        return "Camera not active", 400

    def generate():
        global camera_active, last_metrics
        prev_time = cv2.getTickCount()
        while camera_active:
            ret, frame = camera.read()
            if not ret: break

            results = model.predict(frame, conf=confidence_threshold, stream=True, verbose=False)
            detections = {}
            for r in results:
                frame = r.plot()
                for cls_id in r.boxes.cls:
                    cls_name = model.names[int(cls_id)]
                    detections[cls_name] = detections.get(cls_name,0)+1

            curr_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time
            last_metrics.update({
                "fps": round(fps,2),
                "confidence": confidence_threshold,
                "object_count": sum(detections.values()),
                "detections": detections
            })

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    return jsonify(last_metrics)

# ---------- Save Live Frame ----------
@app.route('/save-frame', methods=['POST'])
def save_frame():
    global camera
    if camera is None:
        return jsonify({"status":"error","msg":"Camera not active"})
    ret, frame = camera.read()
    if not ret:
        return jsonify({"status":"error","msg":"Failed to read frame"})
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg"
    path = os.path.join(SAVED_FRAMES, filename)
    results = model.predict(frame, conf=confidence_threshold, stream=True, verbose=False)
    for r in results: frame = r.plot()
    cv2.imwrite(path, frame)
    return jsonify({"status":"saved","filename":filename})

# ---------- Upload & Detect ----------
@app.route('/upload-detect', methods=['POST'])
def upload_detect():
    if 'image' not in request.files:
        return jsonify({"error":"No file uploaded"}),400
    file = request.files['image']
    if file.filename=='' or not allowed_file(file.filename):
        return jsonify({"error":"Invalid file"}),400
    filename = secure_filename(datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg")
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    img = cv2.imread(path)
    results = model.predict(img, conf=confidence_threshold, stream=True, verbose=False)
    for r in results: img = r.plot()
    cv2.imwrite(path,img)
    return jsonify({"status":"ok","filename":filename})

# ---------- Saved / Uploaded Frames ----------
@app.route('/saved-frames')
def get_saved_frames():
    return jsonify({"frames": os.listdir(SAVED_FRAMES)})

@app.route('/uploaded-frames')
def get_uploaded_frames():
    return jsonify({"frames": os.listdir(UPLOAD_FOLDER)})

@app.route("/saved_frames/<path:filename>")
def serve_saved_frame(filename):
    return send_from_directory(SAVED_FRAMES, filename)

@app.route("/uploads/<path:filename>")
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/confidence', methods=['POST'])
def set_confidence():
    global confidence_threshold
    data = request.get_json()
    confidence_threshold = float(data.get('confidence',0.3))
    return jsonify({"status":"ok","confidence":confidence_threshold})

@app.route('/stop_camera')
def stop_camera():
    global camera_active, camera
    camera_active = False
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status":"stopped"})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
