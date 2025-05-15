import argparse
from pathlib import Path
import cv2
import numpy as np
import time
import onnxruntime as ort
from flask import Flask, Response, render_template_string
from track import CustomTracker

from config import CLASSES_DET, COLORS
from utils import blob, letterbox, det_postprocess_nms

app = Flask(__name__)



def generate_frames(args):
    # Load ONNX model
    session = ort.InferenceSession(args.onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    inp_name = session.get_inputs()[0].name
    H, W = 480, 640

    # Open the camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera_id}")
        return

    frame_count = 0
    start_time = time.time()

    # Initialize custom tracker
    tracker = CustomTracker(max_age=5, min_hits=2, iou_threshold=0.3)

    while True:
        # Frame reading
        read_start = time.time()
        ret, bgr = cap.read()
        read_end = time.time()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        draw = bgr.copy()

        # Preprocessing
        pre_start = time.time()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        pre_end = time.time()

        # Inference
        inf_start = time.time()
        outputs = session.run(None, {inp_name: tensor})
        data = outputs[0]
        inf_end = time.time()

        # Postprocessing
        post_start = time.time()
        bboxes, scores, labels = det_postprocess_nms(data)
        post_end = time.time()

        # Handle detections
        if bboxes.size == 0:
            detections = np.empty((0, 6))
        else:
            bboxes -= dwdh
            bboxes /= ratio
            detections = np.column_stack((bboxes, scores, labels))

        # Update tracker
        track_start = time.time()
        tracks = tracker.update(detections)
        track_end = time.time()

        # 绘制边界框和跟踪ID
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            score, label = track[5], int(track[6])
            cls = CLASSES_DET[label] if label < len(CLASSES_DET) else "Unknown"
            color = COLORS[cls] if cls in COLORS else (255, 255, 255)

            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, f'ID: {track_id}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(draw, f'{cls}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(draw, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display timing information
        read_time = read_end - read_start
        pre_time = pre_end - pre_start
        inf_time = inf_end - inf_start
        post_time = post_end - post_start
        track_time = track_end - track_start

        cv2.putText(draw, f"Frame Read: {read_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Preprocess: {pre_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Inference: {inf_time:.3f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Postprocess: {post_time:.3f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Tracking: {track_time:.3f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', draw)
        frame = buffer.tobytes()

        # Generate MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print(f"Average FPS: {fps:.2f}")

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv8 Detection</title>
        </head>
        <body>
            <h1>YOLOv8 Real-time Detection</h1>
            <img src="{{ url_for('video_feed') }}" width="640">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    args = parse_args()
    return Response(generate_frames(args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-model', type=str, required=True, help='ONNX model file')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    app.run(host='0.0.0.0', port=5001, threaded=True)

# python3 run_camera_onnx.py --onnx-model yolov8n.onnx --camera-id 2
