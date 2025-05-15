import argparse
from pathlib import Path
import cv2
import numpy as np
import time
from models.pycuda_api import TRTEngine
from config import CLASSES_DET, COLORS
from models.utils import blob, det_postprocess, letterbox
from sort.sort import Sort
from flask import Flask, Response  # 添加Flask用于视频流
import sys
import select
import threading

def is_key_pressed():
    return select.select([sys.stdin], [], [], 0)[0]  # 检测终端是否有输入

app = Flask(__name__)

# 全局变量存储最新帧和帧锁（简单实现）
latest_frame = None

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # 控制帧率

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main(args: argparse.Namespace) -> None:
    global latest_frame
    
    Engine = TRTEngine(args.engine)
    H, W = Engine.inp_info[0].shape[-2:]

    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_V4L2)  # 修改此行
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera_id}")
        return

    frame_count = 0
    start_time = time.time()
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # 在后台启动Flask
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, threaded=True))
    flask_thread.daemon = True
    flask_thread.start()
    print("Stream server started at http://0.0.0.0:5001/video_feed")

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
        data = Engine(tensor)
        inf_end = time.time()

        # Postprocessing
        post_start = time.time()
        bboxes, scores, labels = det_postprocess(data)
        post_end = time.time()

        # Handle the case where no bounding box is detected
        if bboxes.size == 0:
            detections = np.array([])
        else:
            bboxes -= dwdh
            bboxes /= ratio
            detections = np.column_stack((bboxes, scores))

        # Update SORT tracker
        track_start = time.time()
        tracks = tracker.update(detections if detections.size > 0 else np.empty((0, 5)))
        track_end = time.time()
        
        # Drawing bounding boxes and tracking IDs
        draw_start = time.time()
        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = map(int, track)
            # Match the detection index to the track (assuming order is preserved)
            if i < len(labels):
                cls_id = int(labels[i])
                score = scores[i] if i < len(scores) else 0
            else:
                cls_id = 0  # Default class if out of bounds
                score = 0

            cls = CLASSES_DET[cls_id]
            color = COLORS[cls]
            # print(f"Track ID: {track_id}, Class: {cls}, Score: {score:.2f}")

            # Draw bounding box
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            # Display track_id and score on separate lines
            cv2.putText(draw, f'ID: {track_id}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(draw, f'{cls}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        draw_end = time.time()

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
        draw_time = draw_end - draw_start

        cv2.putText(draw, f"Frame Read: {read_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Preprocess: {pre_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Inference: {inf_time:.3f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Postprocess: {post_time:.3f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Tracking: {track_time:.3f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw, f"Draw BBox: {draw_time:.3f}s", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        latest_frame = draw  # 更新全局帧

        # 监听键盘输入
        if is_key_pressed():
            key = sys.stdin.read(1)
            if key == 'q':
                print("Exit requested, stopping...")
                break

    cap.release()
    cv2.destroyAllWindows()
    del Engine  # 释放 TRTEngine
    import pycuda.driver as cuda
    cuda.Context.synchronize()  # 确保所有 CUDA 任务完成
    print(f"Average FPS: {fps:.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

# python3 run_camera_vedio_stream.py --engine yolov8n_fp16.engine --camera-id 0