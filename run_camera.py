import argparse
from pathlib import Path
import cv2
import numpy as np
import time
from models.pycuda_api import TRTEngine
from config import CLASSES_DET, COLORS
from models.utils import blob, det_postprocess, letterbox
from sort.sort import Sort  # Import SORT algorithm

def main(args: argparse.Namespace) -> None:
    Engine = TRTEngine(args.engine)
    H, W = Engine.inp_info[0].shape[-2:]

    # Open the camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera_id}")
        return

    frame_count = 0
    start_time = time.time()

    # Initialize SORT tracker with tuned parameters
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # Adjust these to control ID reuse

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

        if args.show:
            cv2.imshow('result', draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {fps:.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID')
    parser.add_argument('--show', action='store_true', help='Show the detection results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

# python3 run_camera.py --engine yolov8n_fp16.engine --camera-id 0 --show
