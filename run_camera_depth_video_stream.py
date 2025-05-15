from typing import Sequence
import argparse
import logging
import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from flask import Flask, Response

logging.getLogger().setLevel(logging.INFO)

app = Flask(__name__)

class DepthEngine:
    """
    Real-time depth estimation using Depth Anything with TensorRT
    """
    def __init__(
        self,
        sensor_id,  # USB camera device index
        H,  # input tensor size height
        W,  # input tensor size width
        trt_engine_path,  # TensorRT engine path
    ):
        # Camera parameters
        self.sensor_id = sensor_id
        self.cap = None
        self.running = False
        
        # Input/output dimensions
        self.height, self.width = H , W  # inference size (height, width)
        self.display_width = 640  # display window width
        self.display_height = 480  # display window height

        # FPS calculation
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()

        # Load TensorRT engine
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(trt_engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        print(f"Engine loaded from {trt_engine_path}")

        # Allocate memory
        self.h_input = cuda.pagelocked_empty(trt.volume((1, 3, self.height, self.width)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume((1, 1, self.height, self.width)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.cuda_stream = cuda.Stream()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, image):
        """预处理图像，使用OpenCV和NumPy替代torchvision"""
        # 转换为浮点数并归一化到[0, 1]
        image = image.astype(np.float32) / 255.0

        # 调整大小到308x308
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        # 标准化（减均值，除标准差）
        image = (image - self.mean) / self.std

        # 转换为通道优先格式 (C, H, W)
        image = image.transpose(2, 0, 1)

        # 添加批次维度 (1, C, H, W)
        image = image[None]

        return image

    def postprocess(self, depth: np.ndarray) -> np.ndarray:
        """
        Postprocess depth map and display center depth value
        """
        # Reshape and resize depth map
        depth = np.reshape(depth, (self.height, self.width))
        depth = cv2.resize(depth, (self.display_width, self.display_height))

        # Normalize to 0-255
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)

        # Convert to color map
        depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

        # Get center point depth value (original scale, not normalized)
        center_x, center_y = self.display_width // 2, self.display_height // 2
        center_depth = depth[center_y, center_x]
        
        # Display center depth value on depth map
        depth_text = f"Center: {center_depth:.2f}"
        cv2.putText(
            depth_display,
            depth_text,
            (10, 30),  # Position at top-left of depth map
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Draw crosshair at center
        cv2.drawMarker(
            depth_display,
            (center_x, center_y),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )

        return depth_display

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Perform depth inference using TensorRT
        """
        image = self.preprocess(image)

        t0 = time.time()
        np.copyto(self.h_input, image.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()

        print(f"Inference time: {time.time() - t0:.4f}s")

        return self.postprocess(self.h_output)

    def generate_frames(self):
        self.cap = cv2.VideoCapture(self.sensor_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with sensor_id={self.sensor_id}")

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Resize frame to match display size
            frame = cv2.resize(frame, (self.display_width, self.display_height))

            # Depth inference
            depth = self.infer(frame)
            
            # Calculate FPS
            curr_time = time.time()
            self.frame_count += 1
            if curr_time - self.prev_time >= 1.0:
                self.fps = self.frame_count / (curr_time - self.prev_time)
                self.frame_count = 0
                self.prev_time = curr_time

            # Display FPS on the frame
            fps_text = f"FPS: {self.fps:.2f}"
            cv2.putText(
                frame, 
                fps_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )

            # Concatenate results
            results = np.concatenate((frame, depth), axis=1)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', results)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        self.cap.release()
    
    def stop(self):
        """
        Stop video capture
        """
        self.running = False
        if self.cap is not None:
            self.cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(depth_engine.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Depth Estimation Stream</title>
    </head>
    <body>
        <h1>Depth Estimation Stream</h1>
        <img src="/video_feed" width="1280" height="480">
    </body>
    </html>
    """

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--sensor_id', type=int, default=0, help='Camera device index')
    args.add_argument('--trt_engine_path', type=str, default="weights/depth_anything_vits14_308.trt", help='TensorRT engine path')
    args.add_argument('--H', type=int, default=308, help='Input Height')
    args.add_argument('--W', type=int, default=308, help='Input Width')
    args.add_argument('--port', type=int, default=5001, help='Flask server port')

    args = args.parse_args()
    depth_engine = DepthEngine(
        sensor_id=args.sensor_id,
        trt_engine_path=args.trt_engine_path,
        H=args.H,
        W=args.W
    )
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=False)
    except KeyboardInterrupt:
        depth_engine.stop()
        
# python3 run_camera_depth_video_stream.py --sensor_id 0 --trt_engine_path depth_anything_vits14_308.trt --H 308 --W 308
