import argparse
from pathlib import Path
import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from flask import Flask, Response, render_template_string, request
from api_infer import SnpeContext, Runtime, PerfProfile, LogLevel  # SNPE推理模块
from config import CLASSES_DET, COLORS
from utils import letterbox, det_postprocess_nms
from track import CustomTracker

app = Flask(__name__)

# 原始参数（1080*1920）
CAMERA_MATRIX = np.array([
    [1.54260292e+03, 0.00000000e+00, 9.79284155e+02],
    [0.00000000e+00, 1.54339791e+03, 6.57131835e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

DIST_COEFFS = np.array([-0.37060393, 0.04148584, -0.00094008, -0.00232051, 0.05975394])
# cv2相机默认图像尺寸480*640
scale_x = 640 / 1920
scale_y = 480 / 1080

# 适配（480*640)
NEW_CAMERA_MATRIX = CAMERA_MATRIX.copy()
NEW_CAMERA_MATRIX[0, 0] *= scale_x  # fx
NEW_CAMERA_MATRIX[1, 1] *= scale_y  # fy
NEW_CAMERA_MATRIX[0, 2] *= scale_x  # cx
NEW_CAMERA_MATRIX[1, 2] *= scale_y  # cy

# 全局变量控制畸变校正状态
use_undistort = False


def generate_frames(args):
    """生成视频帧并进行目标检测与跟踪，支持畸变校正切换"""
    global use_undistort
    # SNPE初始化
    runtime_map = {'cpu': Runtime.CPU, 'gpu': Runtime.GPU, 'dsp': Runtime.DSP}
    selected_runtime = runtime_map[args.runtime.lower()]
    snpe_ort = SnpeContext(args.dlc_model, [], selected_runtime, PerfProfile.BALANCED, LogLevel.INFO)
    assert snpe_ort.Initialize() == 0, "SNPE初始化失败"

    # 推理尺寸
    H, W = args.H, args.W

    # 打开摄像头
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera_id}")
        return

    frame_count = 0
    start_time = time.time()
    tracker = CustomTracker(max_age=5, min_hits=2, iou_threshold=0.3)

    try:
        while True:
            read_start = time.time()
            ret, bgr = cap.read()
            read_end = time.time()
            if not ret:
                print("无法获取帧")
                break

            frame_count += 1
            draw = bgr.copy()

            # 畸变校正（按需）
            pre_start = time.time()
            if use_undistort:
                bgr = cv2.undistort(bgr, NEW_CAMERA_MATRIX, DIST_COEFFS)
                draw = bgr.copy()

            # 预处理
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            dwdh = np.array(dwdh * 2, dtype=np.float32)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(rgb).astype(np.float32) / 255
            pre_end = time.time()

            # 推理
            inf_start = time.time()
            input_feed = {"images": tensor}  # netron 查看模型输入输出名称
            output_names = []
            outputs = snpe_ort.Execute(output_names, input_feed)
            inf_end = time.time()

            # 后处理
            post_start = time.time()
            data = outputs['outputs']
            data = np.array(data)
            data = data.reshape(1, -1, 84)
            bboxes, scores, labels = det_postprocess_nms(data)
            post_end = time.time()

            # 处理检测结果
            if bboxes.size == 0:
                detections = np.empty((0, 6))
            else:
                bboxes -= dwdh
                bboxes /= ratio
                detections = np.column_stack((bboxes, scores, labels))

            # 更新跟踪器
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

            # 计算并显示FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(draw, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # 显示时间信息
            read_time = read_end - read_start
            pre_time = pre_end - pre_start
            inf_time = inf_end - inf_start
            post_time = post_end - post_start
            track_time = track_end - track_start
            # cv2.putText(draw, f"Frame Read: {read_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(draw, f"Preprocess: {pre_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f"Inference: {inf_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(draw, f"Postprocess: {post_time:.3f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(draw, f"Tracking: {track_time:.3f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示畸变校正状态
            status = "Undistorted" if use_undistort else "Distorted"
            cv2.putText(draw, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 编码帧为JPEG
            ret, buffer = cv2.imencode('.jpg', draw)
            frame = buffer.tobytes()

            # 生成MJPEG流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        cap.release()
        print(f"平均FPS: {fps:.2f}")


@app.route('/control', methods=['POST'])
def control():
    """处理畸变校正控制请求"""
    global use_undistort
    action = request.form.get('action')
    print(f"Received action: {action}")  # 调试日志
    if action == 'undistort':
        use_undistort = True
    elif action == 'distort':
        use_undistort = False
    return '', 204


@app.route('/')
def index():
    """渲染HTML页面显示视频流"""
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv8 实时检测</title>
            <script>
                // 监听键盘事件以控制畸变校正
                document.addEventListener('keydown', function(event) {
                    if (event.key === 'u') {
                        fetch('/control', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: 'action=undistort'
                        }).then(response => console.log('Undistort enabled'));
                    } else if (event.key === 'd') {
                        fetch('/control', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: 'action=distort'
                        }).then(response => console.log('Distort enabled'));
                    } else if (event.key === 'q') {
                        alert('退出功能暂不支持，请关闭浏览器窗口或停止服务器。');
                    }
                });
            </script>
        </head>
        <body>
            <h1>YOLOv8 实时检测与跟踪</h1>
            <img src="{{ url_for('video_feed') }}">
            <p>按 'u' 启用畸变校正，按 'd' 禁用畸变校正，按 'q' 退出</p>
            <!-- 添加按钮控制（可选） -->
            <button onclick="sendAction('undistort')">启用畸变校正</button>
            <button onclick="sendAction('distort')">禁用畸变校正</button>
            <script>
                function sendAction(action) {
                    fetch('/control', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: `action=${action}`
                    }).then(response => console.log(`${action} sent`));
                }
            </script>
        </body>
        </html>
    ''')


@app.route('/video_feed')
def video_feed():
    """提供视频流"""
    args = parse_args()
    return Response(generate_frames(args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 实时目标检测与跟踪")
    parser.add_argument('--dlc-model', type=str, required=True, help='DLC模型文件路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--runtime', type=str, choices=['cpu', 'gpu', 'dsp'], default='cpu',
                        help='运行时后端 (cpu, gpu, 或 dsp)')
    parser.add_argument('--H', type=int, default=480, help='推理尺寸H')
    parser.add_argument('--W', type=int, default=640, help='推理尺寸W')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run(host='0.0.0.0', port=5001, threaded=True)


# python3 run_camera_dlc.py --dlc-model yolov8_int8_480_640_Q.dlc --camera-id 2 --runtime dsp --H 480 --W 640
# python3 run_camera_dlc.py --dlc-model yolov8_int8_320_320_Q.dlc --camera-id 2 --runtime dsp --H 320 --W 320


