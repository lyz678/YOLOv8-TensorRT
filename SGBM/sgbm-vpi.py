from flask import Flask, Response, render_template, jsonify,  request
import cv2
import numpy as np
import time
import math
import vpi
from threading import Lock

app = Flask(__name__)

# Global variables for thread-safe access
global_state = {
    'threeD': None,
    'timing_text': '',
    'lock': Lock()
}

# -----------------------------------双目相机的基本参数---------------------------------------------------------
left_camera_matrix = np.array([[516.5066236, -1.444673028, 320.2950423], [0, 516.5816117, 270.7881873], [0., 0., 1.]])
right_camera_matrix = np.array([[511.8428182, 1.295112628, 317.310253], [0, 513.0748795, 269.5885026], [0., 0., 1.]])

# 畸变系数
left_distortion = np.array([[-0.046645194, 0.077595167, 0.012476819, -0.000711358, 0]])
right_distortion = np.array([[-0.061588946, 0.122384376, 0.011081232, -0.000750439, 0]])

# 旋转矩阵和平移矩阵
R = np.array([[0.999911333, -0.004351508, 0.012585312],
              [0.004184066, 0.999902792, 0.013300386],
              [-0.012641965, -0.013246549, 0.999832341]])
T = np.array([-120.3559901, -0.188953775, -0.662073075])

size = (640, 480)

# 校正映射
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

# 参数设置
height, width = 480, 640  # 图像尺寸
window = 5  # 匹配 blockSize=5
maxDisparity = 64  # 匹配 numDisparities=64
minDisparity = 1  # 匹配 minDisparity=1

# 初始化 VPI 流
stream = vpi.Stream()

# Video processing and streaming
def generate_frames():
    global global_state
    capture = cv2.VideoCapture("./car.avi")
    fps = 0.0
    ret, frame = capture.read()
    while ret:
        t1 = time.time()

        ret, frame = capture.read()
        t_read = time.time()
        if not ret:
            break

        # 切割为左右两张图片
        frame1 = frame[0:480, 0:640]
        frame2 = frame[0:480, 640:1280]
        t_cut = time.time()

        # 转换为灰度图
        imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        t_gray = time.time()

        # 重映射
        imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
        t_remap = time.time()

        # 转换为 BGR 格式
        imageL = cv2.cvtColor(imgL_rectified, cv2.COLOR_GRAY2BGR)
        t_color = time.time()

        # 使用 VPI 计算视差
        with stream, vpi.Backend.CUDA:
            left = vpi.asimage(imgL_rectified)
            right = vpi.asimage(imgR_rectified)
            disparity = vpi.Image((width, height), vpi.Format.U16)
            confidenceMap = vpi.Image((width, height), vpi.Format.U16)

            disparity = vpi.stereodisp(left, right, out_confmap=confidenceMap, window=window, maxdisp=maxDisparity)

            disparity_np = disparity.cpu().astype(np.int16)
            disparity_np += minDisparity * 16
        t_sgbm = time.time()

        # 归一化生成深度图（灰度图）
        disp = cv2.normalize(disparity_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 生成深度图（颜色图）
        dis_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        t_norm = time.time()

        # 计算三维坐标
        threeD = cv2.reprojectImageTo3D(disparity_np, Q, handleMissingValues=True)
        threeD = threeD * 16
        t_3d = time.time()

        # 计算帧率和耗时
        fps = (fps + (1. / (time.time() - t1))) / 2
        time_read = t_read - t1
        time_cut = t_cut - t_read
        time_gray = t_gray - t_cut
        time_remap = t_remap - t_gray
        time_color = t_color - t_remap
        time_sgbm = t_sgbm - t_color
        time_norm = t_norm - t_sgbm
        time_3d = t_3d - t_norm
        total_time = time.time() - t1

        # 格式化耗时信息
        text = (f"FPS: {fps:.2f}\nRead: {time_read:.4f}s\nCut: {time_cut:.4f}s\nGray: {time_gray:.4f}s\n"
                f"Remap: {time_remap:.4f}s\nColor: {time_color:.4f}s\nSGBM: {time_sgbm:.4f}s\n"
                f"Norm: {time_norm:.4f}s\n3D: {time_3d:.4f}s\nTotal: {total_time:.4f}s")

        # 更新全局状态
        with global_state['lock']:
            global_state['threeD'] = threeD.copy()
            global_state['timing_text'] = text

        # 编码帧为 JPEG
        ret1, left_jpeg = cv2.imencode('.jpg', frame1)
        ret2, disp_jpeg = cv2.imencode('.jpg', disp)
        ret3, depth_jpeg = cv2.imencode('.jpg', dis_color)

        if not (ret1 and ret2 and ret3):
            continue

        # 生成视频流帧
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + left_jpeg.tobytes() + b'\r\n',  # Left image
               b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + disp_jpeg.tobytes() + b'\r\n',   # Grayscale disparity
               b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + depth_jpeg.tobytes() + b'\r\n')  # Colored depth

    capture.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_left')
def video_feed_left():
    def gen():
        for left_frame, _, _ in generate_frames():
            yield left_frame
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_disp')
def video_feed_disp():
    def gen():
        for _, disp_frame, _ in generate_frames():
            yield disp_frame
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    def gen():
        for _, _, depth_frame in generate_frames():
            yield depth_frame
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    global global_state
    data = request.json
    x, y = int(data['x']), int(data['y'])
    with global_state['lock']:
        if global_state['threeD'] is None or y >= height or x >= width:
            return jsonify({'error': 'Coordinates not available or out of bounds'})
        threeD = global_state['threeD']
        coords = threeD[y, x]
        distance = math.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2) / 1000.0
        return jsonify({
            'x': x,
            'y': y,
            'world_x': coords[0] / 1000.0,
            'world_y': coords[1] / 1000.0,
            'world_z': coords[2] / 1000.0,
            'distance': distance
        })

@app.route('/get_timing')
def get_timing():
    global global_state
    with global_state['lock']:
        return jsonify({'timing': global_state['timing_text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)