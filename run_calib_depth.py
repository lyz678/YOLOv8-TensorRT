#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
import hiwonder
import apriltag
from flask import Flask, Response
import queue
import time
import threading
import yaml
import sys
import select
import tty
import termios
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Flask 应用
app = Flask(__name__)
latest_frame = None
image_queue = queue.Queue(maxsize=3)

# Flask 视频流生成
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            bgr_frame = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', bgr_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

# Flask 视频流端点
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 运行 Flask 服务器的线程
def run_flask():
    app.run(host='0.0.0.0', port=5001, threaded=False)

# 检查键盘输入（非阻塞）
def is_key_pressed():
    return select.select([sys.stdin], [], [], 0)[0]

# 恢复终端设置的全局函数
def restore_terminal():
    try:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
        rospy.loginfo("终端设置已恢复")
    except Exception as e:
        rospy.logerr(f"恢复终端设置失败: {e}")

ROS_NODE_NAME = 'depth_anything_calibration'

class DepthEngine:
    def __init__(self, input_size=(308, 308), trt_engine_path='depth_anything_vits14_308.trt'):
        """初始化Depth Anything引擎"""
        self.height, self.width = input_size
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(trt_engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        rospy.loginfo(f"Depth Anything引擎加载自 {trt_engine_path}")

        # 分配内存
        self.h_input = cuda.pagelocked_empty(trt.volume((1, 3, self.height, self.width)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume((1, 1, self.height, self.width)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.cuda_stream = cuda.Stream()

        # 预处理参数
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

    def infer(self, image, original_size):
        """执行深度推理并返回深度图"""
        image = self.preprocess(image)
        np.copyto(self.h_input, image.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()
        depth = np.reshape(self.h_output, (self.height, self.width))
        return depth

    def get_depth_at_point(self, image, center_x, center_y, original_size):
        """获取指定点的深度值"""
        depth = self.infer(image, original_size)
        scale_x = self.width / original_size[1]
        scale_y = self.height / original_size[0]
        depth_x = int(center_x * scale_x)
        depth_y = int(center_y * scale_y)
        depth_value = depth[depth_y, depth_x]
        return depth_value, depth
        
class DepthCalibrator:
    def __init__(self):
        self.image_sub = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrating = False
        self.output_yaml_path = '/home/lyz/.ros/camera_info/depth_anything_calibration.yaml'
        self.rvec = None
        self.tvec = None
        self.depth_scale = 1.0  # 深度缩放因子，初始值为1
        self.depth_engine = DepthEngine()
        self.center_depth = 0.0  # 图像中心像素的深度值

    def reset(self):
        self.image_sub = None
        self.calibrating = False
        self.rvec = None
        self.tvec = None
        self.depth_scale = 1.0
        self.center_depth = 0.0

    def init_servos(self):
        jetmax = hiwonder.JetMax()
        try:
            jetmax.set_servo(3, 500, 1)
            time.sleep(1)
            jetmax.set_servo(2, 500, 1)
            time.sleep(1)
            jetmax.set_servo(1, 125, 1)
            time.sleep(1)
            rospy.loginfo("舵机初始化完成")
        except Exception as e:
            rospy.logerr(f"无法设置伺服电机位置: {e}")
        return jetmax

    def load_camera_params(self):
        try:
            with open('/home/lyz/.ros/camera_info/head_camera.yaml', 'r') as f:
                camera_info = yaml.safe_load(f)
            self.camera_matrix = np.array(camera_info['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info['distortion_coefficients']['data'], dtype=np.float32).reshape(1, 5)
            rospy.loginfo("相机内参加载成功")
        except Exception as e:
            rospy.logerr(f"加载相机内参失败: {e}")
            self.camera_matrix = np.array([[502.7651238040608, 0, 332.5046103419863],
                                          [0, 503.8029542438583, 224.9520162283406],
                                          [0, 0, 1]], dtype=np.float32)
            self.dist_coeffs = np.array([-0.4577082271919898, 0.2017728618252299, 0.001257240477961667,
                                        0.002973749346017208, 0], dtype=np.float32).reshape(1, 5)

    def save_calibration_params(self):
        if self.rvec is None or self.tvec is None:
            rospy.logwarn("没有可用的标定参数，无法保存")
            return
        calibration_data = {
            'depth_params': {
                'K': self.camera_matrix.tolist(),
                'D': self.dist_coeffs.flatten().tolist(),
                'R': self.rvec.tolist(),
                'T': self.tvec.tolist(),
                'depth_scale': self.depth_scale
            }
        }
        try:
            os.makedirs(os.path.dirname(self.output_yaml_path), exist_ok=True)
            with open(self.output_yaml_path, 'w') as f:
                yaml.safe_dump(calibration_data, f, default_flow_style=None)
            rospy.loginfo(f"标定参数已保存到 {self.output_yaml_path}")
        except Exception as e:
            rospy.logerr(f"保存标定参数到 YAML 文件失败: {e}")

    def calibrate_depth(self, image, original_size):
        global latest_frame
        org_image = np.copy(image)

        # 获取图像中心深度
        center_x, center_y = original_size[1] // 2, original_size[0] // 2
        self.center_depth, depth_map = self.depth_engine.get_depth_at_point(image, center_x, center_y, original_size)

        # 显示标定状态和中心深度
        status_text = "Calibrating" if self.calibrating else "Uncalibrated"
        depth_text = f"Center Depth: {self.center_depth * self.depth_scale:.2f} mm"
        cv2.putText(org_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(org_image, depth_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.drawMarker(org_image, (center_x, center_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        if not self.calibrating:
            latest_frame = org_image.copy()
            return org_image

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        detections = detector.detect(gray)

        if detections:
            tag = detections[0]
            center = tag.center
            c_x, c_y = center[0], center[1]
            corners = np.array(tag.corners, dtype=np.float32)

            cv2.polylines(org_image, [np.int32(corners)], True, (0, 255, 0), 2)
            cv2.circle(org_image, (int(c_x), int(c_y)), 5, (0, 0, 255), -1)

            tag_size = 33.3  # AprilTag尺寸（毫米）
            obj_points = np.array([
                [tag_size/2, -tag_size/2, 0],
                [-tag_size/2, -tag_size/2, 0],
                [-tag_size/2, tag_size/2, 0],
                [tag_size/2, tag_size/2, 0]
            ], dtype=np.float32)

            ret, rvec, tvec = cv2.solvePnP(obj_points, corners, self.camera_matrix, self.dist_coeffs)
            if ret:
                self.rvec = rvec
                self.tvec = tvec
                # 计算真实距离（z轴平移分量，单位：毫米）
                real_distance = tvec[2][0]
                # 获取AprilTag中心点的深度值
                tag_depth, _ = self.depth_engine.get_depth_at_point(image, c_x, c_y, original_size)
                # 计算深度缩放因子
                if tag_depth > 0:
                    self.depth_scale = real_distance / tag_depth
                    rospy.loginfo(f"标定结果：真实距离={real_distance:.2f} mm, 深度值={tag_depth:.2f}, 缩放因子={self.depth_scale:.4f}")
                else:
                    rospy.logwarn("深度值无效，无法计算缩放因子")

                # 绘制坐标轴
                axis_length = tag_size / 2
                axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
                img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                img_points = img_points.astype(np.int32).reshape(-1, 2)

                cv2.line(org_image, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)
                cv2.line(org_image, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)
                cv2.line(org_image, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)

        latest_frame = org_image.copy()
        return org_image

def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

def image_proc(original_size):
    try:
        ros_image = image_queue.get(block=True, timeout=0.1)
    except queue.Empty:
        return
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    image = calibrator.calibrate_depth(image, original_size)
    rgb_image = image.tobytes()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)

def keyboard_control(jetmax, calibrator):
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        rospy.loginfo("已设置终端为非缓冲模式")
        while not rospy.is_shutdown():
            if is_key_pressed():
                key = sys.stdin.read(1)
                if key == 'b':
                    calibrator.calibrating = True
                    rospy.loginfo("开始 Depth Anything 标定")
                elif key == 'c':
                    calibrator.calibrating = False
                    rospy.loginfo("停止 Depth Anything 标定")
                elif key == 'd':
                    try:
                        x, y, z = jetmax.position
                        jetmax.set_position((x, y, 100), 1)
                        rospy.loginfo(f"移动到位置: ({x}, {y}, 100)")
                    except Exception as e:
                        rospy.logerr(f"移动舵机失败: {e}")
                elif key == 'i':
                    try:
                        calibrator.init_servos()
                        rospy.loginfo("舵机已回到初始位置")
                    except Exception as e:
                        rospy.logerr(f"返回初始位置失败: {e}")
                elif key == 's':
                    calibrator.save_calibration_params()
                elif key == 'q':
                    rospy.loginfo("检测到 'q' 键，退出程序")
                    if calibrator.image_sub is not None:
                        calibrator.image_sub.unregister()
                        rospy.loginfo("已取消图像订阅")
                    while not image_queue.empty():
                        try:
                            image_queue.get_nowait()
                        except queue.Empty:
                            break
                    jetmax.go_home(1)
                    rospy.signal_shutdown("用户请求退出")
                    break
            time.sleep(0.01)
    except Exception as e:
        rospy.logerr(f"键盘控制线程异常: {e}")
    finally:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            rospy.loginfo("键盘控制线程：终端设置已恢复")
        except Exception as e:
            rospy.logerr(f"键盘控制线程恢复终端设置失败: {e}")

# 保存初始终端设置
old_terminal_settings = termios.tcgetattr(sys.stdin)

def main():
    global calibrator, image_pub
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    
    calibrator = DepthCalibrator()
    calibrator.reset()
    jetmax = calibrator.init_servos()
    calibrator.load_camera_params()
    
    calibrator.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    rospy.loginfo("流服务器启动于 http://0.0.0.0:5001/video_feed")

    keyboard_thread = threading.Thread(target=keyboard_control, args=(jetmax, calibrator))
    keyboard_thread.daemon = True
    keyboard_thread.start()
    rospy.loginfo("键盘控制已启用：按 'b' 开始标定，按 'c' 停止标定，按 'd' 移动到 z=100，按 'i' 回到初始位置，按 's' 保存参数，按 'q' 退出程序")

    hiwonder.buzzer.on()
    rospy.sleep(0.2)
    hiwonder.buzzer.off()

    try:
        # 获取图像尺寸
        while not rospy.is_shutdown():
            try:
                ros_image = image_queue.get(block=True, timeout=1.0)
                original_size = (ros_image.height, ros_image.width)
                image_queue.put_nowait(ros_image)  # 将图像放回队列
                break
            except queue.Empty:
                continue

        while not rospy.is_shutdown():
            image_proc(original_size)
            rospy.sleep(0.01)
    except KeyboardInterrupt:
        rospy.loginfo("收到 Ctrl+C，退出程序")
    except Exception as e:
        rospy.logerr(f"主线程异常: {e}")
    finally:
        if calibrator.image_sub is not None:
            calibrator.image_sub.unregister()
            rospy.loginfo("主线程：已取消图像订阅")
        jetmax.go_home(1)
        restore_terminal()
        rospy.loginfo("程序退出")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        rospy.logerr(f"程序异常: {e}")
    finally:
        restore_terminal()