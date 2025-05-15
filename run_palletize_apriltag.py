#!/usr/bin/env python3
import math
import rospy
import time
import queue
import threading
import numpy as np
from sensor_msgs.msg import Image
import cv2
import hiwonder
import apriltag
from flask import Flask, Response
import yaml

# 全局变量
ROS_NODE_NAME = 'suck_apriltag'
latest_frame = None
image_queue = queue.Queue(maxsize=3)
shutdown_event = threading.Event()

# Flask 应用
app = Flask(__name__)

# Flask 视频流生成
def generate_frames():
    global latest_frame
    while not shutdown_event.is_set():
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

class CameraHandler:
    def __init__(self, calib_file='/home/lyz/.ros/camera_info/apriltag_calibration_default.yaml'):
        """初始化相机处理器，加载校准参数"""
        self.K = None
        self.D = None
        self.R = None
        self.T = None
        self.load_camera_params(calib_file)
        self.image_sub = None

    def load_camera_params(self, calib_file):
        """加载相机校准参数"""
        try:
            with open(calib_file, 'r') as f:
                params = yaml.safe_load(f)['block_params']
            self.K = np.array(params['K'], dtype=np.float64).reshape(3, 3)
            self.D = np.array(params['D'], dtype=np.float64).reshape(5, 1)
            self.R = np.array(params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(params['T'], dtype=np.float64).reshape(3, 1)
            rospy.loginfo("相机参数加载成功")
        except Exception as e:
            rospy.logerr(f"加载相机参数失败: {e}")

    def undistort_points(self, points):
        """对像素点进行去畸变处理"""
        raw_pixels = np.array([points], dtype=np.float32)
        undistorted = cv2.undistortPoints(raw_pixels, self.K, self.D, P=self.K)
        return undistorted[0, 0]

    def camera_to_world(self, img_points):
        rospy.loginfo(f"输入像素坐标: {img_points}")
        inv_k = np.asmatrix(self.K).I
        r_mat = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(self.R, r_mat)
        inv_r = np.asmatrix(r_mat).I
        transPlaneToCam = np.dot(inv_r, np.asmatrix(self.T))
        world_pts = []
        coords = np.zeros((3, 1), dtype=np.float64)

        for img_pt in img_points:
            coords[0][0], coords[1][0], coords[2][0] = img_pt[0], img_pt[1], 1.0
            worldPtCam = np.dot(inv_k, coords)
            worldPtPlane = np.dot(inv_r, worldPtCam)
            scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
            scale_worldPtPlane = np.multiply(scale, worldPtPlane)
            worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)
            pt = np.zeros((3, 1), dtype=np.float64)
            pt[0][0], pt[1][0], pt[2][0] = worldPtPlaneReproject[0][0], worldPtPlaneReproject[1][0], 0
            world_pts.append([worldPtPlaneReproject[0][0], worldPtPlaneReproject[1][0], 0])
        rospy.loginfo(f"世界坐标输出: {world_pts}")
        return world_pts

    def start_subscriber(self):
        """启动相机图像订阅"""
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        rospy.loginfo("相机图像订阅已启动")

    def stop_subscriber(self):
        """停止相机图像订阅"""
        if self.image_sub:
            self.image_sub.unregister()
            self.image_sub = None
            rospy.loginfo("相机图像订阅已停止")

    def image_callback(self, ros_image):
        """相机图像回调函数"""
        try:
            image_queue.put_nowait(ros_image)
        except queue.Full:
            pass

class AprilTagDetector:
    def __init__(self, tag_stable_threshold=10, center_tolerance=10):
        """初始化AprilTag检测器"""
        self.tag_detection_count = 0
        self.last_tag_center = None
        self.tag_stable_threshold = tag_stable_threshold
        self.center_tolerance = center_tolerance
        self.moving_box = None
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        rospy.loginfo("AprilTag检测器初始化完成")

    def process_apriltag(self, image):
        """处理AprilTag检测"""
        org_image = np.copy(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detections = self.detector.detect(gray)
        has_tag = False

        if detections and self.moving_box is None:
            tag = detections[0]
            c_x, c_y = tag.center

            # 检查标签稳定性
            if self.last_tag_center is None:
                self.last_tag_center = (c_x, c_y)
                self.tag_detection_count = 1
            else:
                dist = np.sqrt((c_x - self.last_tag_center[0])**2 + (c_y - self.last_tag_center[1])**2)
                if dist < self.center_tolerance:
                    self.tag_detection_count += 1
                    self.last_tag_center = (c_x, c_y)
                else:
                    self.tag_detection_count = 1
                    self.last_tag_center = (c_x, c_y)

            # 绘制检测结果
            corners = np.array(tag.corners, dtype=np.int32)
            cv2.polylines(org_image, [corners], True, (0, 255, 0), 2)
            cv2.circle(org_image, (int(c_x), int(c_y)), 5, (0, 0, 255), -1)

            # 判断是否达到稳定检测阈值
            if self.tag_detection_count >= self.tag_stable_threshold:
                has_tag = True
                self.moving_box = (c_x, c_y, "tag36h11")
                self.tag_detection_count = 0
                self.last_tag_center = None
                rospy.loginfo(f"检测到稳定AprilTag，中心: ({c_x}, {c_y})")
        else:
            self.tag_detection_count = 0
            self.last_tag_center = None

        return org_image, has_tag

class RobotController:
    def __init__(self):
        """初始化机械臂控制器"""
        self.jetmax = hiwonder.JetMax()
        self.sucker = hiwonder.Sucker()
        self.moving_thread = None
        self.tag_size = 33.3  # AprilTag尺寸（毫米）
        self.default_stack_height = 92  # 默认堆叠高度（毫米）

    def init_servos(self):
        """初始化伺服电机到默认位置"""
        try:
            self.jetmax.set_servo(3, 500, 1)
            time.sleep(1)
            self.jetmax.set_servo(2, 500, 1)
            time.sleep(1)
            self.jetmax.set_servo(1, 125, 1)
            time.sleep(1)
            rospy.loginfo("伺服电机初始化完成")
        except Exception as e:
            rospy.logerr(f"伺服初始化失败: {e}")

    def cleanup(self):
        """清理硬件资源"""
        try:
            self.jetmax.go_home(1)
            self.sucker.release(1)
            rospy.loginfo("硬件资源清理完成")
        except Exception as e:
            rospy.logerr(f"硬件 Uppsala失败: {e}")

    def calculate_target_position(self, camera_handler, c_x, c_y, rotate=True):
        """计算目标世界坐标"""
        # 去畸变处理
        undistorted_pixel = camera_handler.undistort_points([[c_x, c_y]])
        c_x, c_y = undistorted_pixel[0], undistorted_pixel[1]
        rospy.loginfo(f"去畸变后像素坐标: ({c_x}, {c_y})")

        # 像素坐标转世界坐标
        world_pts = camera_handler.camera_to_world(np.array([[c_x, c_y]]))
        x, y, _ = world_pts[0]  # 直接解包 [x, y, z]
        rospy.loginfo(f"世界坐标偏移: ({x}, {y})")

        # 坐标旋转：抓取顺时针90度，放置逆时针90度
        x_rotated, y_rotated = (y, -x) if rotate else (-y, x)
        rospy.loginfo(f"旋转后坐标: ({x_rotated}, {y_rotated})")

        # 计算目标位置
        cur_x, cur_y, _ = self.jetmax.position
        return cur_x + x_rotated, cur_y + y_rotated

    def pick_object(self, camera_handler, c_x, c_y):
        """执行抓取操作"""
        try:
            new_x, new_y = self.calculate_target_position(camera_handler, c_x, c_y, rotate=True)
            rospy.loginfo(f"抓取目标位置: ({new_x}, {new_y})")

            # 移动到目标上方
            self.jetmax.set_position((new_x, new_y, 100), 2)
            rospy.sleep(2)
            # 下降并吸取
            self.jetmax.set_position((new_x, new_y, 92), 0.5)
            rospy.sleep(0.5)
            self.sucker.suck(1)
            rospy.sleep(1)
            # 抬起
            self.jetmax.set_position((new_x, new_y, 150), 0.5)
            rospy.sleep(0.5)
            rospy.loginfo("抓取完成")
            return new_x, new_y
        except Exception as e:
            rospy.logerr(f"抓取失败: {e}")
            return None, None

    def place_object(self, camera_handler, detector, image):
        """执行放置操作"""
        try:
            # 移动到初始放置姿态
            self.jetmax.go_home(1)
            rospy.sleep(1)
            self.jetmax.set_servo(1, 1000 - 125, 1)
            rospy.sleep(2)

            # 获取新图像进行AprilTag检测
            try:
                ros_image = image_queue.get(block=True, timeout=1.0)
                fresh_image = np.ndarray(
                    shape=(ros_image.height, ros_image.width, 3),
                    dtype=np.uint8,
                    buffer=ros_image.data
                )
            except queue.Empty:
                rospy.logerr("无法获取新图像，使用旧图像")
                fresh_image = image

            # AprilTag检测
            gray = cv2.cvtColor(fresh_image, cv2.COLOR_RGB2GRAY)
            detections = detector.detector.detect(gray)

            if not detections:
                rospy.loginfo("未检测到AprilTag，使用默认放置位置")
                cur_x, cur_y, _ = self.jetmax.position
                self.jetmax.set_position((cur_x + 40, cur_y, self.default_stack_height), 2)
                rospy.sleep(2.5)
            else:
                rospy.loginfo("检测到AprilTag，进行校准")
                tag = detections[0]
                c_x, c_y = tag.center
                corners = np.array(tag.corners, dtype=np.float32)
                obj_points = np.array([
                    [self.tag_size/2, -self.tag_size/2, 0],
                    [-self.tag_size/2, -self.tag_size/2, 0],
                    [-self.tag_size/2, self.tag_size/2, 0],
                    [self.tag_size/2, self.tag_size/2, 0]
                ], dtype=np.float32)

                ret, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_handler.K, camera_handler.D)
                stack_height = self.default_stack_height
                if tvec is not None and camera_handler.T is not None:
                    z_diff = camera_handler.T[2][0] - tvec[2][0] + 40
                    num_stacks = round(z_diff / 40)
                    stack_height = self.default_stack_height + z_diff
                    rospy.loginfo(f"高度差: {z_diff}, 堆叠层数: {num_stacks}")

                new_x, new_y = self.calculate_target_position(camera_handler, c_x, c_y, rotate=False)
                self.jetmax.set_position((new_x, new_y, stack_height), 2)
                rospy.sleep(2.5)

            # 释放物体
            self.sucker.release(1)
            # 抬起并复位
            self.jetmax.set_servo(3, 500, 0.5)
            rospy.sleep(0.5)
            self.jetmax.go_home(1)
            rospy.sleep(1)
            self.jetmax.set_servo(1, 125, 1)
            rospy.sleep(1)
            rospy.loginfo("放置完成")
            return True
        except Exception as e:
            rospy.logerr(f"放置失败: {e}")
            return False

    def move_to_target(self, camera_handler, detector, image):
        """执行完整的抓取和放置操作"""
        try:
            if detector.moving_box is None:
                return False
            c_x, c_y, _ = detector.moving_box
            rospy.loginfo(f"开始移动，初始像素坐标: ({c_x}, {c_y})")

            # 执行抓取
            new_x, new_y = self.pick_object(camera_handler, c_x, c_y)
            if new_x is None or new_y is None:
                rospy.logerr("抓取失败，终止操作")
                return False

            # 执行放置
            success = self.place_object(camera_handler, detector, image)
            return success
        finally:
            rospy.loginfo("移动完成")
            detector.moving_box = None
            self.moving_thread = None

    def start_moving(self, camera_handler, detector, image):
        """启动移动线程"""
        if self.moving_thread is None or not self.moving_thread.is_alive():
            self.moving_thread = threading.Thread(
                target=self.move_to_target,
                args=(camera_handler, detector, image)
            )
            self.moving_thread.start()

class MainController:
    def __init__(self):
        """初始化主控制器"""
        self.camera = CameraHandler()
        self.detector = AprilTagDetector()
        self.robot = RobotController()
        self.image_pub = None

    def init(self):
        """初始化程序"""
        rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
        self.camera.start_subscriber()
        self.robot.init_servos()
        self.image_pub = rospy.Publisher(f'/{ROS_NODE_NAME}/image_result', Image, queue_size=1)
        
        # 启动 Flask 服务器
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        rospy.loginfo("流服务器启动于 http://0.0.0.0:5001/video_feed")

        hiwonder.buzzer.on()
        rospy.sleep(0.2)
        hiwonder.buzzer.off()
        rospy.loginfo("程序初始化完成")

    def process_image(self):
        """处理图像并发布结果"""
        global latest_frame
        try:
            ros_image = image_queue.get(block=False)
        except queue.Empty:
            return False

        image = np.ndarray(
            shape=(ros_image.height, ros_image.width, 3),
            dtype=np.uint8,
            buffer=ros_image.data
        )
        processed_image, has_tag = self.detector.process_apriltag(image)
        latest_frame = processed_image.copy()
        rgb_image = processed_image.tobytes()
        ros_image.data = rgb_image
        self.image_pub.publish(ros_image)
        return has_tag

    def run(self):
        while not rospy.is_shutdown() and not shutdown_event.is_set():
            has_tag = self.process_image()
            if has_tag and self.detector.moving_box and (self.robot.moving_thread is None or not self.robot.moving_thread.is_alive()):
                rospy.loginfo("Starting new move operation")
                self.robot.start_moving(self.camera, self.detector, latest_frame)
            time.sleep(0.01)

    def shutdown(self):
        """程序关闭处理"""
        rospy.loginfo("开始关闭程序...")
        shutdown_event.set()
        self.camera.stop_subscriber()
        self.robot.cleanup()
        if self.robot.moving_thread and self.robot.moving_thread.is_alive():
            self.robot.moving_thread.join(timeout=2.0)
        # Flask 服务器通过 shutdown_event 停止
        hiwonder.buzzer.on()
        rospy.sleep(0.2)
        hiwonder.buzzer.off()
        rospy.loginfo("程序已安全退出")

def main():
    """程序入口"""
    controller = MainController()
    rospy.on_shutdown(controller.shutdown)
    controller.init()
    controller.run()

if __name__ == '__main__':
    main()