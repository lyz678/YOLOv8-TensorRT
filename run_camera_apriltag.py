#!/usr/bin/env python3
import sys
import cv2
import math
import time
import rospy
import numpy as np
import threading
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from jetmax_control.msg import SetServo
import hiwonder
from hiwonder import serial_servo
import queue
import pupil_apriltags as apriltag
import yaml
from flask import Flask, Response
import rospkg

"""
AprilTag 识别定位实验
将jetmax调整到摄像头朝下的形态
将id1及其他id的apriltag放于摄像头下方，程序将识别apriltag并计算其他tag相对与id1的tag的位置
"""

TAG_SIZE = 33.30

# Flask app setup
app = Flask(__name__)

class AprilTagDetect:
    def __init__(self):
        self.camera_params = None
        self.K = None
        self.R = None
        self.T = None
        self.frame = None  # Store the latest processed frame
        self.lock = threading.Lock()  # Thread-safe access to frame
        self.last_time = time.time()  # For FPS calculation

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/block_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            r_mat = np.zeros((3, 3), dtype=np.float64)
            cv2.Rodrigues(self.R, r_mat)
            self.r_mat = r_mat

def camera_to_world(cam_mtx, r_mat, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    inv_r = np.asmatrix(r_mat).I
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)
        worldPtPlane = np.dot(inv_r, worldPtCam)
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt

def image_proc_a(img, state, at_detector):
    frame_gray = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2GRAY)
    params = [state.K[0][0], state.K[1][1], state.K[0][2], state.K[1][2]]
    tags = at_detector.detect(frame_gray, estimate_tag_pose=True, camera_params=params, tag_size=TAG_SIZE)
    for tag in tags:
        corners = tag.corners.reshape(1, -1, 2).astype(int)
        center = tag.center.astype(int)
        cv2.drawContours(img, corners, -1, (255, 0, 0), 3)
        cv2.circle(img, tuple(center), 5, (255, 255, 0), 10)
        rotM = tag.pose_R
        tvec = tag.pose_t
        if tag.tag_id == 1:
            state.r_mat = rotM
            state.T = tvec
        else:
            x, y, _ = camera_to_world(state.K, state.r_mat, state.T, tag.center.reshape((1,1,2)))[0][0]
            theta_z = math.atan2(rotM[1, 0], rotM[0, 0]) * 180.0 / math.pi
            print("id:{}, x:{:0.2f}mm, y:{:0.2f}mm, angle:{:0.2f}deg".format(tag.tag_id, x, y, theta_z))
            s1 = "id:{}".format(tag.tag_id)
            s2 = "x:{:0.2f}mm, y:{:0.2f}mm".format(x, y)
            s3 = "angle:{:0.2f}deg".format(theta_z)
            cv2.putText(img, s1, (center[0] - 50, center[1]), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(img, s2, (center[0] - 50, center[1] + 20), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(img, s3, (center[0] - 50, center[1] + 40), 0, 0.7, (0, 255, 0), 2)
    img_h, img_w = img.shape[:2]
    cv2.line(img, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
    cv2.line(img, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)
    
    # 计算并绘制FPS
    current_time = time.time()
    fps = 1.0 / (current_time - state.last_time) if current_time > state.last_time else 0
    state.last_time = current_time
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img

def image_proc_b(state, at_detector, image_queue, stop_event):
    while not rospy.is_shutdown() and not stop_event.is_set():
        try:
            ros_image = image_queue.get(block=True, timeout=1)
            image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
            frame_result = image.copy()
            frame_result = image_proc_a(frame_result, state, at_detector)
            bgr_image = cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR)
            # 存储帧用于推流
            with state.lock:
                state.frame = bgr_image.copy()
        except queue.Empty:
            continue
        except Exception as e:
            rospy.logerr(f"Error in image_proc_b: {e}")

def image_callback(ros_image, image_queue):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

def gen_frames(state):
    while True:
        with state.lock:
            if state.frame is None:
                # 如果没有帧，生成空白图像
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, jpeg = cv2.imencode('.jpg', blank)
                if not ret:
                    continue
                frame = jpeg.tobytes()
            else:
                ret, jpeg = cv2.imencode('.jpg', state.frame)
                if not ret:
                    continue
                frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 fps

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(state),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    try:
        app.run(host='0.0.0.0', port=5000, threaded=False)
    except Exception as e:
        rospy.logerr(f"Flask server failed: {e}")

ROS_NODE_NAME = "apriltag_detector"
DEFAULT_X, DEFAULT_Y, DEFAULT_Z = 0, 138 + 8.14, 84 + 128.4
TARGET_PIXEL_X, TARGET_PIXEL_Y = 320, 240

def init_servos():
    serial_servo.set_position(1, 100, 1000)
    serial_servo.set_position(2, 500, 1000)
    serial_servo.set_position(3, 500, 1000)

def return_all_servos_to_end():
    print("\nReturning all servos to end position...")
    serial_servo.set_position(1, 500, 1000)
    serial_servo.set_position(2, 700, 1000)
    serial_servo.set_position(3, 550, 1000)
    print("All servos returned to end position")

if __name__ == '__main__':
    state = AprilTagDetect()
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    at_detector = apriltag.Detector()
    image_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()  # 用于控制退出
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    init_servos()
    state.load_camera_params()
    if state.camera_params is None:
        rospy.logerr('Can not load camera parameters')
        sys.exit(-1)
    image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, lambda img: image_callback(img, image_queue), queue_size=1)
    
    # 启动图像处理线程
    processing_thread = threading.Thread(target=image_proc_b, args=(state, at_detector, image_queue, stop_event))
    processing_thread.daemon = True
    processing_thread.start()
    
    # 启动Flask服务器
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    rospy.loginfo("Stream server started at http://0.0.0.0:5000/video_feed")
    
    # 主线程等待退出信号
    try:
        while not stop_event.is_set() and not rospy.is_shutdown():
            rospy.sleep(0.1)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        stop_event.set()
        return_all_servos_to_end()
        sucker.set_state(False)
        cv2.destroyAllWindows()