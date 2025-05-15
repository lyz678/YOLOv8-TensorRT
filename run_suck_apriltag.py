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
import signal
import sys
from werkzeug.serving import make_server
import yaml

# Flask 应用
app = Flask(__name__)
latest_frame = None
image_queue = queue.Queue(maxsize=3)
flask_server = None  # 用于存储 Flask 服务器实例

# Flask 视频流生成
def generate_frames():
    global latest_frame
    while not rospy.is_shutdown():
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
class FlaskServerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.server = make_server('0.0.0.0', 5001, app)
        self.daemon = True

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.join()

ROS_NODE_NAME = 'suck_apriltag'

class ObjectMover:
    def __init__(self):
        self.image_sub = None
        self.camera_params = None
        self.K = None
        self.R = None
        self.T = None
        self.moving_box = None
        self.runner = None

    def reset(self):
        if self.image_sub is not None:
            self.image_sub.unregister()
        self.image_sub = None
        self.moving_box = None
        self.runner = None

    def load_camera_params(self):
        """加载相机校准参数"""
        try:
            with open('/home/lyz/.ros/camera_info/apriltag_calibration_default.yaml', 'r') as f:
                params = yaml.safe_load(f)['block_params']
            self.K = np.array(params['K'], dtype=np.float64).reshape(3, 3)
            self.D = np.array(params['D'], dtype=np.float64).reshape(5, 1)
            self.R = np.array(params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(params['T'], dtype=np.float64).reshape(3, 1)
            rospy.loginfo("相机参数加载成功")
        except Exception as e:
            rospy.logerr(f"加载相机参数失败: {e}")

    def init_servos(self):
        try:
            jetmax.set_servo(3, 500, 1)
            time.sleep(1)
            jetmax.set_servo(2, 500, 1)
            time.sleep(1)
            jetmax.set_servo(1, 125, 1)
            time.sleep(1)
        except Exception as e:
            print(f"\033[31m错误：无法设置伺服电机 1 位置：{e}\033[0m")

    def cleanup(self):
        self.reset()
        try:
            jetmax.go_home(1)
            time.sleep(1)
            sucker.release(1)
            time.sleep(1)
        except Exception as e:
            print(f"\033[31m清理硬件时出错：{e}\033[0m")

def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
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

def moving(state, jetmax, sucker):
    try:
        if state.moving_box is None:
            return
        c_x, c_y, _ = state.moving_box
        cur_x, cur_y, cur_z = jetmax.position
        print("当前位置: ", jetmax.position)
        print("像素坐标: ", c_x, c_y)

        x, y, _ = camera_to_world(state.K, state.R, state.T, np.array([[c_x, c_y]]).reshape((1, 1, 2)))[0][0]
        
        x_rotated = y
        y_rotated = -x

        print("世界坐标偏移: ", x_rotated, y_rotated)

        new_x, new_y = cur_x + x_rotated, cur_y + y_rotated

        jetmax.set_position((new_x, new_y, 100), 2)
        rospy.sleep(2)

        jetmax.set_position((new_x, new_y, 95), 0.5)
        rospy.sleep(0.5)
        sucker.suck(2)
        rospy.sleep(2)

        jetmax.set_position((new_x, new_y, 212.8), 1)
        rospy.sleep(1)
        
        jetmax.set_servo(1, 1000 - 125, 2)
        rospy.sleep(2.5)
        jetmax.set_position((146.14, 0, 95), 2)
        rospy.sleep(2.5)
        
        sucker.release(3)
        jetmax.set_servo(3, 500, 1)
        rospy.sleep(1)

    finally:
        print("移动完成")
        jetmax.go_home(1)
        print("结束位置: ", jetmax.position)
        state.moving_box = None

mover = ObjectMover()
jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()

def init():
    mover.reset()
    mover.load_camera_params()
    mover.init_servos()
    mover.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

def image_proc():
    global latest_frame
    try:
        ros_image = image_queue.get(block=False)
    except queue.Empty:
        return
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    image = process_apriltag(image)
    latest_frame = image.copy()
    rgb_image = image.tobytes()  # 修正 tostring() 为 tobytes()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)

def process_apriltag(image):
    org_image = np.copy(image)
    image = cv2.resize(image, (320, 240))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    detections = detector.detect(gray)

    if detections and mover.runner is None:
        tag = detections[0]
        center = tag.center
        c_x = hiwonder.misc.val_map(center[0], 0, 320, 0, 640)
        c_y = hiwonder.misc.val_map(center[1], 0, 240, 0, 480)

        corners = tag.corners
        corners = [(hiwonder.misc.val_map(c[0], 0, 320, 0, 640),
                    hiwonder.misc.val_map(c[1], 0, 240, 0, 480)) for c in corners]
        corners = np.array(corners, dtype=np.int32)
        cv2.polylines(org_image, [corners], True, (0, 255, 0), 2)
        cv2.circle(org_image, (int(c_x), int(c_y)), 5, (0, 0, 255), -1)

        mover.moving_box = (c_x, c_y, "tag36h11")
        mover.runner = threading.Thread(target=moving, args=(mover, jetmax, sucker))
        mover.runner.start()

    return org_image

def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

def shutdown_hook():
    rospy.loginfo("开始关闭程序...")
    mover.cleanup()
    if mover.runner is not None and mover.runner.is_alive():
        mover.runner.join(timeout=2.0)
    if flask_server is not None:
        flask_server.stop()
    hiwonder.buzzer.on()
    rospy.sleep(0.2)
    hiwonder.buzzer.off()
    rospy.loginfo("程序已安全退出")

def main():
    global image_pub, flask_server
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    rospy.on_shutdown(shutdown_hook)
    
    init()
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)

    flask_server = FlaskServerThread()
    flask_server.start()
    rospy.loginfo("流服务器启动于 http://0.0.0.0:5000/video_feed")

    hiwonder.buzzer.on()
    rospy.sleep(0.2)
    hiwonder.buzzer.off()

    try:
        while not rospy.is_shutdown():
            image_proc()
            time.sleep(0.01)
    except rospy.ROSInterruptException:
        pass
    finally:
        shutdown_hook()

if __name__ == '__main__':
    main()