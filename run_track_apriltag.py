#!/usr/bin/env python3
import math
import rospy
import time
import queue
import threading
from sensor_msgs.msg import Image
import cv2
import numpy as np
import hiwonder
import apriltag
import sys
import select
from flask import Flask, Response
from collections import deque

# Flask 应用
app = Flask(__name__)
latest_frame = None  # 存储最新的处理帧
tracking_data = {"status": "Initializing", "servo_pos": {}, "fps": 0}  # 跟踪状态
image_queue = queue.Queue(maxsize=3)
servo_lock = threading.Lock()

# 检查键盘输入（非阻塞）
def is_key_pressed():
    return select.select([sys.stdin], [], [], 0)[0]

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

# Flask 状态端点
@app.route('/status')
def get_status():
    global tracking_data
    return {
        "status": tracking_data["status"],
        "servo_positions": tracking_data["servo_pos"],
        "fps": tracking_data["fps"]
    }

# 运行 Flask 服务器的线程
def run_flask():
    app.run(host='0.0.0.0', port=5001, threaded=False)

ROS_NODE_NAME = 'object_tracking'
TARGET_PIXEL_X, TARGET_PIXEL_Y = 320, 240

class ObjectTracking:
    def __init__(self):
        self.image_sub = None
        self.lock = threading.RLock()
        self.servo_x = 100
        self.servo_y = 500
        self.angle_speed = 100  # Servo angular speed (degrees/second)

        # AprilTag 跟踪 PID 控制器
        self.apriltag_x_pid = hiwonder.PID(0.07, 0.01, 0.0015)
        self.apriltag_y_pid = hiwonder.PID(0.08, 0.008, 0.001)

        self.last_tag_center = None
        self.lost_target_count = 0
        self.is_running_apriltag = True  # 默认开启跟踪

        self.fps = 0.0
        self.tic = time.time()
        self.fps_deque = deque(maxlen=30)  # 用于计算平均 FPS

    def reset(self):
        self.image_sub = None
        self.last_tag_center = None
        self.lost_target_count = 0
        self.tic = time.time()

    def move_servo(self, servo_id, target_position, angle_speed):
        """
        Move the specified servo to the target position at a uniform angular speed.
        :param servo_id: Servo ID (1: horizontal rotation, 2: forward/backward, 3: vertical height)
        :param target_position: Target position
        :param angle_speed: Angular speed (degrees/second)
        :return: Success status
        """
        with servo_lock:
            # Restrict servo position range
            if servo_id == 1:  # Horizontal rotation
                min_pos, max_pos = 0, 1000
            elif servo_id == 2:  # Forward/backward
                min_pos, max_pos = 0, 1000
            elif servo_id == 3:  # Vertical height
                min_pos, max_pos = 0, 1000
            else:
                return False

            # Check if out of bounds
            target_position = max(min_pos, min(target_position, max_pos))
            
            # Read current servo position
            try:
                current_position = hiwonder.serial_servo.read_pos(servo_id)
            except Exception as e:
                print(f"\033[31mError: Cannot read servo {servo_id} position: {e}\033[0m")
                return False
                
            # Calculate current angle
            current_angle = current_position / 1000 * 240
            # Calculate target angle
            target_angle = target_position / 1000 * 240
            # Calculate angle difference
            angle_diff = target_angle - current_angle
            # Calculate required time (seconds), ensure positive
            move_time = abs(angle_diff / angle_speed)
            
            # Send servo movement command (convert to milliseconds)
            command_time_ms = int(move_time * 1000)
            try:
                hiwonder.serial_servo.set_position(servo_id, target_position, command_time_ms)
                print(f"Command sent - Servo ID: {servo_id}, Current: {current_position}, Target: {target_position}, Time: {command_time_ms}ms")
            except Exception as e:
                print(f"\033[31mError: Cannot set servo {servo_id} position: {e}\033[0m")
                return False
                
            # Update tracking data
            tracking_data["servo_pos"][f"servo_{servo_id}"] = target_position
            
            # Wait for movement to complete
            time.sleep(move_time)
            return True

    def init_servos(self):
        """将所有舵机恢复到初始位置"""
        self.move_servo(1, 100, self.angle_speed)
        self.move_servo(2, 500, self.angle_speed)
        self.move_servo(3, 500, self.angle_speed)

    def return_all_servos_to_end(self):
        """将所有舵机恢复到结束位置"""
        print("\nReturning all servos to end position...")
        self.move_servo(1, 500, 50)
        self.move_servo(2, 700, 50)
        self.move_servo(3, 550, 50)
        print("All servos returned to end position")

tracker = ObjectTracking()
jetmax = hiwonder.JetMax()

def init():
    tracker.reset()
    tracker.init_servos()
    tracker.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

def image_proc():
    global latest_frame, tracking_data
    try:
        ros_image = image_queue.get(block=True)
    except queue.Empty:
        return
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    if tracker.is_running_apriltag:
        image = apriltag_tracking(image)
    # 计算 FPS
    toc = time.time()
    curr_fps = 1.0 / (toc - tracker.tic) if (toc - tracker.tic) > 0 else 0
    tracker.fps_deque.append(curr_fps)
    tracker.fps = sum(tracker.fps_deque) / len(tracker.fps_deque) if tracker.fps_deque else 0
    tracker.tic = toc
    # 更新跟踪数据
    tracking_data["status"] = "Tracking" if tracker.is_running_apriltag else "Idle"
    tracking_data["servo_pos"] = {"servo_x": tracker.servo_x, "servo_y": tracker.servo_y}
    tracking_data["fps"] = tracker.fps
    # 存储最新帧
    latest_frame = image.copy()
    # 发布结果图像
    rgb_image = image.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)

def apriltag_tracking(image):
    org_image = np.copy(image)
    image = cv2.resize(image, (320, 240))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 绘制中线（在原始图像上）
    cv2.line(org_image, (TARGET_PIXEL_X, 0), (TARGET_PIXEL_X, org_image.shape[0]), (255, 255, 0), 1)  # 垂直线
    cv2.line(org_image, (0, TARGET_PIXEL_Y), (org_image.shape[1], TARGET_PIXEL_Y), (255, 255, 0), 1)  # 水平线

    # 绘制 FPS
    fps_text = f"FPS: {tracker.fps:.2f}"
    cv2.putText(org_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    detections = detector.detect(gray)

    tag_center = None
    with tracker.lock:
        if detections:
            tag = detections[0]
            center = tag.center
            c_x = hiwonder.misc.val_map(center[0], 0, 320, 0, 640)
            c_y = hiwonder.misc.val_map(center[1], 0, 240, 0, 480)
            tag_center = (c_x, c_y)
            tracker.lost_target_count = 0

            corners = tag.corners
            corners = [(hiwonder.misc.val_map(c[0], 0, 320, 0, 640),
                        hiwonder.misc.val_map(c[1], 0, 240, 0, 480)) for c in corners]
            corners = np.array(corners, dtype=np.int32)
            cv2.polylines(org_image, [corners], True, (0, 255, 0), 2)
            cv2.circle(org_image, (int(c_x), int(c_y)), 5, (0, 0, 255), -1)

            x_error = c_x - TARGET_PIXEL_X
            if abs(x_error) > 30:
                tracker.apriltag_x_pid.SetPoint = 0
                tracker.apriltag_x_pid.update(x_error)
                tracker.servo_x += tracker.apriltag_x_pid.output
            else:
                tracker.apriltag_x_pid.update(0)

            y_error = c_y - TARGET_PIXEL_Y
            if abs(y_error) > 30:
                tracker.apriltag_y_pid.SetPoint = 0
                tracker.apriltag_y_pid.update(y_error)
                tracker.servo_y -= tracker.apriltag_y_pid.output
            else:
                tracker.apriltag_y_pid.update(0)

            tracker.servo_y = max(350, min(650, tracker.servo_y))
            tracker.servo_x = max(0, min(1000, tracker.servo_x))

            jetmax.set_servo(1, int(tracker.servo_x), duration=0.02)
            jetmax.set_servo(2, int(tracker.servo_y), duration=0.02)

            tracker.last_tag_center = (center[0], center[1])
        else:
            tracker.lost_target_count += 1
            if tracker.lost_target_count > 15:
                tracker.lost_target_count = 0
                tracker.last_tag_center = None

        if tracker.last_tag_center and detections:
            last_x, last_y = tracker.last_tag_center
            dist = math.sqrt((center[0] - last_x) ** 2 + (center[1] - last_y) ** 2)
            if dist > 50:
                tag_center = None
                tracker.lost_target_count += 1

    return org_image

def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

def init():
    tracker.reset()
    tracker.init_servos()
    tracker.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

def main():
    global image_pub
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    init()

    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)

    # 启动 Flask 线程
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    rospy.loginfo("Stream server started at http://0.0.0.0:5001/video_feed")

    hiwonder.buzzer.on()
    rospy.sleep(0.2)
    hiwonder.buzzer.off()

    try:
        while not rospy.is_shutdown():
            image_proc()
            # 检查键盘输入
            if is_key_pressed():
                input_str = sys.stdin.read(1).lower()
                if input_str == 'q':
                    rospy.loginfo("Exit requested, stopping...")
                    tracker.return_all_servos_to_end()
                    rospy.signal_shutdown("User requested shutdown")
                    break
                elif input_str == 'st':
                    tracker.is_running_apriltag = False
                    rospy.loginfo("Stopped tracking")
                elif input_str == 'bt':
                    tracker.is_running_apriltag = True
                    rospy.loginfo("Started tracking mode")
            time.sleep(0.01)  # 减少 CPU 占用
    finally:
        tracker.return_all_servos_to_end()
        rospy.loginfo(f"Final FPS: {tracker.fps:.2f}")

if __name__ == '__main__':
    main()