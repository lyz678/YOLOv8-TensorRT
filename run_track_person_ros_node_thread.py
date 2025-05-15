#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import threading
from pathlib import Path
import cv2
import numpy as np
from models.pycuda_api import TRTEngine
from config import CLASSES_DET, COLORS
from models.utils import blob, det_postprocess, letterbox
from sort.sort import Sort
from collections import deque
from flask import Flask, Response
import sys
import select
import queue
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

def is_key_pressed():
    return select.select([sys.stdin], [], [], 0)[0]

app = Flask(__name__)
latest_frame = None
tracking_data = {"status": "Initializing", "servo_pos": {}, "fps": 0}
image_queue = queue.Queue(maxsize=3)

class PersonTracker:
    def __init__(self, args):
        self.engine = TRTEngine(args.engine)
        self.H, self.W = self.engine.inp_info[0].shape[-2:]
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.angle_speed = 100  # 舵机运动角速度(度/秒)
        self.idle_speed = 50    # 扫描状态角速度(度/秒，降低以减少抖动)
        self.scan_interval = 0.08  # 一次扫描时间(秒)

        # 舵机中间位置
        self.servo_mid = {1: 500, 2: 600, 3: 500}
        
        # 当前舵机位置
        self.current_pos = {1: self.servo_mid[1], 2: self.servo_mid[2], 3: self.servo_mid[3]}
        
        # 单一舵机指令队列
        self.servo_queue = deque(maxlen=3)
        self.servo_lock = threading.Lock()
        self.servo_condition = threading.Condition(self.servo_lock)  # 使用条件变量
        self.servo_moving = {1: False, 2: False, 3: False}
        
        # ROS话题发布者
        self.servo_pub = rospy.Publisher('/servo_command', Int32MultiArray, queue_size=10)
        print("等待舵机控制节点连接...")
        while self.servo_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            time.sleep(0.1)
        print("舵机控制节点已连接")
        
        # 启动单一舵机管理线程
        threading.Thread(target=self.servo_manager, daemon=True).start()
        
        # 初始化舵机2和3到中间位置
        self.move_servo(2, self.servo_mid[2], self.idle_speed)
        self.move_servo(3, self.servo_mid[3], self.idle_speed)
        while any(self.servo_moving.values()):
            time.sleep(0.1)
        
        # 图像中心
        self.image_center_x = self.W // 2
        self.image_center_y = self.H // 2
        
        # 控制参数
        self.x_threshold = self.W * 0.1
        self.y_threshold = self.H * 0.1
        self.size_threshold = 0.3
        
        # 扫描状态
        self.scanning = True
        self.scan_direction = -1
        self.last_scan_time = time.time()
        
        # 追踪模式
        self.tracking_enabled = True
        
        # FPS计算
        self.frame_times = deque(maxlen=10)
        self.avg_fps = 0
    
    def move_servo(self, servo_id, target_position, angle_speed):
        with self.servo_condition:
            for i, (sid, _, _) in enumerate(self.servo_queue):
                if sid == servo_id:
                    self.servo_queue[i] = (servo_id, target_position, angle_speed)
                    self.servo_condition.notify()
                    return
            self.servo_queue.append((servo_id, target_position, angle_speed))
            self.servo_condition.notify()

    def servo_manager(self):
        while True:
            with self.servo_condition:
                while not self.servo_queue:
                    self.servo_condition.wait()
                servo_id, target_position, angle_speed = self.servo_queue.popleft()
                self.servo_moving[servo_id] = True
            
            try:
                msg = Int32MultiArray()
                msg.data = [servo_id, target_position, angle_speed]
                self.servo_pub.publish(msg)
                with self.servo_condition:
                    self.current_pos[servo_id] = target_position
                    tracking_data["servo_pos"][f"servo_{servo_id}"] = target_position
                print(f"舵机{servo_id}指令已发送: 目标位置={target_position}, 速度={angle_speed}")
            except Exception as e:
                print(f"舵机{servo_id}指令发送失败: {str(e)}")
            finally:
                with self.servo_condition:
                    self.servo_moving[servo_id] = False

    def return_all_servos_to_end(self):
        print("\nReturning all servos to end position...")
        self.move_servo(1, 500, 50)
        self.move_servo(2, 700, 50)
        self.move_servo(3, 550, 50)
        while any(self.servo_moving.values()):
            time.sleep(0.1)
        print("All servos returned to end position")

    def set_tracking_mode(self, enabled):
        self.tracking_enabled = enabled
        if enabled:
            self.scanning = True
            tracking_data["status"] = "Tracking enabled - Scanning for target"
        else:
            tracking_data["status"] = "Tracking disabled - Showing all detections"
    
    def update_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            self.avg_fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0
        else:
            self.avg_fps = 0
        tracking_data["fps"] = self.avg_fps
    
    def process_frame(self, rgb):
        global latest_frame, tracking_data
        
        # 开始总计时
        total_start = time.perf_counter()
        
        self.update_fps()
        
        # 前处理计时
        preprocess_start = time.perf_counter()
        rgb, ratio, dwdh = letterbox(rgb, (self.W, self.H))
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000  # 转换为毫秒
        
        # 推理计时
        inference_start = time.perf_counter()
        data = self.engine(tensor)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # 后处理计时
        postprocess_start = time.perf_counter()
        bboxes, scores, labels = det_postprocess(data)
        
        if bboxes.size == 0:
            detections = np.array([])
        else:
            bboxes -= dwdh
            bboxes /= ratio
            detections = np.column_stack((bboxes, scores, labels))
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        
        draw = rgb.copy()
        
        # 跟踪计时
        tracking_start = time.perf_counter()
        person_track = None

        if not self.tracking_enabled:
            for *xyxy, score, cls_id in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                cls = CLASSES_DET[int(cls_id)]
                color = COLORS[cls]
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f'{cls}:{score:.1f}', (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            latest_frame = draw
            tracking_time = (time.perf_counter() - tracking_start) * 1000
        else:
            person_detections = [d for d in detections if len(d) > 5 and d[5] == 0]
            if person_detections:
                largest_person = max(person_detections, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                bboxes = np.array([largest_person[:4]])
                scores = np.array([largest_person[4]])
                labels = np.array([largest_person[5]])
                detections = np.column_stack((bboxes, scores))
            else:
                detections = np.empty((0, 5))
            
            tracks = self.tracker.update(detections if detections.size > 0 else np.empty((0, 5)))
            
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                cls_id = 0
                cls = CLASSES_DET[cls_id]
                color = COLORS[cls]
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f'ID: {track_id}', (x1, y1 - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(draw, (center_x, center_y), 5, (0, 0, 255), -1)
                if person_track is None or (x2-x1)*(y2-y1) > (person_track[2]-person_track[0])*(person_track[3]-person_track[1]):
                    person_track = (center_x, center_y, x1, y1, x2, y2, track_id)
            
            cv2.line(draw, (self.image_center_x, 0), (self.image_center_x, self.H), (255, 0, 0), 1)
            cv2.line(draw, (0, self.image_center_y), (self.W, self.image_center_y), (255, 0, 0), 1)
            
            if person_track:
                if self.scanning:
                    self.scanning = False
                    tracking_data["status"] = "Target found - Tracking"
                else:
                    tracking_data["status"] = "Tracking"
                self.adjust_servos(person_track)
            else:
                if not self.scanning:
                    tracking_data["status"] = "No person - Starting scan"
                    self.start_scanning()
                else:
                    tracking_data["status"] = "Scanning for target"
                    self.perform_scan()
                if self.scanning:
                    if abs(self.current_pos[2] - self.servo_mid[2]) > 10:
                        self.current_pos[2] = self.servo_mid[2]
                        self.move_servo(2, self.current_pos[2], self.idle_speed)
                    if abs(self.current_pos[3] - self.servo_mid[3]) > 10:
                        self.current_pos[3] = self.servo_mid[3]
                        self.move_servo(3, self.current_pos[3], self.idle_speed)
            tracking_time = (time.perf_counter() - tracking_start) * 1000
        
        # 在图像上显示时间
        cv2.putText(draw, f"Preprocess: {preprocess_time:.2f} ms", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(draw, f"Inference: {inference_time:.2f} ms", (10, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(draw, f"Postprocess: {postprocess_time:.2f} ms", (10, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(draw, f"Tracking: {tracking_time:.2f} ms", (10, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 其他信息
        cv2.putText(draw, f"FPS: {self.avg_fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(draw, f"Status: {tracking_data['status']}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        servo_info = f"Servo1: {self.current_pos[1]}, Servo2: {self.current_pos[2]}, Servo3: {self.current_pos[3]}"
        cv2.putText(draw, servo_info, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        latest_frame = draw
        return person_track is not None

    def start_scanning(self):
        self.scanning = True
        self.scan_direction = -1 if self.current_pos[1] >= 500 else 1
        self.last_scan_time = time.time()
        self.move_servo(2, self.servo_mid[2], self.idle_speed)
        self.move_servo(3, self.servo_mid[3], self.idle_speed)
    
    def perform_scan(self):
        current_time = time.time()
        if current_time - self.last_scan_time < self.scan_interval:
            return
        
        self.last_scan_time = current_time
        step = 10  # 减小步长到5
        new_pos = self.current_pos[1] + step * self.scan_direction
        
        # 边界处理
        if new_pos <= 0:
            new_pos = 0
            self.scan_direction = 1
            self.last_scan_time += 0.5  # 在边界暂停0.5秒
        elif new_pos >= 1000:
            new_pos = 1000
            self.scan_direction = -1
            self.last_scan_time += 0.5  # 在边界暂停0.5秒
        
        # 平滑移动
        if abs(new_pos - self.current_pos[1]) > 0.1:  # 确保有足够变化
            self.current_pos[1] = new_pos
            self.move_servo(1, new_pos, self.idle_speed)
    
    def adjust_servos(self, person_track):
        center_x, center_y, x1, y1, x2, y2, _ = person_track
        offset_x = center_x - self.image_center_x
        offset_y = center_y - self.image_center_y
        person_width = x2 - x1
        person_height = y2 - y1
        size_ratio = (person_width * person_height) / (self.W * self.H)
        
        if abs(offset_x) > self.x_threshold:
            step = int(abs(offset_x) / self.x_threshold * 5)
            if offset_x > 0:
                self.current_pos[1] = min(1000, self.current_pos[1] - step)
            else:
                self.current_pos[1] = max(0, self.current_pos[1] + step)
            self.move_servo(1, self.current_pos[1], self.angle_speed)
        
        if abs(offset_y) > self.y_threshold:
            step = int(abs(offset_y) / self.y_threshold * 5)
            if offset_y > 0:
                self.current_pos[3] = min(600, self.current_pos[3] + step)
            else:
                self.current_pos[3] = max(400, self.current_pos[3] - step)
            self.move_servo(3, self.current_pos[3], self.angle_speed)
        
        if size_ratio < self.size_threshold * 0.8:
            self.current_pos[2] = max(300, self.current_pos[2] - 5)
            self.move_servo(2, self.current_pos[2], self.angle_speed)
            tracking_data["movement"] = "Moving closer"
            print("Target too small - Moving closer")
        elif size_ratio > self.size_threshold * 1.2:
            self.current_pos[2] = min(700, self.current_pos[2] + 5)
            self.move_servo(2, self.current_pos[2], self.angle_speed)
            tracking_data["movement"] = "Moving away"
            print("Target too large - Moving away")

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    global tracking_data
    return {
        "status": tracking_data["status"],
        "servo_positions": tracking_data["servo_pos"],
        "fps": tracking_data["fps"]
    }

def run_flask():
    app.run(host='0.0.0.0', port=5001, threaded=False)

def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

def main(args: argparse.Namespace) -> None:
    global latest_frame
    
    rospy.init_node('person_tracker', anonymous=True)
    image_pub = rospy.Publisher('/tracker/output_image', Image, queue_size=1)
    
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    print("Stream server started at http://0.0.0.0:5001/video_feed")
    
    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    tracker = PersonTracker(args)
    
    try:
        while not rospy.is_shutdown():
            try:
                ros_image = image_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            rgb = np.ndarray(shape=(ros_image.height, ros_image.width, 3), 
                           dtype=np.uint8, buffer=ros_image.data)
            tracker.process_frame(rgb)
            
            if latest_frame is not None:
                rgb_image = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
                ros_output = Image()
                ros_output.header = ros_image.header
                ros_output.height = rgb_image.shape[0]
                ros_output.width = rgb_image.shape[1]
                ros_output.encoding = "rgb8"
                ros_output.step = rgb_image.shape[1] * 3
                ros_output.data = rgb_image.tobytes()
                image_pub.publish(ros_output)
            
            if is_key_pressed():
                input_str = sys.stdin.readline().strip().lower()
                if input_str == 'q':
                    print("Exit requested, stopping...")
                    tracker.return_all_servos_to_end()
                    rospy.signal_shutdown("User requested shutdown")
                    break
                elif input_str == 'st':
                    tracker.set_tracking_mode(False)
                    print("Stopped tracking - Showing all detections")
                elif input_str == 'bt':
                    tracker.set_tracking_mode(True)
                    print("Started tracking mode")
            
    finally:
        tracker.return_all_servos_to_end()
        print(f"Final FPS: {tracker.avg_fps:.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Engine file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)