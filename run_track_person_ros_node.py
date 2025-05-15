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
from flask import Flask, Response
import sys
import select
from collections import deque
import rospy
from sensor_msgs.msg import Image
import queue
from std_msgs.msg import Int32MultiArray

def is_key_pressed():
    return select.select([sys.stdin], [], [], 0)[0]

app = Flask(__name__)
latest_frame = None
tracking_data = {"status": "Initializing", "servo_pos": {}}
# 添加图像队列
image_queue = queue.Queue(maxsize=3)

class PersonTracker:
    def __init__(self, args):
        self.engine = TRTEngine(args.engine)
        self.H, self.W = self.engine.inp_info[0].shape[-2:]
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.angle_speed = 100  # 舵机运动角速度(度/秒)
        self.idle_speed = 50  # 扫描状态时的角速度(度/秒)
        
        # 舵机中间位置
        self.servo_mid = {
            1: 500,  # 左右旋转 (0-1000)
            2: 600,  # 前后伸缩 (300-700)
            3: 500   # 上下高度 (400-600)
        }
        
        # 当前舵机位置（初始化为中间位置）
        self.current_pos = {
            1: self.servo_mid[1],
            2: self.servo_mid[2],
            3: self.servo_mid[3],
        }
        
        # 舵机指令队列和线程
        self.servo_queues = {1: deque(maxlen=1), 2: deque(maxlen=1), 3: deque(maxlen=1)}  # 每个舵机一个队列
        self.servo_lock = threading.Lock()
        self.servo_moving = {1: False, 2: False, 3: False}
        
        # 初始化ROS话题发布者
        self.servo_pub = rospy.Publisher('/servo_command', Int32MultiArray, queue_size=10)
        
        # 等待发布者与订阅者连接
        print("等待舵机控制节点连接...")
        while self.servo_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            time.sleep(0.1)
        print("舵机控制节点已连接")
        
        # 启动常驻线程处理舵机指令
        for servo_id in [1, 2, 3]:
            threading.Thread(target=self.servo_command_worker, args=(servo_id,), daemon=True).start()
        
        # 初始化舵机2和3到中间位置
        self.move_servo(2, self.servo_mid[2], self.idle_speed)
        self.move_servo(3, self.servo_mid[3], self.idle_speed)
        while any(self.servo_moving.values()):  # 等待初始化完成
            time.sleep(0.1)
        
        # 图像中心坐标
        self.image_center_x = self.W // 2
        self.image_center_y = self.H // 2
        
        # 控制参数
        self.x_threshold = self.W * 0.1  # 水平方向阈值
        self.y_threshold = self.H * 0.1  # 垂直方向阈值
        self.size_threshold = 0.3  # 大小阈值(占画面比例)
        
        
        # 扫描状态
        self.scanning = True
        self.scan_direction = -1  # -1表示向左(0方向), 1表示向右(1000方向)
        self.last_scan_time = time.time()
        self.scan_interval = 0.1  # 扫描间隔(秒)
        
        # 追踪模式标志
        self.tracking_enabled = True
        
        # FPS计算相关
        self.frame_times = deque(maxlen=10)  # 保存最近10帧的时间戳
        self.avg_fps = 0
    
    def move_servo(self, servo_id, target_position, angle_speed):
        """
        将舵机运动指令添加到队列，支持指令覆盖
        """
        # 将指令添加到队列（覆盖旧指令）
        self.servo_queues[servo_id].append((target_position, angle_speed))

    def servo_command_worker(self, servo_id):
        """
        常驻线程，处理舵机指令队列
        """
        while True:
            if not self.servo_queues[servo_id]:  # 队列为空，等待
                time.sleep(0.01)
                continue

            # 获取最新指令
            target_position, angle_speed = self.servo_queues[servo_id].popleft()
            
            # 发送指令
            with self.servo_lock:
                self.servo_moving[servo_id] = True
            
            try:
                # 发布ROS话题
                msg = Int32MultiArray()
                msg.data = [servo_id, target_position, angle_speed]
                self.servo_pub.publish(msg)
                
                # 更新位置（假设指令发送成功）
                with self.servo_lock:
                    self.current_pos[servo_id] = target_position
                    tracking_data["servo_pos"][f"servo_{servo_id}"] = target_position
                print(f"舵机{servo_id}指令已发送: 目标位置={target_position}, 速度={angle_speed}")
            
            except Exception as e:
                print(f"舵机{servo_id}指令发送失败: {str(e)}")
            
            finally:
                with self.servo_lock:
                    self.servo_moving[servo_id] = False

    def return_all_servos_to_end(self):
        """将所有舵机移到结束位置"""
        print("\nReturning all servos to end position...")
        self.move_servo(1, 500, 50)  # 左右旋转
        self.move_servo(2, 700, 50)  # 前后伸缩
        self.move_servo(3, 550, 50)  # 上下高度
        
        # 等待所有舵机指令发送完成
        while any(self.servo_moving.values()):
            time.sleep(0.1)
        print("All servos returned to end position")

    def set_tracking_mode(self, enabled):
        """设置追踪模式"""
        self.tracking_enabled = enabled
        if enabled:
            self.scanning = True
            tracking_data["status"] = "Tracking enabled - Scanning for target"
        else:
            tracking_data["status"] = "Tracking disabled - Showing all detections"
    
    def update_fps(self):
        """更新FPS计算"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.avg_fps = (len(self.frame_times) - 1) / time_diff
            else:
                self.avg_fps = 0
        else:
            self.avg_fps = 0
    
    def process_frame(self, rgb):
        global latest_frame, tracking_data
        
        # 更新FPS计算
        self.update_fps()
        
        # 预处理
        rgb, ratio, dwdh = letterbox(rgb, (self.W, self.H))
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        
        # 推理
        data = self.engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        
        # 后处理
        if bboxes.size == 0:
            detections = np.array([])
        else:
            bboxes -= dwdh
            bboxes /= ratio
            detections = np.column_stack((bboxes, scores, labels))
        
        # 绘制结果
        draw = rgb.copy()
        
        if not self.tracking_enabled:
            # 显示所有检测结果
            for *xyxy, score, cls_id in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                cls = CLASSES_DET[int(cls_id)]
                color = COLORS[cls]
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f'{cls}:{score:.1f}', (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # 显示FPS
            cv2.putText(draw, f"FPS: {self.avg_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # 显示当前模式状态
            cv2.putText(draw, f"Mode: Detection Only", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # 更新全局帧
            latest_frame = draw
            return False
        
        # 追踪逻辑
        # 只保留人物检测结果(人物类别为0)
        person_detections = [d for d in detections if len(d) > 5 and d[5] == 0]
        
        if len(person_detections) > 0:
            # 选择最大的bbox(按面积)
            largest_person = max(person_detections, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
            bboxes = np.array([largest_person[:4]])
            scores = np.array([largest_person[4]])
            labels = np.array([largest_person[5]])
            detections = np.column_stack((bboxes, scores))
        else:
            detections = np.empty((0, 5))
        
        # 跟踪
        tracks = self.tracker.update(detections if detections.size > 0 else np.empty((0, 5)))
        
        # 绘制结果
        person_track = None
        
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cls_id = 0  # 只跟踪人物
            cls = CLASSES_DET[cls_id]
            color = COLORS[cls]
            
            # 绘制边界框和ID
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, f'ID: {track_id}', (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(draw, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 保存最大的人物跟踪信息
            if person_track is None or (x2-x1)*(y2-y1) > (person_track[2]-person_track[0])*(person_track[3]-person_track[1]):
                person_track = (center_x, center_y, x1, y1, x2, y2, track_id)
        
        # 绘制图像中心线
        cv2.line(draw, (self.image_center_x, 0), (self.image_center_x, self.H), (255, 0, 0), 1)
        cv2.line(draw, (0, self.image_center_y), (self.W, self.image_center_y), (255, 0, 0), 1)
        
        # 更新跟踪状态
        if person_track is not None:
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
            
            # 在扫描模式下，确保舵机2和3保持在中间位置
            if self.scanning:
                if abs(self.current_pos[2] - self.servo_mid[2]) > 10:
                    self.current_pos[2] = self.servo_mid[2]
                    self.move_servo(2, self.current_pos[2], self.idle_speed)
                if abs(self.current_pos[3] - self.servo_mid[3]) > 10:
                    self.current_pos[3] = self.servo_mid[3]
                    self.move_servo(3, self.current_pos[3], self.idle_speed)
        
        # 显示平均FPS
        cv2.putText(draw, f"FPS: {self.avg_fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # 显示跟踪状态
        cv2.putText(draw, f"Status: {tracking_data['status']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # 显示当前舵机位置
        servo_info = f"Servo1: {self.current_pos[1]}, Servo2: {self.current_pos[2]}, Servo3: {self.current_pos[3]}"
        cv2.putText(draw, servo_info, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 更新全局帧
        latest_frame = draw
        
        return person_track is not None
    
    def start_scanning(self):
        """开始扫描模式"""
        self.scanning = True
        self.scan_direction = -1 if self.current_pos[1] >= 500 else 1  # 根据当前位置决定扫描方向
        self.last_scan_time = time.time()
        
        # 确保舵机2和3在中间位置
        self.move_servo(2, self.servo_mid[2], self.idle_speed)
        self.move_servo(3, self.servo_mid[3], self.idle_speed)
    
    def perform_scan(self):
        """执行扫描动作"""
        current_time = time.time()
        if current_time - self.last_scan_time > self.scan_interval:
            self.last_scan_time = current_time
            
            # 计算下一步位置
            step = 20  # 扫描步长
            new_pos = self.current_pos[1] + step * self.scan_direction
            
            # 检查是否到达边界
            if new_pos <= 0:
                new_pos = 0
                self.scan_direction = 1  # 向右转
            elif new_pos >= 1000:
                new_pos = 1000
                self.scan_direction = -1  # 向左转
            
            # 移动舵机，使用 self.idle_speed 作为速度
            self.current_pos[1] = new_pos
            self.move_servo(1, new_pos, self.idle_speed)  # 修改速度参数
    
    def adjust_servos(self, person_track):
        center_x, center_y, x1, y1, x2, y2, track_id = person_track
        
        # 计算与中心的偏移量
        offset_x = center_x - self.image_center_x
        offset_y = center_y - self.image_center_y
        
        # 计算人物大小比例
        person_width = x2 - x1
        person_height = y2 - y1
        size_ratio = (person_width * person_height) / (self.W * self.H)
        
        # 调整左右旋转(1号舵机)
        if abs(offset_x) > self.x_threshold:
            # 根据偏移方向调整舵机
            step = int(abs(offset_x) / self.x_threshold * 5)  # 步进值
            if offset_x > 0:  # 人物在右侧，需要向右转
                self.current_pos[1] = min(1000, self.current_pos[1] - step)
            else:  # 人物在左侧，需要向左转
                self.current_pos[1] = max(0, self.current_pos[1] + step)
            self.move_servo(1, self.current_pos[1], self.angle_speed)
        
        # 调整上下高度(3号舵机)
        if abs(offset_y) > self.y_threshold:
            step = int(abs(offset_y) / self.y_threshold * 5)
            if offset_y > 0:  # 人物在下侧，需要降低高度
                self.current_pos[3] = min(600, self.current_pos[3] + step)
            else:  # 人物在上侧，需要增加高度
                self.current_pos[3] = max(400, self.current_pos[3] - step)
            self.move_servo(3, self.current_pos[3], self.angle_speed)
        
        # 调整前后伸缩(2号舵机) - 基于人物大小
        if size_ratio < self.size_threshold * 0.8:  # 人物太小，需要靠近
            self.current_pos[2] = max(300, self.current_pos[2] - 5)
            self.move_servo(2, self.current_pos[2], self.angle_speed)
            tracking_data["movement"] = "Moving closer"
            print("Target too small - Moving closer")
        elif size_ratio > self.size_threshold * 1.2:  # 人物太大，需要远离
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
        "fps": tracking_data.get("fps", 0)
    }

def run_flask():
    app.run(host='0.0.0.0', port=5001, threaded=False)

# ROS图像回调函数
def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

def main(args: argparse.Namespace) -> None:
    global latest_frame
    
    # 初始化ROS节点
    rospy.init_node('person_tracker', anonymous=True)
    # 创建图像发布者
    global image_pub
    image_pub = rospy.Publisher('/tracker/output_image', Image, queue_size=1)
    
    # 启动Flask服务器
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    print("Stream server started at http://0.0.0.0:5001/video_feed")
    
    # 订阅摄像头画面
    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    
    tracker = PersonTracker(args)
    
    try:
        while not rospy.is_shutdown():
            # 从队列中获取图像
            try:
                ros_image = image_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            # 将ROS图像转换为OpenCV格式
            rgb = np.ndarray(shape=(ros_image.height, ros_image.width, 3), 
                           dtype=np.uint8, 
                           buffer=ros_image.data)            
            # 处理帧
            has_person = tracker.process_frame(rgb)
            
            # 将处理后的图像转换回ROS格式并发布
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
            
            # 监听键盘输入
            if is_key_pressed():
                input_str = sys.stdin.readline().strip().lower()
                if input_str == 'q':
                    print("Exit requested, stopping...")
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