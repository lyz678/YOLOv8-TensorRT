#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import threading
import sys
import select
import termios
import tty
import cv2
import numpy as np
from models.pycuda_api import TRTEngine
from config import CLASSES_DET, COLORS
from models.utils import blob, det_postprocess, letterbox
from collections import deque
from flask import Flask, Response
import queue
import rospy
from sensor_msgs.msg import Image
from hiwonder import serial_servo, Sucker, JetMax

# Global variables
latest_frame = None
tracking_data = {"status": "Initializing", "servo_pos": {"servo_1": 500, "servo_2": 500, "servo_3": 500}, "fps": 0}
image_queue = queue.Queue(maxsize=3)
servo_lock = threading.Lock()

def is_key_pressed():
    return select.select([sys.stdin], [], [], 0)[0]

app = Flask(__name__)

class PersonTracker:
    def __init__(self, args):
        self.engine = TRTEngine(args.engine)
        self.H, self.W = self.engine.inp_info[0].shape[-2:]
        self.angle_speed = 100  # Servo angular speed (degrees/second)
        self.sucker = Sucker()  # Initialize sucker
        self.jetmax = JetMax()

        # Servo middle positions
        self.servo_mid = {1: 500, 2: 500, 3: 500}
        
        # Image center
        self.image_center_x = self.W // 2
        self.image_center_y = self.H // 2
        
        # Control parameters
        self.x_threshold = self.W * 0.1
        self.y_threshold = self.H * 0.1
        self.size_threshold = 0.3
        
        # Tracking mode
        self.tracking_enabled = True
        
        # FPS calculation
        self.frame_times = deque(maxlen=10)
        self.avg_fps = 0
        
        # Key control
        self.running = True
        self.servo_step = 10  # Step size for movement
        self.key_states = {}  # Track key states
        
        # Set terminal to non-buffered mode
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        # Initialize servos to middle position
        for servo_id in [1, 2, 3]:
            if not self.move_servo(servo_id, self.servo_mid[servo_id], self.angle_speed):
                print(f"\033[31mWarning: Servo {servo_id} initialization failed\033[0m")
        
        # Start keyboard control thread
        threading.Thread(target=self.keyboard_control, daemon=True).start()
    
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
                current_position = serial_servo.read_pos(servo_id)
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
                serial_servo.set_position(servo_id, target_position, command_time_ms)
                print(f"Command sent - Servo ID: {servo_id}, Current: {current_position}, Target: {target_position}, Time: {command_time_ms}ms")
            except Exception as e:
                print(f"\033[31mError: Cannot set servo {servo_id} position: {e}\033[0m")
                return False
                
            # Update tracking data
            tracking_data["servo_pos"][f"servo_{servo_id}"] = target_position
            
            # Wait for movement to complete
            time.sleep(move_time)
            after_moving_position = serial_servo.read_pos(servo_id)
            print(f"Servo ID: {servo_id}, After moving Position: {after_moving_position}, Target: {target_position}")
            return True

    def keyboard_control(self):
        """Handle key press and release for servo and sucker control"""
        key_map = {
            'a': (1, lambda pos: pos - self.servo_step),  # Left
            'd': (1, lambda pos: pos + self.servo_step),  # Right
            'w': (3, lambda pos: pos - self.servo_step),  # Up
            's': (3, lambda pos: pos + self.servo_step),  # Down
            'r': (2, lambda pos: pos + self.servo_step),  # Forward
            'f': (2, lambda pos: pos - self.servo_step)   # Backward
        }
        
        while self.running:
            if is_key_pressed():
                try:
                    key = sys.stdin.read(1).lower()
                    if key in key_map:
                        if key not in self.key_states:
                            self.key_states[key] = True
                            servo_id, pos_func = key_map[key]
                            current_pos = tracking_data["servo_pos"][f"servo_{servo_id}"]
                            target_pos = pos_func(current_pos)
                            
                            # Restrict target position within range
                            min_pos = 0
                            max_pos = 1000
                            target_pos = max(min_pos, min(target_pos, max_pos))
                            
                            # Move only if position changes
                            if abs(target_pos - current_pos) > 1:
                                threading.Thread(
                                    target=self.move_servo,
                                    args=(servo_id, target_pos, self.angle_speed),
                                    daemon=True
                                ).start()
                    elif key == 't':
                        if 't' not in self.key_states:
                            self.key_states['t'] = True
                            self.sucker.suck(3)
                            print("Sucker activated")
                    elif key == 'g':
                        if 'g' not in self.key_states:
                            self.key_states['g'] = True
                            self.sucker.release(3)
                            print("Sucker released")
                    elif key == 'q':
                        self.running = False
                    elif key == 'st':
                        self.set_tracking_mode(False)
                        print("Tracking stopped - Showing all detections")
                    elif key == 'bt':
                        self.set_tracking_mode(True)
                        print("Tracking mode started")
                except Exception as e:
                    print(f"\033[31mKey handling error: {e}\033[0m")
            else:
                self.key_states.clear()  # Clear key states
            time.sleep(0.01)  # Reduce CPU usage
    
    def return_all_servos_to_mid(self):
        print("\nReturning all servos to middle position...")
        for servo_id in [1, 2, 3]:
            self.move_servo(servo_id, self.servo_mid[servo_id], 50)
        print("All servos returned to middle position")

    def return_all_servos_to_end(self):
        print("\nReturning all servos to end position...")
        self.move_servo(1, 500, 50)
        self.move_servo(2, 700, 50)
        self.move_servo(3, 550, 50)
        print("All servos returned to end position")

    def set_tracking_mode(self, enabled):
        self.tracking_enabled = enabled
        if enabled:
            tracking_data["status"] = "Tracking Enabled"
        else:
            tracking_data["status"] = "Tracking Disabled - Showing All Detections"
    
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
        
        # Start total timing
        total_start = time.perf_counter()
        
        self.update_fps()
        
        # Preprocessing timing
        preprocess_start = time.perf_counter()
        rgb, ratio, dwdh = letterbox(rgb, (self.W, self.H))
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Inference timing
        inference_start = time.perf_counter()
        data = self.engine(tensor)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Postprocessing timing
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
        
        # Tracking timing
        tracking_start = time.perf_counter()

        for *xyxy, score, cls_id in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            cls = CLASSES_DET[int(cls_id)]
            color = COLORS[cls]
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, f'{cls}:{score:.1f}', (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        cv2.line(draw, (self.image_center_x, 0), (self.image_center_x, self.H), (255, 0, 0), 1)
        cv2.line(draw, (0, self.image_center_y), (self.W, self.image_center_y), (255, 0, 0), 1)
        
        if detections.size > 0:
            tracking_data["status"] = "Targets Detected"
        else:
            tracking_data["status"] = "No Targets Detected"
        
        tracking_time = (time.perf_counter() - tracking_start) * 1000
        
        # Display timing on image
        cv2.putText(draw, f"Preprocess: {preprocess_time:.2f} ms", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(draw, f"Inference: {inference_time:.2f} ms", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(draw, f"Postprocess: {postprocess_time:.2f} ms", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(draw, f"Tracking: {tracking_time:.2f} ms", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Other information
        cv2.putText(draw, f"FPS: {self.avg_fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(draw, f"Status: {tracking_data['status']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        servo_info = f"Servo1: {tracking_data['servo_pos']['servo_1']}, Servo2: {tracking_data['servo_pos']['servo_2']}, Servo3: {tracking_data['servo_pos']['servo_3']}"
        cv2.putText(draw, servo_info, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        latest_frame = draw
        return detections.size > 0

    def cleanup(self):
        # Release sucker
        self.sucker.release(3)
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

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
    app.run(host='0.0.0.0', port=5000, threaded=False)

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
    print("Stream server started: http://0.0.0.0:5001/video_feed")
    
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
            
            # Main loop no longer handles keys, relies on keyboard control thread
            
    finally:
        tracker.running = False
        tracker.return_all_servos_to_end()
        tracker.cleanup()
        print(f"Final FPS: {tracker.avg_fps:.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Engine file path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)