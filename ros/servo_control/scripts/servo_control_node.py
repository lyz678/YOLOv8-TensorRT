#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int32MultiArray
from hiwonder import serial_servo

class ServoController:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('servo_control_node', anonymous=True)
        
        # 舵机位置限制
        self.pos_limits = {
            1: (0, 1000),    # 左右旋转
            2: (300, 700),   # 前后伸缩
            3: (400, 600)    # 上下高度
        }      
        
        # 订阅舵机指令话题
        rospy.Subscriber('/servo_command', Int32MultiArray, self.servo_command_callback)
        
        rospy.loginfo("舵机控制节点已启动")
        rospy.spin()

    def servo_command_callback(self, msg):
        if len(msg.data) != 3:
            rospy.logerr("无效的舵机指令格式")
            return
        
        servo_id, target_pos, speed = msg.data
        
        # 检查输入参数有效性
        if servo_id not in [1, 2, 3]:
            rospy.logerr(f"无效的舵机ID: {servo_id}")
            return

        min_pos, max_pos = self.pos_limits[servo_id]
        if not (min_pos <= target_pos <= max_pos):
            rospy.logerr(f"目标位置超出范围[{min_pos}, {max_pos}]")
            return

        if speed <= 0:
            rospy.logerr("速度必须大于0")
            return

        try:
            # 读取当前位置
            current_pos = serial_servo.read_pos(servo_id)
            if current_pos == -1:
                rospy.logwarn(f"舵机{servo_id}读取当前位置失败，使用默认值")
                current_pos = (min_pos + max_pos) // 2

            # 计算运动参数
            current_angle = current_pos / 1000 * 240
            target_angle = target_pos / 1000 * 240
            angle_diff = abs(target_angle - current_angle)
            move_time = angle_diff / speed
            
            # 发送运动指令
            serial_servo.set_position(servo_id, target_pos, int(move_time * 1000))
            rospy.loginfo(f"Sending command - Servo ID: {servo_id}, Current position: {current_pos}, Target position: {target_pos}, Move speed: {speed}, Move time: {int(move_time * 1000)}ms")
        
        except Exception as e:
            rospy.logerr(f"移动舵机{servo_id}失败: {str(e)}")

if __name__ == "__main__":
    try:
        ServoController()
    except rospy.ROSInterruptException:
        pass