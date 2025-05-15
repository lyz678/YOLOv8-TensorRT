#!/usr/bin/env python3

from . import serial_servo
from pwm_servo import PWMServo 
import time

"""
jetmax上的舵机的位置数值范围为0-1000, 对应0-240度
需要注意， 舵机的位置数值和时间数值都需要用整数
"""
def main():
    serial_servo.set_position(1, 600, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到600位置
    time.sleep(1)
    serial_servo.set_position(1, 300, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到300位置
    time.sleep(1)
    serial_servo.set_position(1, 500, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到500位置
    time.sleep(1)
    serial_servo.set_position(2, 600, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到600位置
    time.sleep(1)
    serial_servo.set_position(2, 300, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到300位置
    time.sleep(1)
    serial_servo.set_position(2, 500, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到500位置
    time.sleep(1)
    serial_servo.set_position(3, 300, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到300位置
    time.sleep(1)
    serial_servo.set_position(3, 600, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到600位置
    time.sleep(1)
    serial_servo.set_position(3, 500, 2000) #让id为1的舵机用2000毫秒时间从当前位置运动到500位置
    time.sleep(1)

    # pwm_servo = PWMServo(i2c_port=1, servo_id=1, min_position=0, max_position=180, min_duration=0.02, max_duration=30, deviation=0)
    # pwm_servo.set_position(135, 1) #控制1号舵机用1秒转动到90度位置
    # time.sleep(1)    
    # pwm_servo.set_position(45, 1) #控制1号舵机用1秒转动到90度位置
    # time.sleep(1)
    # pwm_servo.set_position(90, 1) #控制1号舵机用1秒转动到90度位置      
    # time.sleep(1)

if __name__ == "__main__":
    main()