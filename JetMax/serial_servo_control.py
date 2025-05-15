#!/usr/bin/env python3

from . import serial_servo, serial_servo_io
import time

"""
jetmax上的舵机的位置数值范围为0～1000, 对应0～240度
需要注意， 舵机的位置数值和时间数值都需要用整数
"""

def main():
    
    serial_servo.set_position(1, 500, 3000) #让id为1的舵机用8000毫秒时间从当前位置运动到0位置
    time.sleep(3)

    pos = serial_servo.read_pos(1)
    print('Position：%d' % pos)
    

    serial_servo.set_position(1, 0, 3000) #让id为1的舵机用8000毫秒时间从当前位置运动到0位置
    time.sleep(3)

    pos = serial_servo.read_pos(1)
    print('Position：%d' % pos)
    

    serial_servo.set_position(1, 500, 3000) #让id为1的舵机用5000毫秒时间从当前位置运动到1000位置
    time.sleep(3)

    pos = serial_servo.read_pos(1)
    print('Position：%d' % pos)
    

    serial_servo.set_position(1, 1000, 3000) #让id为1的舵机用5000毫秒时间从当前位置运动到1000位置
    time.sleep(3)

    pos = serial_servo.read_pos(1)
    print('Position：%d' % pos)
    

    #如果我们需要用角度来控制的话可以将及角度转化为数值例如现在要转转动到180度位置
    p = 180 / 240 * 1000
    serial_servo.set_position(1, int(p), 3000) #这就是用4000毫秒从当前位置运动到 180度位置
    time.sleep(3)

    pos = serial_servo.read_pos(1)
    print('Position：%d' % pos)
    

    serial_servo.set_position(1, int(120 / 240 * 1000), 3000) #这就是用2000毫秒从当前位置运动到 120度位置
    time.sleep(3)

    pos = serial_servo.read_pos(1)
    print('Position：%d' % pos)
    

    # serial_servo.set_position(2, 300, 3000) #让id为1的舵机用8000毫秒时间从当前位置运动到0位置
    # time.sleep(3)
    # serial_servo.set_position(2, 500, 3000) #让id为1的舵机用5000毫秒时间从当前位置运动到1000位置
    # time.sleep(3)
    # serial_servo.set_position(2, 600, 3000) #让id为1的舵机用5000毫秒时间从当前位置运动到1000位置
    # time.sleep(3)
    # #如果我们需要用角度来控制的话可以将及角度转化为数值例如现在要转转动到180度位置
    # p = 180 / 240 * 1000
    # serial_servo.set_position(2, int(p), 3000) #这就是用4000毫秒从当前位置运动到 180度位置
    # time.sleep(3)
    # serial_servo.set_position(2, int(120 / 240 * 1000), 3000) #这就是用2000毫秒从当前位置运动到 120度位置
    # time.sleep(3)

    # serial_servo.set_position(3, 400, 3000) #让id为1的舵机用8000毫秒时间从当前位置运动到0位置
    # time.sleep(3)
    # serial_servo.set_position(3, 500, 3000) #让id为1的舵机用5000毫秒时间从当前位置运动到1000位置
    # time.sleep(3)
    # serial_servo.set_position(3, 600, 3000) #让id为1的舵机用5000毫秒时间从当前位置运动到1000位置
    # time.sleep(3)
    # #如果我们需要用角度来控制的话可以将及角度转化为数值例如现在要转转动到180度位置
    # p = 180 / 240 * 1000
    # serial_servo.set_position(3, int(p), 3000) #这就是用4000毫秒从当前位置运动到 180度位置
    # time.sleep(3)
    # serial_servo.set_position(3, int(120 / 240 * 1000), 3000) #这就是用2000毫秒从当前位置运动到 120度位置
    # time.sleep(3)


if __name__ == "__main__":
    main()

#python3 -m JetMax.serial_servo_control

