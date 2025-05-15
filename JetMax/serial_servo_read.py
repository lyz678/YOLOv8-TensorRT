#!/usr/bin/env python3

from . import serial_servo
import time

"""
总线舵机支持读取舵机的多个参数及状态
"""

def main():
    while True:
        # id_ = 1
        # print("ID: %d" %  id_)
        # pos = serial_servo.read_position(id_)
        # print('Position：%d' % pos)
        # now_temp = serial_servo.read_temp(id_)
        # print('Temperature：%d°' % now_temp)
        # now_vin = serial_servo.read_vin(id_)
        # print('Voltage input：%dmv' % now_vin)
        # d = serial_servo.read_deviation(id_)
        # print('Deviation：%d' % d)
        # limit = serial_servo.read_angle_limit(id_)
        # print('Position range:%d-%d' % (limit[0], limit[1]))
        # vin = serial_servo.read_vin_limit(id_)
        # print('Voltage range:%dmv-%dmv' % (vin[0], vin[1]))
        # temp = serial_servo.read_temp_limit(id_)
        # print('Temperature limit:50°-%d°' % temp)
        # print("\n\n")
        # time.sleep(1)
        serial_servo.show_servo_state()


if __name__ == "__main__":
    main()
