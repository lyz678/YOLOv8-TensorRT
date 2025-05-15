import time
from . import serial_servo


def move_servo(servo_id, target_position, angle_speed):
    """
    让指定ID的舵机以统一的角速度转到目标位置。
    :param servo_id: 舵机 ID (1: 左右旋转, 2: 前后伸缩, 3: 上下高度)
    :param target_position: 目标位置
    """
    # 读取当前舵机位置
    current_position = serial_servo.read_pos(servo_id)
    
    # 计算当前角度
    current_angle = current_position / 1000 * 240
    # 计算目标角度
    target_angle = target_position / 1000 * 240
    # 计算角度差
    angle_diff = target_angle - current_angle
    # 计算所需时间（单位：秒），确保时间为正
    move_time = abs(angle_diff / angle_speed)
    
    # 发送舵机运动指令（转换为毫秒）
    serial_servo.set_position(servo_id, int(target_position), int(move_time * 1000))
    
    # 等待运动完成
    time.sleep(move_time)

def main():

    # 设定统一的角速度（度/秒）
    angle_speed = 30  

    # 让 1 号舵机左右旋转 posion 0-1000
    # move_servo(1, 0, angle_speed)
    # move_servo(1, 500, angle_speed)
    # move_servo(1, 1000, angle_speed)
    # move_servo(1, 500, angle_speed)

    # 让 2 号舵机前后伸缩 posion 300-600
    # move_servo(2, 300, angle_speed)
    # move_servo(2, 500, angle_speed)
    # move_servo(2, 700, angle_speed)
    # move_servo(2, 500, angle_speed)

    # 让 3 号舵机上下高度 3号舵机控制上下高度，positon 400-600
    move_servo(3, 400, angle_speed)
    move_servo(3, 500, angle_speed)
    move_servo(3, 600, angle_speed)
    move_servo(3, 500, angle_speed)

if __name__ == "__main__":
    main()

#sudo chmod 666 /dev/ttyTHS1
#python3 -m JetMax.serial_servo_speed
