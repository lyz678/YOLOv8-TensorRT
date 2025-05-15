import cv2
import numpy as np
import time
import math

# -----------------------------------双目相机的基本参数---------------------------------------------------------
left_camera_matrix = np.array([[516.5066236, -1.444673028, 320.2950423], [0, 516.5816117, 270.7881873], [0., 0., 1.]])
right_camera_matrix = np.array([[511.8428182, 1.295112628, 317.310253], [0, 513.0748795, 269.5885026], [0., 0., 1.]])

# 畸变系数
left_distortion = np.array([[-0.046645194, 0.077595167, 0.012476819, -0.000711358, 0]])
right_distortion = np.array([[-0.061588946, 0.122384376, 0.011081232, -0.000750439, 0]])

# 旋转矩阵和平移矩阵
R = np.array([[0.999911333, -0.004351508, 0.012585312],
              [0.004184066, 0.999902792, 0.013300386],
              [-0.012641965, -0.013246549, 0.999832341]])
T = np.array([-120.3559901, -0.188953775, -0.662073075])

size = (640, 480)

# 校正映射
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

# 鼠标回调函数
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")
        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2) / 1000.0
        print("距离是：", distance, "m")

# 加载视频文件
capture = cv2.VideoCapture("./car.avi")
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("depth", cv2.WINDOW_AUTOSIZE)

# ------------------------------------BM算法----------------------------------------------------------
#   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，必须为奇数
#   numDisparities              BM感知的范围，越大生成的精度越好，速度越慢，必须被16整除
#   minDisparity                最小视差，默认为0
#   uniquenessRatio             唯一性比率，用于过滤错误匹配
#   speckleWindowSize           斑点窗口大小，用于去除小斑点
#   speckleRange                斑点范围，用于斑点检测
# ------------------------------------------------------------------------------------------------------
blockSize = 5  # 必须为奇数
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=blockSize)
stereo.setMinDisparity(1)
stereo.setUniquenessRatio(10)
stereo.setSpeckleWindowSize(100)
stereo.setSpeckleRange(100)

# 读取视频
fps = 0.0
ret, frame = capture.read()
while ret:
    t1 = time.time()

    ret, frame = capture.read()
    t_read = time.time()

    if not ret:
        break

    # 切割为左右两张图片
    frame1 = frame[0:480, 0:640]
    frame2 = frame[0:480, 640:1280]
    t_cut = time.time()

    # 转换为灰度图
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    t_gray = time.time()

    # 重映射
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
    t_remap = time.time()

    # 转换为 BGR 格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)
    t_color = time.time()

    # 使用 BM 算法计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)
    t_bm = time.time()

    # 归一化生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)
    t_norm = time.time()

    # 计算三维坐标
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    threeD = threeD * 16
    t_3d = time.time()

    # 鼠标回调
    cv2.setMouseCallback("depth", onmouse_pick_points, threeD)

    # 计算帧率和各模块耗时
    fps = (fps + (1. / (time.time() - t1))) / 2
    time_read = t_read - t1
    time_cut = t_cut - t_read
    time_gray = t_gray - t_cut
    time_remap = t_remap - t_gray
    time_color = t_color - t_remap
    time_bm = t_bm - t_color
    time_norm = t_norm - t_bm
    time_3d = t_3d - t_norm
    total_time = time.time() - t1

    # 显示耗时和帧率
    text = f"FPS: {fps:.2f}\nRead: {time_read:.4f}s\nCut: {time_cut:.4f}s\nGray: {time_gray:.4f}s\nRemap: {time_remap:.4f}s\nColor: {time_color:.4f}s\nBM: {time_bm:.4f}s\nNorm: {time_norm:.4f}s\n3D: {time_3d:.4f}s\nTotal: {total_time:.4f}s"
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (10, 50 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("depth", dis_color)
    cv2.imshow("left", frame1)
    cv2.imshow(WIN_NAME, disp)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# 释放资源
capture.release()
cv2.destroyAllWindows()