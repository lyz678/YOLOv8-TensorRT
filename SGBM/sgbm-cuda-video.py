import cv2
import numpy as np
import time
import random
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

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

# 参数设置
blockSize = 5  # 需为奇数，推荐 5-21
halfBlockSize = blockSize // 2
numDisparities = 64  # 必须是 16 的倍数
minDisparity = 1
height, width = 480, 640  # 图像尺寸
P1 = 8 * 1 * blockSize * blockSize  # SGBM 平滑惩罚项 P1（单通道图像）
P2 = 32 * 1 * blockSize * blockSize  # SGBM 平滑惩罚项 P2

# CUDA 内核代码（SGBM 简化版：4 方向半全局优化）
mod = SourceModule("""
texture<unsigned char, 2, cudaReadModeElementType> imgL_tex;
texture<unsigned char, 2, cudaReadModeElementType> imgR_tex;

__global__ void stereoSGBM(
    short *disparity, int width, int height, 
    int blockSize, int halfBlockSize, int numDisparities, int minDisparity,
    int P1, int P2) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < halfBlockSize || x >= width - halfBlockSize || y < halfBlockSize || y >= height - halfBlockSize) {
        return;  // 跳过边界像素
    }

    // 动态规划路径代价数组
    const int maxDisp = minDisparity + numDisparities;
    int cost[64];  // 假设 numDisparities <= 64
    int Lr[4][64];  // 4 个方向的路径代价：左、右、上、下

    // 第一步：计算初始匹配代价（SAD）
    for (int d = minDisparity; d < maxDisp; d++) {
        int sad = 0;
        for (int dy = -halfBlockSize; dy <= halfBlockSize; dy++) {
            for (int dx = -halfBlockSize; dx <= halfBlockSize; dx++) {
                int leftVal = tex2D(imgL_tex, x + dx + 0.5f, y + dy + 0.5f);
                int rightVal = (x + dx - d >= 0) ? tex2D(imgR_tex, x + dx - d + 0.5f, y + dy + 0.5f) : 0;
                sad += abs(leftVal - rightVal);
            }
        }
        cost[d - minDisparity] = sad;
    }

    // 第二步：半全局优化（4 个方向：左、右、上、下）
    for (int d = 0; d < numDisparities; d++) {
        Lr[0][d] = cost[d];  // 左方向
        Lr[1][d] = cost[d];  // 右方向
        Lr[2][d] = cost[d];  // 上方向
        Lr[3][d] = cost[d];  // 下方向
    }

    // 左方向
    if (x > halfBlockSize) {
        for (int d = 0; d < numDisparities; d++) {
            int minPrev = 999999;
            for (int prevD = 0; prevD < numDisparities; prevD++) {
                int penalty = (abs(d - prevD) == 1) ? P1 : (abs(d - prevD) > 1 ? P2 : 0);
                int prevCost = Lr[0][prevD] + penalty;
                minPrev = min(minPrev, prevCost);
            }
            Lr[0][d] = cost[d] + minPrev - min(minPrev, cost[d]);
        }
    }

    // 右方向
    if (x < width - halfBlockSize - 1) {
        for (int d = 0; d < numDisparities; d++) {
            int minPrev = 999999;
            for (int prevD = 0; prevD < numDisparities; prevD++) {
                int penalty = (abs(d - prevD) == 1) ? P1 : (abs(d - prevD) > 1 ? P2 : 0);
                int prevCost = Lr[1][prevD] + penalty;
                minPrev = min(minPrev, prevCost);
            }
            Lr[1][d] = cost[d] + minPrev - min(minPrev, cost[d]);
        }
    }

    // 上方向
    if (y > halfBlockSize) {
        for (int d = 0; d < numDisparities; d++) {
            int minPrev = 999999;
            for (int prevD = 0; prevD < numDisparities; prevD++) {
                int penalty = (abs(d - prevD) == 1) ? P1 : (abs(d - prevD) > 1 ? P2 : 0);
                int prevCost = Lr[2][prevD] + penalty;
                minPrev = min(minPrev, prevCost);
            }
            Lr[2][d] = cost[d] + minPrev - min(minPrev, cost[d]);
        }
    }

    // 下方向
    if (y < height - halfBlockSize - 1) {
        for (int d = 0; d < numDisparities; d++) {
            int minPrev = 999999;
            for (int prevD = 0; prevD < numDisparities; prevD++) {
                int penalty = (abs(d - prevD) == 1) ? P1 : (abs(d - prevD) > 1 ? P2 : 0);
                int prevCost = Lr[3][prevD] + penalty;
                minPrev = min(minPrev, prevCost);
            }
            Lr[3][d] = cost[d] + minPrev - min(minPrev, cost[d]);
        }
    }

    // 第三步：合并路径代价并选择最佳视差
    int bestDisparity = 0;
    int minCost = 999999;
    for (int d = 0; d < numDisparities; d++) {
        int totalCost = Lr[0][d] + Lr[1][d] + Lr[2][d] + Lr[3][d];
        if (totalCost < minCost) {
            minCost = totalCost;
            bestDisparity = d + minDisparity;
        }
    }

    // 存储视差结果
    disparity[y * width + x] = bestDisparity * 16;  // 视差缩放（与 OpenCV 一致）
}
""", options=["--use_fast_math"], arch="sm_53")

# 获取 CUDA 内核和纹理引用
stereoSGBM = mod.get_function("stereoSGBM")
imgL_tex = mod.get_texref("imgL_tex")
imgR_tex = mod.get_texref("imgR_tex")

# 设置线程块和网格大小
block = (16, 16, 1)  # 每个线程块 16x16
grid_x = (width + block[0] - 1) // block[0]
grid_y = (height + block[1] - 1) // block[1]
grid = (grid_x, grid_y, 1)

# 分配 GPU 内存（一次性分配）
imgL_array = np.zeros((height, width), dtype=np.uint8)
imgR_array = np.zeros((height, width), dtype=np.uint8)
imgL_gpu = cuda.mem_alloc(width * height * np.uint8().nbytes)
imgR_gpu = cuda.mem_alloc(width * height * np.uint8().nbytes)
disparity = np.zeros((height, width), dtype=np.int16)
disparity_gpu = cuda.mem_alloc(disparity.nbytes)

# 绑定纹理内存（初始绑定，需在循环中更新）
cuda.bind_array_to_texref(cuda.matrix_to_array(imgL_array, "C"), imgL_tex)
cuda.bind_array_to_texref(cuda.matrix_to_array(imgR_array, "C"), imgR_tex)

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

    # 上传到 GPU 并绑定到纹理内存
    cuda.memcpy_htod(imgL_gpu, img1_rectified)
    cuda.memcpy_htod(imgR_gpu, img2_rectified)
    imgL_array[:, :] = img1_rectified
    imgR_array[:, :] = img2_rectified
    cuda.bind_array_to_texref(cuda.matrix_to_array(imgL_array, "C"), imgL_tex)
    cuda.bind_array_to_texref(cuda.matrix_to_array(imgR_array, "C"), imgR_tex)

    # 调用 CUDA 内核计算视差
    stereoSGBM(disparity_gpu, np.int32(width), np.int32(height), 
               np.int32(blockSize), np.int32(halfBlockSize), np.int32(numDisparities), np.int32(minDisparity),
               np.int32(P1), np.int32(P2),
               block=block, grid=grid)

    # 拷贝结果回 CPU
    cuda.memcpy_dtoh(disparity, disparity_gpu)
    t_sgbm = time.time()

    # 归一化生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)
    t_norm = time.time()

    # 计算三维坐标
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    threeD = threeD * 16  # 缩放因子与 SGBM 保持一致
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
    time_sgbm = t_sgbm - t_color
    time_norm = t_norm - t_sgbm
    time_3d = t_3d - t_norm
    total_time = time.time() - t1

    # 显示耗时和帧率
    text = f"FPS: {fps:.2f}\nRead: {time_read:.4f}s\nCut: {time_cut:.4f}s\nGray: {time_gray:.4f}s\nRemap: {time_remap:.4f}s\nColor: {time_color:.4f}s\nSGBM: {time_sgbm:.4f}s\nNorm: {time_norm:.4f}s\n3D: {time_3d:.4f}s\nTotal: {total_time:.4f}s"
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
