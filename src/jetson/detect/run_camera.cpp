//
// Created by ubuntu on 1/20/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [engine_path]\n", argv[0]);
        return -1;
    }

    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    cv::VideoCapture cap(0); // 打开默认 USB 摄像头（索引 0）
    if (!cap.isOpened()) {
        printf("Cannot open USB camera\n");
        return -1;
    }

    cv::Mat             res, image;
    cv::Size            size = cv::Size{640, 480}; //W H
    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    double fps = 0.0;
    auto frame_start = std::chrono::system_clock::now();

    while (true) {
        if (!cap.read(image)) {
            printf("Failed to read frame from camera\n");
            break;
        }

        objs.clear();

        // 预处理计时
        auto preprocess_start = std::chrono::system_clock::now();
        yolov8->copy_from_Mat(image, size);
        auto preprocess_end = std::chrono::system_clock::now();
        double preprocess_time = (double)std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000.;

        // 推理计时
        auto infer_start = std::chrono::system_clock::now();
        yolov8->infer();
        auto infer_end = std::chrono::system_clock::now();
        double infer_time = (double)std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count() / 1000.;

        // 后处理计时
        auto postprocess_start = std::chrono::system_clock::now();
        yolov8->postprocess(objs);
        auto postprocess_end = std::chrono::system_clock::now();
        double postprocess_time = (double)std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start).count() / 1000.;

        // 绘制检测结果
        yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);

        // 计算 FPS
        auto frame_end = std::chrono::system_clock::now();
        double frame_time = (double)std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count() / 1000.;
        fps = 1000.0 / frame_time;
        frame_start = frame_end;

        // 在图像上显示 FPS 和各阶段时间
        char text[256];
        snprintf(text, sizeof(text), "FPS: %.2f", fps);
        cv::putText(res, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        snprintf(text, sizeof(text), "Preprocess: %.2f ms", preprocess_time);
        cv::putText(res, text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        snprintf(text, sizeof(text), "Inference: %.2f ms", infer_time);
        cv::putText(res, text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        snprintf(text, sizeof(text), "Postprocess: %.2f ms", postprocess_time);
        cv::putText(res, text, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 显示结果
        cv::imshow("result", res);

        // 按 'q' 退出
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}


// ./yolov8_realtime /home/lyz/Desktop/YOLOv8-TensorRT/yolov8n_fp16.engine