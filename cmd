python3 export-det.py --weights yolov8n.pt --iou-thres 0.65 --conf-thres 0.25 --topk 50 --opset 11 --sim --input-shape 1 3 480 640 --device cuda:0


python3 build.py --weights yolov8n.onnx --iou-thres 0.65 --conf-thres 0.25 --topk 50 --fp16  --device cuda:0


trtexec --onnx=yolov8s.onnx --saveEngine=yolov8n_fp16.engine --fp16


python3 infer-det.py --engine yolov8s.engine --imgs data --show --out-dir outputs --device cuda:0


python3 infer-det-without-torch.py --engine yolov8s_fp16.engine --imgs data --out-dir outputs --method pycuda

python3 run_camera.py --model yolov8s_fp16.engine --q fp16

python3 run_camera.py --engine yolov8s_fp16.engine --camera-id 0 --show
python3 run_camera.py --engine yolov8n_fp16.engine --camera-id 0 --show
python3 run_camera_video_stream.py --engine yolov8n_fp16.engine --camera-id 0

sudo chmod 666 /dev/ttyTHS1
python3 run_track_person.py --engine yolov8n_fp16.engine --camera-id 0

roscore
rosrun usb_cam usb_cam_node
python3 run_track_person_ros.py --engine yolov8n_fp16.engine


export root=${PWD}
cd src/detect/end2end
mkdir build && cd build
cmake ..
make
mv yolov8 ${root}
cd ${root}
