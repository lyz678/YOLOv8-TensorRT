import torch
from ultralytics import YOLO
import torch.quantization
import argparse
from io import BytesIO
from pathlib import Path
import onnx
from models.common import PostDetect, optim  # 导入PostDetect和optim

def train_qat(weights, data, epochs=100, batch=16, device='0', save_dir='runs/qat', input_shape=(1, 3, 480, 640)):
    """
    YOLOv8 QAT训练函数
    
    参数:
        weights: 预训练模型路径
        data: 数据集配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        device: 训练设备
        save_dir: 保存目录
        input_shape: 输入张量形状 (batch_size, channels, height, width)
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("正在加载模型...")
    model = YOLO(weights)
    
    # 准备QAT
    print("正在准备QAT模型...")
    qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    def set_qconfig(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.qconfig = qconfig
        for child in module.children():
            set_qconfig(child)
    
    model.model.train()
    set_qconfig(model.model)
    model.model = torch.quantization.prepare_qat(model.model)
    
    # 训练参数
    train_args = {
        'data': data,
        'epochs': epochs,
        'imgsz': input_shape[-2:],
        'batch': batch,
        'device': device,
        'project': str(save_dir),
        'name': 'qat_training',
        'exist_ok': True,
        'pretrained': True,
    }
    
    # 开始训练
    print("开始QAT训练...")
    results = model.train(**train_args)
    
    # 转换为量化模型
    print("正在转换为量化模型...")
    model.model.eval()
    quantized_model = torch.quantization.convert(model.model)
    
    # 保存量化模型
    save_path = save_dir / 'yolov8n_int8_qat.pt'
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'input_shape': input_shape
    }, save_path)
    print(f"量化模型已保存至: {save_path}")
    
    # 导出 ONNX 和 TensorRT
    print("正在导出 ONNX 和 TensorRT engine...")
    
    # 加载量化模型
    quantized_yolo = YOLO(weights)  # 重新加载原始模型结构
    quantized_yolo.model.load_state_dict(torch.load(save_path)['model_state_dict'])
    quantized_yolo.model.eval()
    
    # 添加后处理层
    print("添加后处理层...")
    for m in quantized_yolo.model.modules():
        optim(m)  # 转换为PostDetect等后处理模块
        m.to(device)
    quantized_yolo.model.to(device)
    
    # 导出 ONNX
    print("正在导出ONNX模型...")
    onnx_save_path = save_dir / 'yolov8n_int8_qat.onnx'
    fake_input = torch.randn(input_shape).to(device)
    for _ in range(2):  # 预热模型
        quantized_yolo.model(fake_input)
    
    with BytesIO() as f:
        torch.onnx.export(
            quantized_yolo.model,
            fake_input,
            f,
            opset_version=11,
            input_names=['images'],
            output_names=['num_dets', 'bboxes', 'scores', 'labels'],
            dynamic_axes=None  # 固定输入形状
        )
        f.seek(0)
        onnx_model = onnx.load(f)
    
    # 验证并保存ONNX模型
    print("正在验证ONNX模型...")
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_save_path)
    print(f"ONNX模型已保存至: {onnx_save_path}")
    
    # 导出 TensorRT
    print("正在导出TensorRT engine...")
    quantized_yolo.export(
        format='engine',
        imgsz=input_shape[-2:],
        int8=True,  # 启用 INT8 量化
        device=device,
        workspace=4,  # 增加工作空间大小
        verbose=True  # 显示详细日志
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 QAT训练和导出')
    parser.add_argument('--weights', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='训练设备')
    parser.add_argument('--save-dir', type=str, default='runs/qat', help='保存目录')
    parser.add_argument('--input-shape', nargs='+', type=int, default=[1, 3, 480, 640],
                        help='输入形状 [batch_size, channels, height, width]')
    
    args = parser.parse_args()
    
    if len(args.input_shape) != 4:
        raise ValueError("input_shape必须包含4个值: [batch_size, channels, height, width]")
    
    train_qat(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        save_dir=args.save_dir,
        input_shape=tuple(args.input_shape)
    )

if __name__ == '__main__':
    main()



# python qat.py --weights yolov8n.pt --data data/coco128.yaml --epochs 10 --batch 4 --device cuda:0 --save-dir runs/qat_coco128 --input-shape 1 3 480 640
