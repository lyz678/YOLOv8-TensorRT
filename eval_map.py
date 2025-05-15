import argparse
from pathlib import Path
import json
import torch
from tqdm import tqdm
import numpy as np
import cv2
from ultralytics import YOLO
from models import TRTModule
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

class COCO128Evaluator:
    def __init__(self, dataset_path, save_dir="eval_results"):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images/train2017"
        self.labels_dir = self.dataset_path / "labels/train2017"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.results = []
        self.num_classes = 80  # COCO数据集类别数
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, ..., 0.95]
        
    def load_label(self, label_file):
        """加载YOLO格式的标签文件"""
        boxes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    cls_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([cls_id, x, y, w, h])
        return np.array(boxes)
    
    def calculate_ap(self, recalls, precisions):
        """计算平均精度(AP)"""
        # 使用所有不同阈值下的精度值的平均值
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))
        
        # 计算PR曲线下的面积
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
            
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap
        
    def evaluate_predictions(self, predictions, ground_truths, iou_threshold=0.5):
        """评估单张图像的预测结果"""
        metrics = {
            'TP': 0,  # True Positives
            'FP': 0,  # False Positives
            'FN': 0   # False Negatives
        }
        
        if len(ground_truths) == 0:
            metrics['FP'] += len(predictions)
            return metrics
            
        if len(predictions) == 0:
            metrics['FN'] += len(ground_truths)
            return metrics
        
        # 计算IoU矩阵
        ious = np.zeros((len(predictions), len(ground_truths)))
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truths):
                ious[i,j] = self.calculate_iou(pred, gt)
        
        # 匹配预测框和真实框
        matched_gt = set()
        for i in range(len(predictions)):
            max_iou = np.max(ious[i])
            if max_iou >= iou_threshold:
                gt_idx = np.argmax(ious[i])
                if gt_idx not in matched_gt:
                    metrics['TP'] += 1
                    matched_gt.add(gt_idx)
                else:
                    metrics['FP'] += 1
            else:
                metrics['FP'] += 1
                
        metrics['FN'] += len(ground_truths) - len(matched_gt)
        return metrics
    
    @staticmethod
    def calculate_iou(box1, box2):
        """计算两个框的IoU"""
        # 转换为xyxy格式
        b1_x1 = box1[0] - box1[2]/2
        b1_y1 = box1[1] - box1[3]/2
        b1_x2 = box1[0] + box1[2]/2
        b1_y2 = box1[1] + box1[3]/2
        
        b2_x1 = box2[0] - box2[2]/2
        b2_y1 = box2[1] - box2[3]/2
        b2_x2 = box2[0] + box2[2]/2
        b2_y2 = box2[1] + box2[3]/2
        
        # 计算交集
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        return inter_area / (b1_area + b2_area - inter_area)
    
    def evaluate_model(self, model, model_type, device="cuda:0"):
        """评估模型性能，计算mAP"""
        # 存储每个类别在每个IoU阈值下的预测结果
        predictions_by_class = {i: [] for i in range(self.num_classes)}
        ground_truths_by_class = {i: [] for i in range(self.num_classes)}
        
        is_engine = not isinstance(model, YOLO)
        if is_engine:
            H, W = model.inp_info[0].shape[-2:]
        
        # 收集所有预测和真实标签
        for img_path in tqdm(list(self.images_dir.glob("*.jpg")), desc=f"评估 {model_type}"):
            # 加载图像和标签
            img = cv2.imread(str(img_path))
            label_file = self.labels_dir / f"{img_path.stem}.txt"
            ground_truths = self.load_label(label_file)
            
            if is_engine:
                # TensorRT推理
                img_resized, ratio, dwdh = letterbox(img, (W, H))
                rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                tensor = blob(rgb, return_seg=False)
                dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=device)
                tensor = torch.tensor(tensor, device=device)
                
                data = model(tensor)
                bboxes, scores, labels = det_postprocess(data)
                
                predictions = []
                if bboxes.numel() > 0:
                    bboxes -= dwdh
                    bboxes /= ratio
                    
                    # 转换为YOLO格式 (cls_id, x, y, w, h)
                    for bbox, score, label in zip(bboxes, scores, labels):
                        if score > 0.25:  # 置信度阈值
                            x1, y1, x2, y2 = bbox.tolist()
                            w = x2 - x1
                            h = y2 - y1
                            x = x1 + w/2
                            y = y1 + h/2
                            predictions.append([int(label), x/img.shape[1], y/img.shape[0], 
                                             w/img.shape[1], h/img.shape[0]])
            else:
                # PyTorch YOLO推理
                results = model(img)
                predictions = []
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for box, score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                        if score > 0.25:  # 置信度阈值
                            x1, y1, x2, y2 = box.tolist()
                            w = x2 - x1
                            h = y2 - y1
                            x = x1 + w/2
                            y = y1 + h/2
                            predictions.append([int(cls), x/img.shape[1], y/img.shape[0], 
                                             w/img.shape[1], h/img.shape[0]])
            
            # 将预测结果和真实标签按类别存储
            for pred in predictions:
                cls_id = int(pred[0])
                predictions_by_class[cls_id].append({
                    'confidence': pred[5] if len(pred) > 5 else 1.0,
                    'bbox': pred[1:5]
                })
                
            for gt in ground_truths:
                cls_id = int(gt[0])
                ground_truths_by_class[cls_id].append({
                    'bbox': gt[1:5]
                })
        
        # 计算每个类别的AP
        aps = []
        for cls_id in range(self.num_classes):
            if not ground_truths_by_class[cls_id]:
                continue
                
            cls_aps = []
            for iou_threshold in self.iou_thresholds:
                # 按置信度排序预测框
                cls_preds = sorted(predictions_by_class[cls_id], 
                                 key=lambda x: x['confidence'], reverse=True)
                cls_gts = ground_truths_by_class[cls_id]
                
                tp = np.zeros(len(cls_preds))
                fp = np.zeros(len(cls_preds))
                
                # 计算TP和FP
                for pred_idx, pred in enumerate(cls_preds):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(cls_gts):
                        iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                            
                    if best_iou >= iou_threshold:
                        tp[pred_idx] = 1
                    else:
                        fp[pred_idx] = 1
                
                # 计算累积值
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                # 计算精度和召回率
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
                recalls = tp_cumsum / len(cls_gts)
                
                # 计算AP
                ap = self.calculate_ap(recalls, precisions)
                cls_aps.append(ap)
            
            # 计算该类别在所有IoU阈值下的平均AP
            aps.append(np.mean(cls_aps))
        
        # 计算mAP
        mAP = np.mean(aps)
        
        return {
            'mAP': mAP,
            'AP_by_class': {i: ap for i, ap in enumerate(aps) if ap > 0},
            'mAP@.5': aps[0],  # IoU=0.5时的mAP
            'mAP@.5:.95': mAP  # 所有IoU阈值下的平均mAP
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt-model', type=str, help='PyTorch模型路径(.pt)')
    parser.add_argument('--fp16-engine', type=str, help='FP16 TensorRT引擎路径')
    parser.add_argument('--int8-ptq-engine-min-max', type=str, help='INT8 PTQ TensorRT引擎路径')
    parser.add_argument('--int8-ptq-engine-entropy', type=str, help='INT8 PTQ TensorRT引擎路径')
    parser.add_argument('--int8-qat-engine', type=str, help='INT8 QAT TensorRT引擎路径')
    parser.add_argument('--dataset', type=str, required=True, help='COCO128数据集目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    args = parser.parse_args()
    
    evaluator = COCO128Evaluator(args.dataset)
    results = {}
    
    # 评估PyTorch模型
    if args.pt_model:
        model = YOLO(args.pt_model)
        results['PyTorch'] = evaluator.evaluate_model(model, "PyTorch", args.device)
        
    # 评估FP16模型
    if args.fp16_engine:
        model = TRTModule(args.fp16_engine, torch.device(args.device))
        model.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        results['TensorRT-FP16'] = evaluator.evaluate_model(model, "TensorRT-FP16", args.device)
        
    # 评估INT8 PTQ模型
    if args.int8_ptq_engine_min_max:
        model = TRTModule(args.int8_ptq_engine_min_max, torch.device(args.device))
        model.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        results['TensorRT-INT8-PTQ-MIN-MAX'] = evaluator.evaluate_model(model, "TensorRT-INT8-PTQ-MIN-MAX", args.device)
        del model
    
    if args.int8_ptq_engine_entropy:
        model = TRTModule(args.int8_ptq_engine_entropy, torch.device(args.device))
        model.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        results['TensorRT-INT8-PTQ-ENTROPY'] = evaluator.evaluate_model(model, "TensorRT-INT8-PTQ-ENTROPY", args.device)
        del model

    # 评估INT8 QAT模型
    if args.int8_qat_engine:
        model = TRTModule(args.int8_qat_engine, torch.device(args.device))
        model.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        results['TensorRT-INT8-QAT'] = evaluator.evaluate_model(model, "TensorRT-INT8-QAT", args.device)

    # 打印评估结果
    print("\n=== 模型性能比较 ===")
    print("模型类型                       mAP@.5    mAP@.5:.95")
    print("-" * 50)
    for model_type, metrics in results.items():
        print(f"{model_type:<30} {metrics['mAP@.5']:.4f}    {metrics['mAP@.5:.95']:.4f}")

if __name__ == '__main__':
    main()



# python eval_map.py --dataset /media/lyz/Data1/coco128 --device cuda:0 --fp16-engine yolov8n_fp16.engine --pt-model yolov8n.pt --int8-ptq-engine-min-max yolov8n_int8_ptq_min_max.engine --int8-ptq-engine-entropy yolov8n_int8_ptq_entropy.engine --int8-qat-engine runs/qat_coco128/qat_training/weights/best.engine

