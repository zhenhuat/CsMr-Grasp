import argparse
import os
import pickle
from argparse import Namespace
from time import time

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

from lib.utils.logger import logger
from lib.utils.config import CN
from lib.datasets.grasp_query import Queries
from lib.utils.net_utils import setup_seed
from lib.models.grabnet.grabnet_arch import GrabNet


class ActionClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_dims=[64, 256, 64], token_compression_ratios=[4, 4, 4, 64]):
        super().__init__()
        
        self.initial_conv = nn.Conv1d(input_channels, hidden_dims[0], kernel_size=1)
        
        self.channel_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.channel_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )
        
        self.token_compressors = nn.ModuleList()
        for ratio in token_compression_ratios:
            self.token_compressors.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=ratio, stride=ratio),
                    nn.BatchNorm1d(hidden_dims[-1]),
                    nn.ReLU()
                )
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.initial_conv(x)
        for block in self.channel_blocks:
            x = block(x)
        for compressor in self.token_compressors:
            x = compressor(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class DumpedDexGraspDataset(torch.utils.data.Dataset):
    def __init__(self, dumped_dir):
        self.dumped_dir = dumped_dir
        self.pkl_files = [os.path.join(dumped_dir, f) for f in os.listdir(dumped_dir) if f.endswith('.pkl')]
        self.pkl_files.sort()

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], "rb") as f:
            data = pickle.load(f)
        return data


def evaluate(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(23)
    rank = 0
    
    # Create dataset for dumped grasps
    dataset = DumpedDexGraspDataset(arg.dumped_dir)
    logger.info(f"Found {len(dataset)} dumped grasp samples")
    
    # Initialize classifier and load weights
    classifier = ActionClassifier(input_channels=3)
    classifier.load_state_dict(torch.load(arg.classifier_path, map_location='cpu'))
    classifier.eval()
    
    # Use GPU if available
    if torch.cuda.is_available():
        classifier = classifier.to(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        classifier = classifier.to('cpu')
        device = torch.device('cpu')
    
    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    # 统计
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 100 == 0:
                print(f"Processing sample {idx}/{len(dataset)}")
            
            data = dataset[idx]
            
            # 获取手部顶点 - 优先使用 hand_verts_r，如果不存在则使用 hand_verts_c
            if 'hand_verts_r' in data:
                hand_verts = data['hand_verts_r']
            elif 'hand_verts_c' in data:
                hand_verts = data['hand_verts_c']
            else:
                print(f"Warning: Sample {idx} has no hand vertices, skipping")
                continue
            
            # 获取物体点云
            if 'obj_verts_ds' not in data:
                print(f"Warning: Sample {idx} has no object points, skipping")
                continue
            
            obj_verts_ds = data['obj_verts_ds']
            
            hand_verts_r = torch.from_numpy(hand_verts).unsqueeze(0).float().to(device)
            obj_verts_ds = torch.from_numpy(obj_verts_ds).unsqueeze(0).float().to(device)
            
            features = torch.cat([hand_verts_r, obj_verts_ds], dim=1)
            
            logits = classifier(features)
            _, preds = torch.max(logits, dim=1)
            
            # 标签：使用 arg.intent 作为标签
            if arg.intent is not None:
                labels = torch.tensor([intent_to_label[arg.intent]], dtype=torch.long, device=logits.device)
            else:
                print("Warning: arg.intent not specified, defaulting to 'use'")
                labels = torch.tensor([0], dtype=torch.long, device=logits.device)

            # 统计
            label = labels[0].item()
            pred = preds[0].item()
            
            class_stats[label]['total'] += 1
            if pred == label:
                class_stats[label]['correct'] += 1
                total_correct += 1
            total_samples += 1
    
    # 打印结果
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Correct predictions: {total_correct}")
    if total_samples > 0:
        logger.info(f"Overall Accuracy: {total_correct / total_samples:.4f}")

        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            acc = stats['correct'] / stats['total']
            logger.info(f"Class [{label_to_intent[label]}] - Correct: {stats['correct']}, Total: {stats['total']}, Acc: {acc:.4f}")
    
    # 保存结果
    results_path = os.path.join(arg.output_dir, f"classification_results_{arg.intent}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({
            "total_samples": total_samples,
            "total_correct": total_correct,
            "class_stats": dict(class_stats),
            "intent": arg.intent,
            "dumped_dir": arg.dumped_dir,
        }, f)
    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    exp_time = time()
    parser = argparse.ArgumentParser(description='Evaluate classification accuracy on dumped DexGraspNet grasps')
    
    parser.add_argument("--dumped_dir", type=str, 
                        default="/home/zhenhua/grasp/OakInk-Grasp-Generation-main/dex_result/csmr_hold/result",
                        help='Directory with dumped grasp .pkl files')
    parser.add_argument("--classifier_path", type=str,
                        default="/home/zhenhua/grasp/OakInk-Grasp-Generation-main/pre_classification//best_classifier.pth",
                        help='Path to classifier weights')
    parser.add_argument("--intent", type=str, required=True,
                        choices=["use", "hold"],
                        help='Intent label for all samples (use or hold)')
    parser.add_argument("--output_dir", type=str,
                        default="/home/zhenhua/grasp/OakInk-Grasp-Generation-main/dex_result")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--cfg", type=str, default="")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(23)
    logger.info("Evaluate classification accuracy on dumped DexGraspNet grasps!")
    logger.info(f"Intent: {args.intent}")
    logger.info(f"Dumped directory: {args.dumped_dir}")
    logger.info(f"Classifier: {args.classifier_path}")
    
    # 创建简单的 cfg 对象
    cfg = CN()
    evaluate(cfg, args, exp_time)
