import argparse
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 自定义模块
import sys
sys.path.append('/home/robin/fareal/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_data import grasp_data_collate
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


# ActionClassifier 模型定义
class ActionClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_dims=[64, 256, 64], token_compression_ratios=[4, 4, 4, 64]):
        super().__init__()
        
        # 初始通道扩展 (3 → 64)
        self.initial_conv = nn.Conv1d(input_channels, hidden_dims[0], kernel_size=1)
        
        # 通道先升后降的模块序列
        self.channel_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.channel_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )
        
        # Token 压缩模块序列（通过跨步卷积）
        self.token_compressors = nn.ModuleList()
        for ratio in token_compression_ratios:
            self.token_compressors.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=ratio, stride=ratio),
                    nn.BatchNorm1d(hidden_dims[-1]),
                    nn.ReLU()
                )
            )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            # nn.Sigmoid()  # 二分类输出
        )

    def forward(self, x):
        # x: (B, 4874, 3)
        x = x.permute(0, 2, 1)  # (B, 3, 4874)
        
        # 初始通道扩展
        x = self.initial_conv(x)  # (B, 64, 4874)
        
        # 通道变换：64 → 256 → 64
        for block in self.channel_blocks:
            x = block(x)
        
        # Token 压缩：4874 → 1218 → 304 → 76 → 1
        for compressor in self.token_compressors:
            x = compressor(x)
        
        # 全局平均 pooling (B, 64, 1) → (B, 64)
        x = x.squeeze(-1)
        
        # 二分类输出
        return self.classifier(x)


def train_classifier(cfg: CN, arg: argparse.Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建 dataset 和 dataloader
    dataset = builder.build_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    data_loader = DataLoader(dataset,
                             batch_size=256,
                             shuffle=True,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    dataset_val = builder.build_dataset(cfg.DATASET.VAL, data_preset=cfg.DATA_PRESET)
    val_loader = DataLoader(dataset_val,
                            batch_size=256,
                            shuffle=False,
                            num_workers=int(arg.workers),
                            drop_last=False,
                            worker_init_fn=worker_init_fn,
                            collate_fn=grasp_data_collate)

    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)

    # 初始化模型、损失函数、优化器
    model = ActionClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0.0

    # 创建 log 文件路径
    log_dir = '/home/robin/fareal/OakInk-Grasp-Generation-main/classification/'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training_log.txt")

    def log_info(message):
        logger.info(message)
        with open(log_file_path, 'a') as f:
            f.write(message + "\n")

    # 开始训练循环
    for epoch in range(100):  # 训练 50 个 epoch
        model.train()
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        data_iter = tqdm(data_loader, desc=f"[Epoch {epoch}] Training", leave=False)
        for i, batch in enumerate(data_iter):
            batch = transform(batch)
            hand_verts = batch[Queries.HAND_VERTS_OBJ].to(device)
            obj_verts = batch[Queries.OBJ_VERTS_OBJ_DS].to(device)
            intent_vec = batch["intent_vec"].to(device)
            labels = torch.argmax(intent_vec, dim=1).long()

            features = torch.cat([hand_verts, obj_verts], dim=1)  # (B, 778+4096, 3)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            data_iter.set_postfix(loss=f"{loss.item():.4f}")

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro')
        rec = recall_score(all_labels, all_preds, average='macro')

        log_info(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | "
                 f"Train Acc: {acc:.4f} | Train Prec: {prec:.4f} | Train Rec: {rec:.4f}")

        # 验证阶段
        model.eval()
        val_total_correct = 0
        val_total_samples = 0
        val_all_preds = []
        val_all_labels = []

        val_iter = tqdm(val_loader, desc=f"[Epoch {epoch}] Validating", leave=False)
        for j, val_batch in enumerate(val_iter):
            val_batch = transform(val_batch)
            hand_verts_val = val_batch[Queries.HAND_VERTS_OBJ].to(device)
            obj_verts_val = val_batch[Queries.OBJ_VERTS_OBJ_DS].to(device)
            intent_vec_val = val_batch["intent_vec"].to(device)
            labels_val = torch.argmax(intent_vec_val, dim=1).long()

            features_val = torch.cat([hand_verts_val, obj_verts_val], dim=1)
            logits_val = model(features_val)
            val_loss = F.cross_entropy(logits_val, labels_val)

            preds_val = torch.argmax(logits_val, dim=1)
            val_total_correct += (preds_val == labels_val).sum().item()
            val_total_samples += labels_val.size(0)

            val_all_preds.extend(preds_val.cpu().numpy())
            val_all_labels.extend(labels_val.cpu().numpy())

            val_iter.set_postfix(loss=f"{val_loss.item():.4f}")

        val_acc = accuracy_score(val_all_labels, val_all_preds)
        val_prec = precision_score(val_all_labels, val_all_preds, average='macro')
        val_rec = recall_score(val_all_labels, val_all_preds, average='macro')
        class_names = ['use', 'hold']
        report = classification_report(val_all_labels, val_all_preds, target_names=class_names, digits=4)

        log_info(f"[Epoch {epoch}] Val Loss: {val_loss.item():.4f} | "
                 f"Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}")
        log_info(report)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(log_dir, "best_classifier.pth")
            torch.save(model.state_dict(), save_path)
            log_info(f"Saved new best model with Val Acc={val_acc:.4f}")

def test_classifier(cfg: CN, arg: argparse.Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建 dataset 和 dataloader
    dataset_val = builder.build_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    val_loader = DataLoader(dataset_val,
                            batch_size=128,
                            shuffle=False,
                            num_workers=int(arg.workers),
                            drop_last=False,
                            worker_init_fn=worker_init_fn,
                            collate_fn=grasp_data_collate)

    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)

    # 初始化模型、损失函数、优化器
    model = ActionClassifier().to(device)

    # 加载模型权重
    checkpoint_path = "/home/robin/fareal/OakInk-Grasp-Generation-main/classification//best_classifier.pth"
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 因为你在训练时用了 torch.save(model.state_dict(), ...)
    model.load_state_dict(checkpoint)

    print(f"Loaded model weights from {checkpoint_path}")

    def log_info(message):
        logger.info(message)

    best_acc = 0.0

    # 验证阶段
    model.eval()
    val_total_correct = 0
    val_total_samples = 0
    val_all_preds = []
    val_all_labels = []

    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="Validating", leave=False)
        for j, val_batch in enumerate(val_iter):
            val_batch = transform(val_batch)
            hand_verts_val = val_batch[Queries.HAND_VERTS_OBJ].to(device)
            obj_verts_val = val_batch[Queries.OBJ_VERTS_OBJ_DS].to(device)
            intent_vec_val = val_batch["intent_vec"].to(device)
            labels_val = torch.argmax(intent_vec_val, dim=1).long()

            features_val = torch.cat([hand_verts_val, obj_verts_val], dim=1)
            logits_val = model(features_val)
            val_loss = F.cross_entropy(logits_val, labels_val)

            preds_val = torch.argmax(logits_val, dim=1)
            val_total_correct += (preds_val == labels_val).sum().item()
            val_total_samples += labels_val.size(0)

            val_all_preds.extend(preds_val.cpu().numpy())
            val_all_labels.extend(labels_val.cpu().numpy())

            val_iter.set_postfix(loss=f"{val_loss.item():.4f}")

    val_acc = accuracy_score(val_all_labels, val_all_preds)
    val_prec = precision_score(val_all_labels, val_all_preds, average='macro')
    val_rec = recall_score(val_all_labels, val_all_preds, average='macro')
    class_names = ['use', 'hold']
    report = classification_report(val_all_labels, val_all_preds, target_names=class_names, digits=4)

    log_info(f"Val Loss: {val_loss.item():.4f} | "
             f"Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}")
    log_info(report)



if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    logger.info(f"Using GPU: {arg.gpu_id}")

    # 解析额外参数
    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="train", choices=["train", "val", "test"])
    arg_extra, _ = parser.parse_known_args()
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    # 加载配置文件
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Start training intent classifier <====")

    # 开始训练
    train_classifier(cfg, arg, exp_time)
    # test_classifier(cfg, arg, exp_time)