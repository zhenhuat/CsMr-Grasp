import argparse
import os
from argparse import Namespace
from time import time

import json
import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

# 重写torch.load函数，默认使用CPU
original_load = torch.load
def custom_load(f, map_location=torch.device('cpu'), *args, **kwargs):
    return original_load(f, map_location=map_location, *args, **kwargs)
torch.load = custom_load

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn

import torch.nn as nn
from collections import defaultdict


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


class DexGraspNetDataset(torch.utils.data.Dataset):
    """Dataset for loading DexGraspNet mesh files"""
    
    def __init__(self, dataset_dir, max_samples=100):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.samples = []
        self.max_samples = max_samples
        
        print("Scanning DexGraspNet dataset...")
        # 扫描所有物体模型
        obj_dirs = []
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                obj_dirs.append(item)
        
        print(f"Found {len(obj_dirs)} object directories")
        
        # 限制样本数量
        if max_samples > 0:
            obj_dirs = obj_dirs[:max_samples]
        
        print(f"Processing first {len(obj_dirs)} directories...")
        
        valid_count = 0
        skipped_no_coacd = 0
        skipped_no_obj = 0
        
        for i, obj_dir in enumerate(sorted(obj_dirs)):
            if i % 100 == 0:
                print(f"Processing directory {i}/{len(obj_dirs)}")
            
            obj_path = os.path.join(dataset_dir, obj_dir)
            if not os.path.isdir(obj_path):
                continue
            
            # 查找OBJ文件
            coacd_dir = os.path.join(obj_path, 'coacd')
            if not os.path.isdir(coacd_dir):
                skipped_no_coacd += 1
                if skipped_no_coacd <= 5:
                    print(f"Skipping {obj_dir}: no coacd directory")
                continue
            
            # 使用分解后的OBJ文件
            obj_file = os.path.join(coacd_dir, 'decomposed.obj')
            if not os.path.exists(obj_file):
                skipped_no_obj += 1
                if skipped_no_obj <= 5:
                    print(f"Skipping {obj_dir}: no decomposed.obj file")
                continue
            
            # 提取物体类别
            if '-' in obj_dir:
                category = obj_dir.split('-')[1]
            else:
                category = 'unknown'
            
            sample_id = obj_dir
            self.samples.append({
                'sample_id': sample_id,
                'obj_path': obj_file,
                'category': category,
            })
            valid_count += 1
        
        print(f"Found {valid_count} valid samples")
        print(f"Skipped {skipped_no_coacd} directories: no coacd directory")
        print(f"Skipped {skipped_no_obj} directories: no decomposed.obj file")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load OBJ file
        import trimesh
        mesh = trimesh.load(sample_info['obj_path'], process=False)
        obj_vertices = mesh.vertices.astype(np.float32)
        obj_faces = mesh.faces.astype(np.int32)
        
        # Create data dict
        data_out = {
            'sample_id': sample_info['sample_id'],
            'obj_vertices': torch.from_numpy(obj_vertices),
            'obj_faces': torch.from_numpy(obj_faces),
            'category': sample_info['category'],
        }
        
        return data_out


def dexgraspnet_collate(batch):
    """Collate function for DexGraspNet dataset"""
    sample_ids = [item['sample_id'] for item in batch]
    categories = [item['category'] for item in batch]
    
    # Collect vertices and faces
    obj_vertices_list = []
    obj_faces_list = []
    
    for item in batch:
        obj_vertices_list.append(item['obj_vertices'])
        obj_faces_list.append(item['obj_faces'])
    
    return {
        'sample_id': sample_ids,
        'category': categories,
        'obj_vertices': obj_vertices_list,
        'obj_faces': obj_faces_list,
    }

def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    
    # Build dataset
    dataset_dir = arg.dexgraspnet_data_dir
    ds_test = DexGraspNetDataset(
        dataset_dir=dataset_dir,
        max_samples=arg.max_samples
    )
    
    data_loader = DataLoader(ds_test,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=worker_init_fn,
                             collate_fn=dexgraspnet_collate)
    
    # Build transform and model
    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    
    # Use GPU if available
    rank = 0
    if torch.cuda.is_available():
        model = DP(model).to(rank)
        transform = DP(transform).to(rank)
        device = torch.device(f"cuda:{rank}")
        logger.info(f"Using GPU {rank} for computation")
    else:
        model.to('cpu')
        transform.to('cpu')
        device = torch.device('cpu')
        logger.info("Using CPU for computation")


    memory_bps = np.load('./memory//memory_bps.npy')
    memory_hand = np.load('./memory//memory_hand.npy')
    memory_id = np.load('./memory//memory_id.npy')
    # memory_ov = np.load('./memory/memory_ov.npy')
    # memory_transl = np.load('./memory/memory_transl.npy')

    # 将NumPy数组转换为PyTorch Tensor
    memory_bps = torch.from_numpy(memory_bps)
    memory_hand = torch.from_numpy(memory_hand)
    memory_id = torch.from_numpy(memory_id)

    
    model.eval()
    
    # 初始化分类器并加载权重
    classifier = ActionClassifier(input_channels=3)
    classifier.load_state_dict(torch.load("pre_classification/best_classifier.pth", map_location=torch.device('cpu')))
    classifier = DP(classifier).to(rank)
    classifier.eval()
    
    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    # 初始化统计字典
    intent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # 为每个物体生成 hold 和 use 两种语义条件下的抓取
    intents = ['hold', 'use']
    
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Calculating classification accuracy")
        for i, inp in enumerate(data_iter):
            # 预处理：添加 OBJ_VERTS_OBJ_DS 和 OBJ_NORMALS_OBJ_DS
            import trimesh
            n_sample_verts = 4096
            obj_verts_ds_list = []
            obj_normals_ds_list = []
            
            for j in range(len(inp['obj_vertices'])):
                # 只取有效顶点
                verts = inp['obj_vertices'][j]
                faces = inp['obj_faces'][j]
                
                # 创建 trimesh
                mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy(), process=False)
                
                # 采样点和法线
                points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                face_normals = mesh.face_normals[face_indices]
                
                obj_verts_ds_list.append(torch.from_numpy(points).float())
                obj_normals_ds_list.append(torch.from_numpy(face_normals).float())
            
            # Pad sampled vertices and normals
            obj_verts_ds_padded = torch.zeros(len(obj_verts_ds_list), n_sample_verts, 3, dtype=torch.float32)
            obj_normals_ds_padded = torch.zeros(len(obj_verts_ds_list), n_sample_verts, 3, dtype=torch.float32)
            
            for j, (verts_ds, normals_ds) in enumerate(zip(obj_verts_ds_list, obj_normals_ds_list)):
                obj_verts_ds_padded[j, :verts_ds.shape[0]] = verts_ds
                obj_normals_ds_padded[j, :normals_ds.shape[0]] = normals_ds
            
            # Create dummy hand data
            batch_size = len(inp['obj_vertices'])
            dummy_hand_pose = torch.zeros(batch_size, 48, dtype=torch.float32).to(device)
            dummy_hand_shape = torch.zeros(batch_size, 10, dtype=torch.float32).to(device)
            dummy_joints = torch.zeros(batch_size, 21, 3, dtype=torch.float32).to(device)
            dummy_hand_verts = torch.zeros(batch_size, 778, 3, dtype=torch.float32).to(device)
            dummy_hand_transl = torch.zeros(batch_size, 3, dtype=torch.float32).to(device)
            
            # 为每个语义条件生成抓取
            for intent in intents:
                # Determine intent ID
                intent_id = 1 if intent == 'use' else 2
                dummy_intent_id = torch.tensor([intent_id] * batch_size, dtype=torch.long).to(device)
                
                # Pad vertices to the same size
                max_verts = max(v.shape[0] for v in inp['obj_vertices'])
                obj_verts_padded_batch = torch.zeros(len(inp['obj_vertices']), max_verts, 3, dtype=torch.float32)
                for j, verts in enumerate(inp['obj_vertices']):
                    # Repeat vertices to match max size
                    repeat_times = int(max_verts / verts.shape[0] + 1)
                    repeated_verts = torch.cat([verts] * repeat_times)[:max_verts]
                    obj_verts_padded_batch[j] = repeated_verts
                
                # Pad faces to the same size
                max_faces = max(f.shape[0] for f in inp['obj_faces'])
                obj_faces_padded_batch = torch.zeros(len(inp['obj_faces']), max_faces, 3, dtype=torch.int32)
                for j, faces in enumerate(inp['obj_faces']):
                    # Repeat faces to match max size
                    repeat_times = int(max_faces / faces.shape[0] + 1)
                    repeated_faces = torch.cat([faces] * repeat_times)[:max_faces]
                    obj_faces_padded_batch[j] = repeated_faces
                
                # Prepare transform input
                transform_inp = {
                    Queries.OBJ_VERTS_OBJ: obj_verts_padded_batch.to(device),
                    Queries.OBJ_FACES: obj_faces_padded_batch.to(device),
                    Queries.OBJ_ROTMAT: torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device),
                    Queries.OBJ_VERTS_OBJ_DS: obj_verts_ds_padded.to(device),
                    Queries.OBJ_NORMALS_OBJ_DS: obj_normals_ds_padded.to(device),
                    Queries.HAND_POSE_OBJ: dummy_hand_pose,
                    Queries.HAND_SHAPE: dummy_hand_shape,
                    Queries.JOINTS_OBJ: dummy_joints,
                    Queries.HAND_VERTS_OBJ: dummy_hand_verts,
                    Queries.HAND_TRANSL_OBJ: dummy_hand_transl,
                    Queries.INTENT_ID: dummy_intent_id,
                    'sample_id': inp['sample_id'],
                }
                
                # Apply transform
                transform_inp = transform(transform_inp)

                transform_inp['memory_bps'] = memory_bps
                transform_inp['memory_hand'] = memory_hand
                transform_inp['memory_intent'] = memory_id
                
                # Add intent vector
                intent_vec = torch.zeros((batch_size, cfg.DATA_PRESET.N_INTENTS))
                intent_vec[:, intent_id - 1] = 1.0
                transform_inp['intent_vec'] = intent_vec.to(device)
                transform_inp['intent_name'] = intent
                
                # Generate grasps
                prd, _ = model(inp=transform_inp, step_idx=i, mode="test",intent_name=intent)
                
                # Get generated hand vertices
                h_verts_r = prd["Refine.hand_verts"].detach()  # [B, 778, 3]
                o_verts_ds = transform_inp[Queries.OBJ_VERTS_OBJ_DS].detach()  # [B, 4096, 3]
                
                # Calculate features and classify
                features = torch.cat([h_verts_r, o_verts_ds], dim=1)  # [B, 4874, 3]
                logits = classifier(features)
                _, preds = torch.max(logits, dim=1)
                
                # Get labels (based on the intent we used for generation)
                label = intent_to_label[intent]
                
                # Update statistics
                for j in range(len(preds)):
                    pred = preds[j].item()
                    intent_stats[intent]['total'] += 1
                    if pred == label:
                        intent_stats[intent]['correct'] += 1
    
    # Print results
    total_correct = sum(v['correct'] for v in intent_stats.values())
    total_samples = sum(v['total'] for v in intent_stats.values())
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Correct predictions: {total_correct}")
    logger.info(f"Overall Accuracy: {total_correct / total_samples:.4f}")
    
    for intent in sorted(intent_stats.keys()):
        stats = intent_stats[intent]
        acc = stats['correct'] / stats['total']
        logger.info(f"Intent [{intent}] - Correct: {stats['correct']}, Total: {stats['total']}, Acc: {acc:.4f}")


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")
    
    parser = argparse.ArgumentParser(description='Calculate DexGraspNet classification accuracy')
    parser.add_argument("--max_samples", type=int, default=-1,
                       help='Maximum samples to process (-1 means all samples)')
    parser.add_argument("--dexgraspnet_data_dir", type=str,
                       default="/home/zhenhua/grasp/DexGraspNet-main/meshdata",
                       help='Path to DexGraspNet meshdata directory')
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold"],
                        help="specify intent for conditional generation")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))
    
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Calculating DexGraspNet classification accuracy <====")
    main(cfg, arg, exp_time)
