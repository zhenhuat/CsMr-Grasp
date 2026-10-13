import argparse
import os
from argparse import Namespace
from time import time

import json
import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

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


class AffordPoseDataset(torch.utils.data.Dataset):
    """Dataset for loading AffordPose JSON files"""
    
    def __init__(self, dataset_dir, max_samples_per_object=10):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.samples = []
        self.max_samples_per_object = max_samples_per_object
        
        print("Scanning AffordPose dataset...")
        # 扫描所有类别
        categories = [d for d in os.listdir(dataset_dir) 
                     if os.path.isdir(os.path.join(dataset_dir, d)) and not d.endswith('.7z')]
        
        for category in sorted(categories):
            category_dir = os.path.join(dataset_dir, category)
            if not os.path.isdir(category_dir):
                continue
            
            print(f"Processing category: {category}")
            for obj_instance in sorted(os.listdir(category_dir)):
                obj_instance_dir = os.path.join(category_dir, obj_instance)
                if not os.path.isdir(obj_instance_dir):
                    continue
                
                for action in sorted(os.listdir(obj_instance_dir)):
                    action_dir = os.path.join(obj_instance_dir, action)
                    if not os.path.isdir(action_dir):
                        continue
                    
                    # Get JSON files
                    json_files = sorted([f for f in os.listdir(action_dir) if f.endswith('.json')])
                    
                    # Sample up to max_samples_per_object
                    sample_indices = np.linspace(0, len(json_files)-1, 
                                               min(max_samples_per_object, len(json_files)), 
                                               dtype=int)
                    
                    for idx in sample_indices:
                        json_file = json_files[idx]
                        json_path = os.path.join(action_dir, json_file)
                        
                        sample_id = f"{category}_{obj_instance}_{action}_{os.path.splitext(json_file)[0]}"
                        self.samples.append({
                            'sample_id': sample_id,
                            'json_path': json_path,
                            'category': category,
                            'instance': obj_instance,
                            'action': action,
                        })
        
        print(f"Found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load JSON file
        with open(sample_info['json_path'], 'r') as f:
            data = json.load(f)
        
        # Parse object mesh
        obj_mesh_data = data.get('object_mesh', [])
        obj_vertices, obj_faces = self._parse_obj_mesh(obj_mesh_data)
        
        # Get object transformation
        trans_obj = np.array(data.get('trans_obj', [0.0, 0.0, 0.0]), dtype=np.float32)
        quat_obj = np.array(data.get('quat_obj', [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        rot_mat = R.from_quat(quat_obj).as_matrix().astype(np.float32)
        
        # Apply transformation to object vertices
        obj_vertices = obj_vertices @ rot_mat.T + trans_obj
        
        # Create data dict
        data_out = {
            'sample_id': sample_info['sample_id'],
            'obj_vertices': torch.from_numpy(obj_vertices),
            'obj_faces': torch.from_numpy(obj_faces),
            'obj_rotmat': torch.from_numpy(rot_mat),
            'obj_trans': torch.from_numpy(trans_obj),
            'category': sample_info['category'],
            'instance': sample_info['instance'],
            'action': sample_info['action'],
        }
        
        return data_out
    
    def _parse_obj_mesh(self, obj_mesh_data):
        """Parse object mesh from JSON format"""
        vertices = []
        faces = []
        
        if isinstance(obj_mesh_data, str):
            lines = obj_mesh_data.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()[1:4]
                    if len(parts) >= 3:
                        vertices.append([float(v) for v in parts])
                elif line.startswith('f '):
                    parts = line.split()[1:4]
                    if len(parts) >= 3:
                        face = [int(p.split('/')[0]) - 1 for p in parts]
                        faces.append(face)
        elif isinstance(obj_mesh_data, list):
            for item in obj_mesh_data:
                if isinstance(item, str):
                    lines = item.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('v '):
                            parts = line.split()[1:4]
                            if len(parts) >= 3:
                                vertices.append([float(v) for v in parts])
                        elif line.startswith('f '):
                            parts = line.split()[1:4]
                            if len(parts) >= 3:
                                face = [int(p.split('/')[0]) - 1 for p in parts]
                                faces.append(face)
        
        # Fallback triangulation if no faces
        if len(faces) == 0 and len(vertices) > 2:
            for i in range(0, len(vertices) - 2, 3):
                faces.append([i, i+1, i+2])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def affordpose_collate(batch):
    """Collate function for AffordPose dataset"""
    sample_ids = [item['sample_id'] for item in batch]
    categories = [item['category'] for item in batch]
    instances = [item['instance'] for item in batch]
    actions = [item['action'] for item in batch]
    
    # Find max vertices
    max_verts = max(item['obj_vertices'].shape[0] for item in batch)
    
    # Pad vertices and faces
    obj_vertices_list = []
    obj_faces_list = []
    obj_rotmats = []
    obj_trans = []
    
    for item in batch:
        obj_vertices_list.append(item['obj_vertices'])
        obj_faces_list.append(item['obj_faces'])
        obj_rotmats.append(item['obj_rotmat'])
        obj_trans.append(item['obj_trans'])
    
    return {
        'sample_id': sample_ids,
        'category': categories,
        'instance': instances,
        'action': actions,
        'obj_vertices': obj_vertices_list,
        'obj_faces': obj_faces_list,
        'obj_rotmat': torch.stack(obj_rotmats),
        'obj_trans': torch.stack(obj_trans),
    }

def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    
    # Build dataset
    dataset_dir = arg.affordpose_data_dir
    ds_test = AffordPoseDataset(
        dataset_dir=dataset_dir,
        max_samples_per_object=arg.max_samples
    )
    
    data_loader = DataLoader(ds_test,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=worker_init_fn,
                             collate_fn=affordpose_collate)
    
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
    
    model.eval()
    
    # 初始化分类器并加载权重
    classifier = ActionClassifier(input_channels=3)
    classifier.load_state_dict(torch.load("/home/zhenhua/grasp/OakInk-Grasp-Generation-main/pre_classification//best_classifier.pth"))
    classifier = DP(classifier).to(rank)
    classifier.eval()
    
    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    # 初始化统计字典：按类别和意图统计
    category_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    overall_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # 为每个物体生成 hold 和 use 两种语义条件的抓取
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
                # 只取有效顶点（移除 padding 的零）
                verts = inp['obj_vertices'][j]
                faces = inp['obj_faces'][j]
                
                # 创建 trimesh
                mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy(), process=False)
                
                # 采样点和法线
                points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                face_normals = mesh.face_normals[face_indices]
                
                # 确保有 4096 个点（如果不够就复制）
                points = torch.from_numpy(points).float()
                face_normals = torch.from_numpy(face_normals).float()
                
                # 如果采样点不够 4096，就复制点
                if points.shape[0] < n_sample_verts:
                    repeat_times = (n_sample_verts // points.shape[0]) + 1
                    points = torch.cat([points] * repeat_times)[:n_sample_verts]
                    face_normals = torch.cat([face_normals] * repeat_times)[:n_sample_verts]
                
                obj_verts_ds_list.append(points)
                obj_normals_ds_list.append(face_normals)
            
            # Stack sampled vertices and normals (no padding needed now)
            obj_verts_ds_padded = torch.stack(obj_verts_ds_list)
            obj_normals_ds_padded = torch.stack(obj_normals_ds_list)
            
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
                
                # 找到最大的顶点和面数量
                max_verts = max(v.shape[0] for v in inp['obj_vertices'])
                max_faces = max(f.shape[0] for f in inp['obj_faces'])
                
                # 对顶点和面进行 padding（只用于 OBJ_VERTS_OBJ 和 OBJ_FACES）
                obj_verts_padded_batch = torch.zeros(len(inp['obj_vertices']), max_verts, 3, dtype=torch.float32)
                for j, verts in enumerate(inp['obj_vertices']):
                    # 重复顶点以匹配最大尺寸
                    repeat_times = int(max_verts / verts.shape[0] + 1)
                    repeated_verts = torch.cat([verts] * repeat_times)[:max_verts]
                    obj_verts_padded_batch[j] = repeated_verts
                
                obj_faces_padded_batch = torch.zeros(len(inp['obj_faces']), max_faces, 3, dtype=torch.int32)
                for j, faces in enumerate(inp['obj_faces']):
                    # 重复面以匹配最大尺寸
                    repeat_times = int(max_faces / faces.shape[0] + 1)
                    repeated_faces = torch.cat([faces] * repeat_times)[:max_faces]
                    obj_faces_padded_batch[j] = repeated_faces
                
                # Prepare transform input
                transform_inp = {
                    Queries.OBJ_VERTS_OBJ: obj_verts_padded_batch.to(device),
                    Queries.OBJ_FACES: obj_faces_padded_batch.to(device),
                    Queries.OBJ_ROTMAT: inp['obj_rotmat'].to(device),
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
                
                # Add intent vector
                intent_vec = torch.zeros((batch_size, cfg.DATA_PRESET.N_INTENTS))
                intent_vec[:, intent_id - 1] = 1.0
                transform_inp['intent_vec'] = intent_vec.to(device)
                transform_inp['intent_name'] = intent
                
                # Generate grasps
                prd, _ = model(inp=transform_inp, step_idx=i, mode="test")
                
                # Get generated hand vertices
                h_verts_r = prd["Refine.hand_verts"].detach()  # [B, 778, 3]
                o_verts_ds = transform_inp[Queries.OBJ_VERTS_OBJ_DS].detach()  # [B, 4096, 3]
                
                # Calculate features and classify
                features = torch.cat([h_verts_r, o_verts_ds], dim=1)  # [B, 4874, 3]
                logits = classifier(features)
                _, preds = torch.max(logits, dim=1)
                
                # Get labels (based on the intent we used for generation)
                label = intent_to_label[intent]
                
                # Update statistics by category
                for j in range(len(preds)):
                    category = inp['category'][j]
                    pred = preds[j].item()
                    
                    # Update category stats
                    category_stats[category][intent]['total'] += 1
                    if pred == label:
                        category_stats[category][intent]['correct'] += 1
                    
                    # Update overall stats
                    overall_stats[intent]['total'] += 1
                    if pred == label:
                        overall_stats[intent]['correct'] += 1
    
    # Print overall results
    total_correct = sum(v['correct'] for v in overall_stats.values())
    total_samples = sum(v['total'] for v in overall_stats.values())
    
    logger.info("=" * 80)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Correct predictions: {total_correct}")
    logger.info(f"Overall Accuracy: {total_correct / total_samples:.4f}")
    
    for intent in sorted(overall_stats.keys()):
        stats = overall_stats[intent]
        acc = stats['correct'] / stats['total']
        logger.info(f"Intent [{intent}] - Correct: {stats['correct']}, Total: {stats['total']}, Acc: {acc:.4f}")
    
    # Print results by category
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS BY CATEGORY")
    logger.info("=" * 80)
    
    for category in sorted(category_stats.keys()):
        logger.info(f"\nCategory: {category}")
        logger.info("-" * 40)
        
        category_total_correct = 0
        category_total_samples = 0
        
        for intent in sorted(category_stats[category].keys()):
            stats = category_stats[category][intent]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                logger.info(f"  Intent [{intent}] - Correct: {stats['correct']}, Total: {stats['total']}, Acc: {acc:.4f}")
                category_total_correct += stats['correct']
                category_total_samples += stats['total']
        
        if category_total_samples > 0:
            category_acc = category_total_correct / category_total_samples
            logger.info(f"  Category Overall - Correct: {category_total_correct}, Total: {category_total_samples}, Acc: {category_acc:.4f}")


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")
    
    parser = argparse.ArgumentParser(description='Calculate affordance classification accuracy by category')
    parser.add_argument("--max_samples", type=int, default=1,
                       help='Maximum samples per object instance')
    parser.add_argument("--affordpose_data_dir", type=str,
                       default="/home/zhenhua/grasp/AffordPose-main/AffordPose Data/",
                       help='Path to AffordPose Data directory')
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))
    
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Calculating affordance classification accuracy by category <====")
    main(cfg, arg, exp_time)
