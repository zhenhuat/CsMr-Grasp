import argparse
import os
from argparse import Namespace
from time import time
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

from lib.datasets.grasp_data import grasp_data_collate
from lib.models.model_abc import ModelABC
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix, format_args_cfg
from lib.utils.net_utils import build_optimizer, build_scheduler, clip_gradient, setup_seed, worker_init_fn
from lib.utils.recorder import Recorder
from lib.utils.summarizer import SummaryWriter
from oikit.oi_shape.utils import ALL_INTENT


class AffordPoseDataset(torch.utils.data.Dataset):
    # 8 种 affordance 类型
    AFFORDANCE_TYPES = ['handle-grasp', 'press', 'lift', 'pull', 'twist', 'wrap-grasp', 'support', 'lever']
    
    def __init__(self, dataset_dir, split="train", categories=None, max_samples_per_object=10, n_points=4096):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.samples = []
        self.max_samples_per_object = max_samples_per_object
        self.n_points = n_points
        
        if categories is None:
            categories = [d for d in os.listdir(dataset_dir) 
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.endswith('.7z')]
        
        categories.sort()
        
        # 8-1-1 划分：按样本索引
        # 先收集所有样本
        all_samples = []
        for category in categories:
            category_dir = os.path.join(dataset_dir, category)
            if not os.path.isdir(category_dir):
                continue
            
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
                        
                        all_samples.append({
                            'json_path': json_path,
                            'category': category,
                            'instance': obj_instance,
                            'action': action,
                        })
        
        # 8-1-1 split
        np.random.seed(23)
        np.random.shuffle(all_samples)
        total = len(all_samples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        
        if split == "train":
            self.samples = all_samples[:train_end]
        elif split == "val":
            self.samples = all_samples[train_end:val_end]
        elif split == "test":
            self.samples = all_samples[val_end:]
        
        print(f"Found {len(self.samples)} {split} samples")
        

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        with open(sample_info['json_path'], 'r') as f:
            data = json.load(f)
        
        # Parse object mesh
        obj_mesh_data = data.get('object_mesh', [])
        obj_vertices, obj_faces = self._parse_obj_mesh(obj_mesh_data)
        
        # Parse hand data
        hand_data = data.get('hand', {})
        hand_joints = np.array(hand_data.get('joints', np.zeros((21, 3))))
        hand_verts = np.array(hand_data.get('vertices', np.zeros((778, 3))))
        hand_pose = np.array(hand_data.get('pose', np.zeros(48)))
        hand_shape = np.array(hand_data.get('shape', np.zeros(10)))
        hand_transl = np.array(hand_data.get('transl', np.zeros(3)))
        
        # Get object transformation
        trans_obj = np.array(data.get('trans_obj', [0.0, 0.0, 0.0]), dtype=np.float32)
        quat_obj = np.array(data.get('quat_obj', [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        rot_mat = R.from_quat(quat_obj).as_matrix().astype(np.float32)
        
        # Apply transformation to object vertices
        obj_vertices = obj_vertices @ rot_mat.T + trans_obj
        
        # Map action to intent (8种 affordance类型)
        action = sample_info['action']
        intent = self._action_to_intent(action)
        
        # Get index from our 8 types (1-based: 1-8)
        intent_id = self.AFFORDANCE_TYPES.index(intent) + 1
        
        # Create data dict
        data_out = {
            Queries.OBJ_ID: f"{sample_info['category']}_{sample_info['instance']}",
            Queries.OBJ_VERTS_OBJ: torch.from_numpy(obj_vertices).float(),
            Queries.OBJ_FACES: torch.from_numpy(obj_faces).int(),
            Queries.OBJ_ROTMAT: torch.from_numpy(rot_mat).float(),
            Queries.HAND_VERTS_OBJ: torch.from_numpy(hand_verts).float(),
            Queries.HAND_POSE_OBJ: torch.from_numpy(hand_pose).float(),
            Queries.HAND_SHAPE: torch.from_numpy(hand_shape).float(),
            Queries.JOINTS_OBJ: torch.from_numpy(hand_joints).float(),
            Queries.HAND_TRANSL_OBJ: torch.from_numpy(hand_transl).float(),
            Queries.INTENT_ID: torch.tensor([intent_id]).long(),
            'intent_name': intent,
            'action': action,
        }
        
        return data_out
    
    def _parse_obj_mesh(self, obj_mesh_data):
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
    
    def _action_to_intent(self, action):
        # 将动作分类为 8 种 affordance 类型
        action_lower = action.lower()
        
        # 检查每个类型
        affordance_keywords = {
            'handle-grasp': ['handle-grasp', 'handle grasp', 'grasp', 'hold', 'grab', 'take', 'grip'],
            'press': ['press', 'push', 'squeeze', 'operate'],
            'lift': ['lift', 'pick', 'raise', 'elevate'],
            'pull': ['pull', 'draw', 'drag'],
            'twist': ['twist', 'turn', 'rotate', 'unscrew', 'screw'],
            'wrap-grasp': ['wrap-grasp', 'wrap grasp', 'wrap'],
            'support': ['support', 'hold up', 'carry'],
            'lever': ['lever', 'pry', 'prise', 'move']
        }
        
        # 找到最匹配的
        for affordance_type, keywords in affordance_keywords.items():
            for keyword in keywords:
                if keyword in action_lower:
                    return affordance_type
        
        # 默认为 'handle-grasp'
        return 'handle-grasp'


def affordpose_collate(batch):
    sample_ids = [item[Queries.OBJ_ID] for item in batch]
    
    # Find max vertices and faces for padding
    max_verts = max(item[Queries.OBJ_VERTS_OBJ].shape[0] for item in batch)
    max_faces = max(item[Queries.OBJ_FACES].shape[0] for item in batch)
    
    # Pad everything
    batch_out = {
        Queries.OBJ_ID: sample_ids,
    }
    
    for key in [Queries.OBJ_VERTS_OBJ, Queries.OBJ_FACES, Queries.OBJ_ROTMAT, 
                Queries.HAND_VERTS_OBJ, Queries.HAND_POSE_OBJ, Queries.HAND_SHAPE,
                Queries.JOINTS_OBJ, Queries.HAND_TRANSL_OBJ, Queries.INTENT_ID]:
        
        if key == Queries.OBJ_VERTS_OBJ:
            padded = torch.zeros(len(batch), max_verts, 3, dtype=torch.float32)
            for i, item in enumerate(batch):
                verts = item[key]
                padded[i, :verts.shape[0]] = verts
            batch_out[key] = padded
        elif key == Queries.OBJ_FACES:
            padded = torch.zeros(len(batch), max_faces, 3, dtype=torch.int32)
            for i, item in enumerate(batch):
                faces = item[key]
                padded[i, :faces.shape[0]] = faces
            batch_out[key] = padded
        else:
            batch_out[key] = torch.stack([item[key] for item in batch])
    
    # Add intent_vec
    intent_ids = torch.cat([item[Queries.INTENT_ID] for item in batch])
    n_intents = 8
    intent_vec = torch.zeros(len(batch), n_intents)
    for i, intent_id in enumerate(intent_ids):
        intent_vec[i, intent_id - 1] = 1.0  # 假设 intent_id 是从 1 开始的
    batch_out['intent_vec'] = intent_vec
    
    # Add intent_name
    batch_out['intent_name'] = [item['intent_name'] for item in batch]
    
    return batch_out


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    
    # 确保 N_INTENTS = 8 匹配我们的 8 种 affordance 类型
    cfg.DATA_PRESET.N_INTENTS = 8
    
    # Monkey patch Recorder 来避免中文文件名的 git 编码问题
    import lib.utils.recorder
    original_get_git_commit = lib.utils.recorder.Recorder.get_git_commit
    def patched_get_git_commit(self):
        try:
            return original_get_git_commit(self)
        except Exception as e:
            logger.warning(f"Cannot get git commit: {e}, skip")
            return ""
    lib.utils.recorder.Recorder.get_git_commit = patched_get_git_commit
    
    recorder = Recorder(arg.exp_id, cfg, rank=0, time_f=time_f, root_path="./checkpoints/affordpose")
    summary = SummaryWriter(log_dir=recorder.tensorboard_path)

    # Create datasets
    affordpose_dir = arg.affordpose_data_dir
    train_data = AffordPoseDataset(affordpose_dir, split="train", 
                                  max_samples_per_object=arg.max_samples_per_object)
    val_data = AffordPoseDataset(affordpose_dir, split="val", 
                                max_samples_per_object=arg.max_samples_per_object)
    
    train_loader = DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=True,
                              num_workers=int(arg.workers),
                              pin_memory=True,
                              drop_last=True,
                              worker_init_fn=worker_init_fn,
                              collate_fn=affordpose_collate)

    val_loader = DataLoader(val_data,
                            batch_size=arg.val_batch_size,
                            shuffle=False,
                            num_workers=int(arg.workers),
                            drop_last=False,
                            worker_init_fn=worker_init_fn,
                            collate_fn=affordpose_collate)

    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)
    
    # 先 setup() 再用 DP 包装
    model.setup(summary_writer=summary, log_freq=arg.log_freq)
    
    # Use GPU if available
    rank = int(arg.gpu_id)  # 使用指定的 GPU ID
    if torch.cuda.is_available():
        # 打印所有可用的 GPU 设备
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} available GPUs: {[f'cuda:{i}' for i in range(num_gpus)]}")
        
        # 检查 GPU 设备是否存在
        if rank >= num_gpus:
            logger.warning(f"GPU {rank} not available, using GPU 0 instead")
            rank = 0
        
        logger.info(f"Using GPU {rank} (requested: {arg.gpu_id})")
        model = DP(model, device_ids=[rank]).to(rank)
        transform = DP(transform, device_ids=[rank]).to(rank)
        device = torch.device(f"cuda:{rank}")
        logger.info(f"Using GPU {rank} for computation")
    else:
        model.to('cpu')
        transform.to('cpu')
        device = torch.device('cpu')
        logger.info("Using CPU for computation")

    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)
    epoch = 0
    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume, arg.resume_epoch)

    n_steps = len(train_loader) * cfg.TRAIN.EPOCH
    logger.warning(f"===== start training [{epoch}, {cfg.TRAIN.EPOCH}), total iters: {n_steps} >>>>")
    
    for epoch_idx in range(epoch, cfg.TRAIN.EPOCH):
        model.train()
        trainbar = etqdm(train_loader, rank=0)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()
            step_idx = epoch_idx * len(train_loader) + bidx
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 预处理：添加 OBJ_VERTS_OBJ_DS 和 OBJ_NORMALS_OBJ_DS
            import trimesh
            import numpy as np
            n_sample_verts = 4096
            obj_verts_ds_list = []
            obj_normals_ds_list = []
            
            for i in range(len(batch[Queries.OBJ_VERTS_OBJ])):
                # 只取有效顶点（移除 padding 的零）
                verts_t = batch[Queries.OBJ_VERTS_OBJ][i]
                faces_t = batch[Queries.OBJ_FACES][i]
                verts_np = verts_t.cpu().numpy()
                faces_np = faces_t.cpu().numpy()
                
                # 过滤掉 padding 的零顶点
                valid_mask = (verts_np.sum(axis=1) != 0)
                valid_verts = verts_np[valid_mask]
                
                # 过滤掉无效的面
                max_v_idx = len(valid_verts) - 1
                valid_faces = []
                for f in faces_np:
                    if f.sum() == 0:
                        continue
                    # 确保面索引都是有效的
                    if all(idx <= max_v_idx for idx in f):
                        valid_faces.append(f)
                
                if len(valid_faces) == 0:
                    # 如果没有面，添加简单的面
                    if len(valid_verts) >= 3:
                        for j in range(0, len(valid_verts) - 2, 3):
                            valid_faces.append([j, j+1, j+2])
                
                if len(valid_verts) >= 3:
                    try:
                        # 创建 trimesh
                        mesh = trimesh.Trimesh(vertices=valid_verts, faces=valid_faces, process=False)
                        # 采样点和法线
                        points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                        face_normals = mesh.face_normals[face_indices]
                    except:
                        # 如果失败，用简单的零填充
                        points = np.zeros((n_sample_verts, 3), dtype=np.float32)
                        face_normals = np.zeros((n_sample_verts, 3), dtype=np.float32)
                else:
                    # 顶点不够，用零填充
                    points = np.zeros((n_sample_verts, 3), dtype=np.float32)
                    face_normals = np.zeros((n_sample_verts, 3), dtype=np.float32)
                
                obj_verts_ds_list.append(torch.from_numpy(points).float())
                obj_normals_ds_list.append(torch.from_numpy(face_normals).float())
            
            batch[Queries.OBJ_VERTS_OBJ_DS] = torch.stack(obj_verts_ds_list).to(device)
            batch[Queries.OBJ_NORMALS_OBJ_DS] = torch.stack(obj_normals_ds_list).to(device)
            
            # Apply transform
            batch = transform(batch)
            
            # Forward pass
            prd, loss_dict = model(batch, step_idx, "train", epoch_idx=epoch_idx)
            loss = loss_dict["loss"]
            loss.backward()
            
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()

            trainbar.set_description(f"{bar_prefix['train']} Epoch {epoch_idx} | {loss.item():.4f}")

        scheduler.step()
        logger.info(f"{bar_prefix['train']} Epoch {epoch_idx} | loss: {loss.item():.4f}, Done")
        
        # 处理 DP 包装
        if isinstance(model, DP):
            recorder.record_checkpoints(model.module, optimizer, scheduler, epoch_idx, arg.snapshot)
            model.module.on_train_finished(recorder, epoch_idx)
        else:
            recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
            model.on_train_finished(recorder, epoch_idx)

        if (epoch_idx != cfg.TRAIN.EPOCH - 1 and epoch_idx % arg.eval_freq == 0):
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                valbar = etqdm(val_loader, rank=0)
                for bidx, batch in enumerate(valbar):
                    step_idx = epoch_idx * len(val_loader) + bidx
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # 预处理：添加 OBJ_VERTS_OBJ_DS 和 OBJ_NORMALS_OBJ_DS
                    import trimesh
                    import numpy as np
                    n_sample_verts = 4096
                    obj_verts_ds_list = []
                    obj_normals_ds_list = []
                    
                    for i in range(len(batch[Queries.OBJ_VERTS_OBJ])):
                        # 只取有效顶点（移除 padding 的零）
                        verts_t = batch[Queries.OBJ_VERTS_OBJ][i]
                        faces_t = batch[Queries.OBJ_FACES][i]
                        verts_np = verts_t.cpu().numpy()
                        faces_np = faces_t.cpu().numpy()
                        
                        # 过滤掉 padding 的零顶点
                        valid_mask = (verts_np.sum(axis=1) != 0)
                        valid_verts = verts_np[valid_mask]
                        
                        # 过滤掉无效的面
                        max_v_idx = len(valid_verts) - 1
                        valid_faces = []
                        for f in faces_np:
                            if f.sum() == 0:
                                continue
                            # 确保面索引都是有效的
                            if all(idx <= max_v_idx for idx in f):
                                valid_faces.append(f)
                        
                        if len(valid_faces) == 0:
                            # 如果没有面，添加简单的面
                            if len(valid_verts) >= 3:
                                for j in range(0, len(valid_verts) - 2, 3):
                                    valid_faces.append([j, j+1, j+2])
                        
                        if len(valid_verts) >= 3:
                            try:
                                # 创建 trimesh
                                mesh = trimesh.Trimesh(vertices=valid_verts, faces=valid_faces, process=False)
                                # 采样点和法线
                                points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                                face_normals = mesh.face_normals[face_indices]
                            except:
                                # 如果失败，用简单的零填充
                                points = np.zeros((n_sample_verts, 3), dtype=np.float32)
                                face_normals = np.zeros((n_sample_verts, 3), dtype=np.float32)
                        else:
                            # 顶点不够，用零填充
                            points = np.zeros((n_sample_verts, 3), dtype=np.float32)
                            face_normals = np.zeros((n_sample_verts, 3), dtype=np.float32)
                        
                        obj_verts_ds_list.append(torch.from_numpy(points).float())
                        obj_normals_ds_list.append(torch.from_numpy(face_normals).float())
                    
                    batch[Queries.OBJ_VERTS_OBJ_DS] = torch.stack(obj_verts_ds_list).to(device)
                    batch[Queries.OBJ_NORMALS_OBJ_DS] = torch.stack(obj_normals_ds_list).to(device)
                    
                    batch = transform(batch)
                    prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

            if isinstance(model, DP):
                model.module.on_val_finished(recorder, epoch_idx)
            else:
                model.on_val_finished(recorder, epoch_idx)

    # do last evaluation
    logger.info("do last validation and save results")
    with torch.no_grad():
        model.eval()
        valbar = etqdm(val_loader, rank=0)
        for bidx, batch in enumerate(valbar):
            step_idx = epoch_idx * len(val_loader) + bidx
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 预处理：添加 OBJ_VERTS_OBJ_DS 和 OBJ_NORMALS_OBJ_DS
            import trimesh
            import numpy as np
            n_sample_verts = 4096
            obj_verts_ds_list = []
            obj_normals_ds_list = []
            
            for i in range(len(batch[Queries.OBJ_VERTS_OBJ])):
                # 只取有效顶点（移除 padding 的零）
                verts_t = batch[Queries.OBJ_VERTS_OBJ][i]
                faces_t = batch[Queries.OBJ_FACES][i]
                verts_np = verts_t.cpu().numpy()
                faces_np = faces_t.cpu().numpy()
                
                # 过滤掉 padding 的零顶点
                valid_mask = (verts_np.sum(axis=1) != 0)
                valid_verts = verts_np[valid_mask]
                
                # 过滤掉无效的面
                max_v_idx = len(valid_verts) - 1
                valid_faces = []
                for f in faces_np:
                    if f.sum() == 0:
                        continue
                    # 确保面索引都是有效的
                    if all(idx <= max_v_idx for idx in f):
                        valid_faces.append(f)
                
                if len(valid_faces) == 0:
                    # 如果没有面，添加简单的面
                    if len(valid_verts) >= 3:
                        for j in range(0, len(valid_verts) - 2, 3):
                            valid_faces.append([j, j+1, j+2])
                
                if len(valid_verts) >= 3:
                    try:
                        # 创建 trimesh
                        mesh = trimesh.Trimesh(vertices=valid_verts, faces=valid_faces, process=False)
                        # 采样点和法线
                        points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                        face_normals = mesh.face_normals[face_indices]
                    except:
                        # 如果失败，用简单的零填充
                        points = np.zeros((n_sample_verts, 3), dtype=np.float32)
                        face_normals = np.zeros((n_sample_verts, 3), dtype=np.float32)
                else:
                    # 顶点不够，用零填充
                    points = np.zeros((n_sample_verts, 3), dtype=np.float32)
                    face_normals = np.zeros((n_sample_verts, 3), dtype=np.float32)
                
                obj_verts_ds_list.append(torch.from_numpy(points).float())
                obj_normals_ds_list.append(torch.from_numpy(face_normals).float())
            
            batch[Queries.OBJ_VERTS_OBJ_DS] = torch.stack(obj_verts_ds_list).to(device)
            batch[Queries.OBJ_NORMALS_OBJ_DS] = torch.stack(obj_normals_ds_list).to(device)
            
            batch = transform(batch)
            prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

    if isinstance(model, DP):
        model.module.on_val_finished(recorder, epoch_idx)
    else:
        model.on_val_finished(recorder, epoch_idx)


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    
    parser = argparse.ArgumentParser(description='Train on AffordPose dataset')
    parser.add_argument("--affordpose_data_dir", type=str,
                        default="/home/zhenhua/grasp/AffordPose-main/AffordPose Data/",
                        help='Path to AffordPose Data directory')
    parser.add_argument("--max_samples_per_object", type=int, default=10,
                        help='Max samples per object per action')
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help='Validation batch size')
    parser.add_argument("--eval_freq", type=int, default=5,
                        help='Evaluate every N epochs')
    arg_extra, _ = parser.parse_known_args()
    
    # 只更新不存在的字段
    arg_dict = vars(arg)
    for k, v in vars(arg_extra).items():
        if k not in arg_dict:
            arg_dict[k] = v
    arg = argparse.Namespace(**arg_dict)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    
    logger.info(f"Using single GPU: {arg.gpu_id}")
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    logger.info("====> Training on AffordPose Dataset <====")
    logger.info(f"Loading config from {arg.cfg}")
    
    main(cfg, arg, exp_time)
