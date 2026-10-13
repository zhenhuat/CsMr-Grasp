import argparse
import hashlib
import os
import pickle
from argparse import Namespace
from time import time

import json
import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

# 强制使用CPU，禁用CUDA
torch.cuda.is_available = lambda: False

# 重写torch.load函数，默认使用CPU
original_load = torch.load
def custom_load(f, map_location=torch.device('cpu'), *args, **kwargs):
    return original_load(f, map_location=map_location, *args, **kwargs)
torch.load = custom_load

# 重写torch.Tensor.cuda()方法，使其不执行任何操作（保持在CPU上）
original_cuda = torch.Tensor.cuda
def custom_cuda(self, *args, **kwargs):
    return self
torch.Tensor.cuda = custom_cuda

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.etqdm import etqdm

from oikit.oi_shape.utils import ALL_INTENT


class DexGraspNetDataset(torch.utils.data.Dataset):
    """Dataset for loading DexGraspNet mesh files"""
    
    def __init__(self, dataset_dir, max_samples=-1):
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
        
        for i, obj_dir in enumerate(sorted(obj_dirs)):
            if i % 100 == 0:
                print(f"Processing directory {i}/{len(obj_dirs)}")
            
            obj_path = os.path.join(dataset_dir, obj_dir)
            if not os.path.isdir(obj_path):
                continue
            
            # 查找OBJ文件
            coacd_dir = os.path.join(obj_path, 'coacd')
            if not os.path.isdir(coacd_dir):
                continue
            
            # 使用分解后的OBJ文件
            obj_file = os.path.join(coacd_dir, 'decomposed.obj')
            if not os.path.exists(obj_file):
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
    
    # Create output directory
    dump_path = arg.output_dir
    os.makedirs(dump_path, exist_ok=True)
    logger.info(f"Dumping results to {dump_path}")
    
    def dumper(prd, inp, step_idx, **kwargs):
        """Save generated grasps to pickle files"""
        sample_ids = inp['sample_id']
        obj_rotmat = inp[Queries.OBJ_ROTMAT].detach().cpu().numpy()
        
        hand_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
        hand_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
        
        n_samples = hand_verts_r.shape[0]
        
        for i in range(n_samples):
            sample_id = sample_ids[i]
            grasp_hash = hashlib.md5(hand_verts_r[i].tobytes()).hexdigest()[:10]
            
            sample_fname = f"{sample_id}_{grasp_hash}.pkl"
            
            # Get object vertices for this sample
            obj_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS][i].detach().cpu().numpy()
            
            sample_content = {
                "sample_id": sample_id,
                "obj_rotmat": obj_rotmat[i].reshape(3, 3),
                "obj_verts_ds": obj_verts_ds,
                "hand_verts_c": hand_verts_c[i],
                "hand_verts_r": hand_verts_r[i],
                "grasp_hash": grasp_hash,
                "intent": kwargs.get('intent_name', 'unknown'),
            }
            
            with open(os.path.join(dump_path, sample_fname), "wb") as f:
                pickle.dump(sample_content, f)
        
        return True
    
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
    
    # Use CPU only
    device = torch.device('cpu')
    transform = transform.to(device)
    model = model.to(device)
    logger.info("Using CPU for computation")
    
    memory_bps = np.load('./memory//memory_bps.npy')
    memory_hand = np.load('./memory//memory_hand.npy')
    memory_id = np.load('./memory//memory_id.npy')
    
    # 将NumPy数组转换为PyTorch Tensor并移动到正确的设备
    memory_bps = torch.from_numpy(memory_bps).to(device)
    memory_hand = torch.from_numpy(memory_hand).to(device)
    memory_id = torch.from_numpy(memory_id).to(device)
    
    model.eval()
    
    # Determine intent ID
    if arg.intent is not None:
        intent_id = int(ALL_INTENT[arg.intent])
    else:
        intent_id = 1  # default to "use"
    
    # Generate grasps
    with torch.no_grad():
        data_iter = etqdm(data_loader, desc=f"{bar_prefix['test']} Generating grasps")
        
        for i, inp in enumerate(data_iter):
            # Process object vertices
            obj_vertices_list = inp['obj_vertices']
            obj_faces_list = inp['obj_faces']
            
            # Find max vertices for padding
            max_verts = max(v.shape[0] for v in obj_vertices_list)
            max_faces = max(f.shape[0] for f in obj_faces_list)
            
            # Pad and stack
            obj_verts_padded = torch.zeros(len(obj_vertices_list), max_verts, 3, dtype=torch.float32)
            obj_faces_padded = torch.zeros(len(obj_faces_list), max_faces, 3, dtype=torch.int32)
            
            for j, (verts, faces) in enumerate(zip(obj_vertices_list, obj_faces_list)):
                obj_verts_padded[j, :verts.shape[0]] = verts
                obj_faces_padded[j, :faces.shape[0]] = faces
            
            # Sample points from mesh surface using trimesh
            n_sample_verts = cfg.DATA_PRESET.N_RESAMPLED_OBJ_POINTS  # 4096
            obj_verts_ds_list = []
            obj_normals_ds_list = []
            
            for j, (verts, faces) in enumerate(zip(obj_vertices_list, obj_faces_list)):
                # Create trimesh object
                import trimesh
                mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy(), process=False)
                
                # Sample points from surface
                points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                
                # Get face normals
                tri_normals = mesh.face_normals[face_indices]
                
                obj_verts_ds = torch.from_numpy(points).float()
                obj_normals_ds = torch.from_numpy(tri_normals).float()
                
                obj_verts_ds_list.append(obj_verts_ds)
                obj_normals_ds_list.append(obj_normals_ds)
            
            # Pad sampled vertices and normals
            obj_verts_ds_padded = torch.zeros(len(obj_verts_ds_list), n_sample_verts, 3, dtype=torch.float32)
            obj_normals_ds_padded = torch.zeros(len(obj_normals_ds_list), n_sample_verts, 3, dtype=torch.float32)
            
            for j, (verts_ds, normals_ds) in enumerate(zip(obj_verts_ds_list, obj_normals_ds_list)):
                obj_verts_ds_padded[j, :verts_ds.shape[0]] = verts_ds
                obj_normals_ds_padded[j, :normals_ds.shape[0]] = normals_ds
            
            # Create dummy hand data
            batch_size = len(obj_vertices_list)
            dummy_hand_pose = torch.zeros(batch_size, 48, dtype=torch.float32).to(device)
            dummy_hand_shape = torch.zeros(batch_size, 10, dtype=torch.float32).to(device)
            dummy_joints = torch.zeros(batch_size, 21, 3, dtype=torch.float32).to(device)
            dummy_hand_verts = torch.zeros(batch_size, 778, 3, dtype=torch.float32).to(device)
            dummy_hand_transl = torch.zeros(batch_size, 3, dtype=torch.float32).to(device)
            
            dummy_intent_id = torch.tensor([intent_id] * batch_size, dtype=torch.long).to(device)
            
            transform_inp = {
                Queries.OBJ_VERTS_OBJ: obj_verts_padded.to(device),
                Queries.OBJ_FACES: obj_faces_padded.to(device),
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
            
            # Add intent vector if specified
            if arg.intent is not None:
                intent_vec = torch.zeros((batch_size, cfg.DATA_PRESET.N_INTENTS))
                intent_vec[:, intent_id - 1] = 1.0
                transform_inp['intent_vec'] = intent_vec.to(device)
                transform_inp['intent_name'] = arg.intent
            
            # Generate grasps
            prd, _ = model(inp=transform_inp, step_idx=i, mode="test", intent_name=arg.intent, callback=dumper)
    
    logger.info(f"Finished dumping results to {dump_path}")


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")
    
    parser = argparse.ArgumentParser(description='Dump DexGraspNet grasps')
    parser.add_argument("--output_dir", type=str,
                       default="/home/zhenhua/grasp/OakInk-Grasp-Generation-main/dex_result",
                       help='Output directory for dumped grasps')
    parser.add_argument("--dexgraspnet_data_dir", type=str,
                       default="/home/zhenhua/grasp/DexGraspNet-main/meshdata",
                       help='Path to DexGraspNet meshdata directory')
    parser.add_argument("--max_samples", type=int, default=-1,
                       help='Maximum samples to process (-1 means all samples)')
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold"],
                        help="specify intent for conditional generation")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))
    
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Dumping DexGraspNet grasps <====")
    main(cfg, arg, exp_time)
