import argparse
import os
import sys
import time
from argparse import Namespace

import torch
from torch.nn.parallel import DataParallel as DP
import numpy as np

sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.logger import logger


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def get_gpu_memory():
    """获取当前 GPU 显存占用 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def main(cfg: CN, arg: Namespace, time_f: float):
    rank = 0
    
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    model.eval()
    
    total_cnet_params = count_params(model.module.CNet)
    total_rnet_params = count_params(model.module.RNet)
    total_params = total_cnet_params + total_rnet_params
    
    logger.info(f"\n{'='*60}")
    logger.info("MODEL PARAMETERS")
    logger.info(f"{'='*60}")
    logger.info(f"CoarseNet:  {total_cnet_params / 1e6:.2f}M")
    logger.info(f"RefineNet:  {total_rnet_params / 1e6:.2f}M")
    logger.info(f"Total:      {total_params / 1e6:.2f}M")
    logger.info(f"{'='*60}\n")
    
    batch_size = 1

    memory_bps = np.load('./memory/memory_bps.npy')
    memory_hand = np.load('./memory/memory_hand.npy')
    memory_id = np.load('./memory/memory_id.npy')
    
    dummy_obj_bps = torch.randn(batch_size, 4096, device=rank)
    dummy_intent_vec = torch.zeros(batch_size, 2, device=rank)
    dummy_intent_vec[:, 1] = 1.0
    dummy_obj_verts = torch.randn(batch_size, 4096, 3, device=rank)
    dummy_obj_faces = torch.randint(0, 1000, (batch_size, 1000, 3), dtype=torch.int64, device=rank)
    dummy_hand_pose = torch.randn(batch_size, 48, device=rank)
    dummy_hand_transl = torch.randn(batch_size, 3, device=rank)
    dummy_hand_shape = torch.randn(batch_size, 10, device=rank)
    dummy_joints_obj = torch.randn(batch_size, 21, 3, device=rank)
    dummy_hand_verts_obj = torch.randn(batch_size, 778, 3, device=rank)
    dummy_obj_rotmat = torch.eye(3, device=rank).unsqueeze(0).repeat(batch_size, 1, 1)
    
    memory_bps = torch.from_numpy(memory_bps).to(device=rank)
    memory_hand = torch.from_numpy(memory_hand).to(device=rank)
    memory_id = torch.from_numpy(memory_id).to(device=rank)
    
    dummy_inp = {
        Queries.OBJ_BPS: dummy_obj_bps,
        Queries.INTENT_VEC: dummy_intent_vec,
        Queries.OBJ_VERTS_OBJ_DS: dummy_obj_verts,
        Queries.OBJ_FACES: dummy_obj_faces,
        Queries.HAND_POSE_OBJ: dummy_hand_pose,
        Queries.HAND_TRANSL_OBJ: dummy_hand_transl,
        Queries.HAND_SHAPE: dummy_hand_shape,
        Queries.JOINTS_OBJ: dummy_joints_obj,
        Queries.HAND_VERTS_OBJ: dummy_hand_verts_obj,
        Queries.OBJ_ROTMAT: dummy_obj_rotmat,
        'intent_name': ['use'],
        'memory_bps': memory_bps,
        'memory_hand': memory_hand,
        'memory_intent': memory_id,
    }
    
    num_runs = 100
    num_warmup = 50
    
    # GPU 显存占用
    logger.info(f"\n{'='*60}")
    logger.info("GPU MEMORY")
    logger.info(f"{'='*60}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    mem_before = get_gpu_memory()
    
    with torch.no_grad():
        model(inp=dummy_inp, step_idx=0, mode='test', intent_name='use')
        torch.cuda.synchronize()
    
    mem_after = get_gpu_memory()
    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    logger.info(f"Memory before inference: {mem_before:.2f} MB")
    logger.info(f"Memory after inference:  {mem_after:.2f} MB")
    logger.info(f"Peak memory allocated: {mem_peak:.2f} MB")
    logger.info(f"{'='*60}\n")
    
    # Full Model 推理时间
    logger.info(f"\n{'='*60}")
    logger.info("INFERENCE TIME (Full Model)")
    logger.info(f"Warmup: {num_warmup} runs, Measure: {num_runs} runs")
    logger.info(f"{'='*60}")
    
    with torch.no_grad():
        for _ in range(num_warmup):
            model(inp=dummy_inp, step_idx=0, mode='test', intent_name='use')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            model(inp=dummy_inp, step_idx=0, mode='test', intent_name='use')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # ms
    
    full_time = sum(times) / len(times)
    
    logger.info(f"Full Model: {full_time:.2f}ms per sample (avg of {num_runs} runs)")
    logger.info(f"Min: {min(times):.2f}ms, Max: {max(times):.2f}ms, Std: {((sum((t-full_time)**2 for t in times)/len(times))**0.5):.2f}ms")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    exp_time = time.time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Calculating GPU Memory and Inference Time <====")
    main(cfg, arg, exp_time)
