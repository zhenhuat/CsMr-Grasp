import argparse
import os
import sys
import time
from argparse import Namespace

import torch
from torch.nn.parallel import DataParallel as DP

sys.path.append('/home/robin/fareal/icassp_grasp/')

import lib.models
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.logger import logger


def count_params(module):
    return sum(p.numel() for p in module.parameters())


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
    }
    
    num_runs = 100
    num_warmup = 20
    
    logger.info(f"\n{'='*60}")
    logger.info("WARMUP")
    logger.info(f"{'='*60}")
    with torch.no_grad():
        for _ in range(num_warmup):
            model(inp=dummy_inp, step_idx=0, mode='test', intent_name='use')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    logger.info("Warmup done")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"\n{'='*60}")
    logger.info("INFERENCE TIME (per sample)")
    logger.info(f"{'='*60}")
    
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            model(inp=dummy_inp, step_idx=0, mode='test', intent_name='use')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
    full_time = (end - start) / num_runs
    logger.info(f"Full Model: {full_time*1000:.2f}ms")
    logger.info(f"{'='*60}\n")
    
    try:
        from thop import profile
        
        class FullModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, obj_bps, intent_vec, obj_verts_ds, obj_faces, hand_pose, hand_transl, hand_shape):
                inp = {
                    Queries.OBJ_BPS: obj_bps,
                    Queries.INTENT_VEC: intent_vec,
                    Queries.OBJ_VERTS_OBJ_DS: obj_verts_ds,
                    Queries.OBJ_FACES: obj_faces,
                    Queries.HAND_POSE_OBJ: hand_pose,
                    Queries.HAND_TRANSL_OBJ: hand_transl,
                    Queries.HAND_SHAPE: hand_shape,
                    Queries.JOINTS_OBJ: torch.zeros(1, 21, 3, device=obj_bps.device),
                    Queries.HAND_VERTS_OBJ: torch.randn(1, 778, 3, device=obj_bps.device),
                    Queries.OBJ_ROTMAT: torch.eye(3, device=obj_bps.device).unsqueeze(0),
                }
                return self.model(inp=inp, step_idx=0, mode='test', intent_name='use')
        
        wrapper = FullModelWrapper(model.module)
        
        macs, params = profile(wrapper, 
                              inputs=(dummy_obj_bps, dummy_intent_vec, 
                                     dummy_obj_verts, dummy_obj_faces,
                                     dummy_hand_pose, dummy_hand_transl, 
                                     dummy_hand_shape),
                              verbose=False)
        flops = macs * 2
        
        logger.info(f"{'='*60}")
        logger.info("FLOPS")
        logger.info(f"{'='*60}")
        logger.info(f"GFLOPs: {flops / 1e9:.2f}")
        logger.info(f"{'='*60}\n")
        
    except ImportError:
        logger.info("thop not available, skipping FLOPS calculation")


if __name__ == '__main__':
    exp_time = time.time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Calculating Parameters, FLOPs and Inference Time <====")
    main(cfg, arg, exp_time)
