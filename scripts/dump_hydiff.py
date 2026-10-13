import argparse
import hashlib
import os
import pickle
from argparse import Namespace
from time import time
from typing import List

import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

import torch.nn.functional as F

import sys
sys.path.append('/home/robin/fareal/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_data import grasp_data_collate
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.recorder import Recorder


n_samples = 20

def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    dump_path = os.path.join(recorder.dump_path, "results")
    os.makedirs(dump_path, exist_ok=True)
    logger.info(f"Dumping results to {dump_path}")

    def dumper(prd, inp, step_idx, **kwargs):
        obj_id: List[str] = inp[Queries.OBJ_ID]
        obj_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach().cpu().numpy()
        obj_rotmat = inp[Queries.OBJ_ROTMAT].detach().cpu().numpy()
        if Queries.ALT_HAND_TRANSL_OBJ in inp:
            alt_hand_verts = inp[Queries.ALT_HAND_VERTS_OBJ].detach().cpu().numpy()

        hand_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
        ##gt
        # hand_verts_c = inp['hand_verts_obj'].detach().cpu().numpy()
        hand_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
        n_samples = hand_verts_c.shape[0]  # batch
        for i in range(n_samples):
            # calcluate the md5 of hand_verts_r, and only use the first 10 hashcode
            grasp_hash = hashlib.md5(hand_verts_r[i].tobytes()).hexdigest()[:10]
            sample_fname = f"{obj_id[i]}_{grasp_hash}.pkl"
            sample_content = {
                "obj_id": obj_id[i],
                "obj_rotmat": obj_rotmat[i],
                "obj_verts_ds": obj_verts_ds[i],
                "hand_verts_c": hand_verts_c[i],
                "hand_verts_r": hand_verts_r[i],
            }
            if Queries.ALT_HAND_TRANSL_OBJ in inp:
                sample_content["alt_hand_verts"] = alt_hand_verts[i]
            with open(os.path.join(dump_path, sample_fname), "wb") as f:
                pickle.dump(sample_content, f)
        return True

    dataset = builder.build_dataset(split_to_cfg[arg.data_split], data_preset=cfg.DATA_PRESET)
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
                             shuffle=True,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    train_data = builder.build_dataset(split_to_cfg['train'], data_preset=cfg.DATA_PRESET)
    memory_loader =  DataLoader(train_data,
                             batch_size=128,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)


    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    model.eval()
    with torch.no_grad():
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np

        # K = 50  # 聚类中心的数量
        # samples_per_center = 10  # 每个聚类中心最近的样本数

        # kmeans = MiniBatchKMeans(n_clusters=K, random_state=0)

        # # 用于收集每个聚类中心最近5个样本的列表
        # nearest_samples = {
        #     'bps': {i: [] for i in range(K)},
        #     'hand': {i: [] for i in range(K)},
        #     'id': {i: [] for i in range(K)},
        #     'anchor': {i: [] for i in range(K)}
        # }

        # for  m_bidx,m_batch in enumerate(memory_loader):
        #     m_batch = transform(m_batch)
        #     obj_bps = m_batch['obj_bps']
        #     hand_pose_obj = m_batch['hand_pose_obj']
        #     intent_id = m_batch['intent_id']

        #     anchor =  torch.cat((obj_bps,intent_id.unsqueeze(1)),dim=-1)
        #     # 使用当前样本更新聚类中心
        #     # print(obj_bps.shape)
        #     # exit()
        #     kmeans.partial_fit(anchor.cpu())

        #     # 对每个样本，计算到各聚类中心的距离
        #     distances_to_centers = kmeans.transform(anchor.cpu())
        #     cluster_idxs = np.argmin(distances_to_centers, axis=1)

        #     for i, cluster_idx in enumerate(cluster_idxs):
        #         # 确定当前样本的各项数据
        #         current_obj_bps = obj_bps[i].cpu()  # 假设这里 obj_bps 是一个批量的数据
        #         current_hand_pose_obj = hand_pose_obj[i].cpu()  # 同上
        #         current_intent_id = intent_id[i].cpu()  # 同上

        #         current_anchor = torch.cat((current_obj_bps,current_intent_id.unsqueeze(0)),dim=-1)

        #         # 更新逻辑保持不变，只是现在是基于每个样本而不是整体
        #         if len(nearest_samples['anchor'][cluster_idx]) < samples_per_center:
        #             nearest_samples['bps'][cluster_idx].append(current_obj_bps)
        #             nearest_samples['hand'][cluster_idx].append(current_hand_pose_obj)
        #             nearest_samples['id'][cluster_idx].append(current_intent_id)
        #             nearest_samples['anchor'][cluster_idx].append(current_anchor)
        #         else:
        #             # 如果该中心已有足够的样本，替换最远的样本
        #             distances = [np.linalg.norm(current_anchor - np.array(sample)) for sample in nearest_samples['anchor'][cluster_idx]]
        #             max_distance_idx = np.argmax(distances)
        #             # 这里需要更新的是当前样本到聚类中心的距离和最远样本到聚类中心的距离的比较
        #             if distances[max_distance_idx] > distances_to_centers[i, cluster_idx]:
        #                 nearest_samples['bps'][cluster_idx][max_distance_idx] = current_obj_bps
        #                 nearest_samples['hand'][cluster_idx][max_distance_idx] = current_hand_pose_obj
        #                 nearest_samples['id'][cluster_idx][max_distance_idx] = current_intent_id
        #                 nearest_samples['anchor'][cluster_idx][max_distance_idx] = current_anchor
        # # print(nearest_samples['id'])
        # # 聚类结束，构建最终张量
        # final_bps = [sample for center in range(K) for sample in nearest_samples['bps'][center]]
        # final_hand = [sample for center in range(K) for sample in nearest_samples['hand'][center]]
        # final_id = [sample for center in range(K) for sample in nearest_samples['id'][center]]

        # # 转换为张量
        # memory_bps = torch.stack(final_bps)
        # memory_hand = torch.stack(final_hand)
        # memory_id = torch.stack(final_id)

        # memory_bps = memory_bps.clone().detach()
        # memory_hand = memory_hand.clone().detach()
        # memory_id = memory_id.clone().detach()

        # memory_bps = np.load('./memory/memory_bps.npy')
        # memory_hand = np.load('./memory/memory_hand.npy')
        # memory_id = np.load('./memory/memory_id.npy')
        memory_bps = np.load('./memory//memory_bps.npy')
        memory_hand = np.load('./memory//memory_hand.npy')
        memory_id = np.load('./memory//memory_id.npy')
        # memory_ov = np.load('./memory/memory_ov.npy')
        # memory_transl = np.load('./memory/memory_transl.npy')

        # 将NumPy数组转换为PyTorch Tensor
        memory_bps = torch.from_numpy(memory_bps)
        memory_hand = torch.from_numpy(memory_hand)
        memory_id = torch.from_numpy(memory_id)
        # memory_ov = torch.from_numpy(memory_ov)
        # memory_transl = torch.from_numpy(memory_transl)


        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
        memorybar =  etqdm(memory_loader, rank=rank)
        for i, inp in enumerate(data_iter):
            inp = transform(inp)
            ## retrival memory
            # batch_obj_bps = inp['obj_bps']
            # distances = []
            # memory_hand = []
            # for m_bidx,m_batch in enumerate(memorybar):
            #     m_batch = transform(m_batch)
            #     m_bps = m_batch['obj_bps']
            #     m_hand = m_batch['hand_pose_obj']
            #     distances.append(F.pairwise_distance(batch_obj_bps.unsqueeze(1), m_bps.unsqueeze(0), p=2))

            #     memory_hand.append(m_hand)
            # distances = torch.cat(distances, dim=1)
            # memory_hand = torch.cat(memory_hand, dim=0)
            # _, indices = torch.topk(distances, k=n_samples, dim=1, largest=False)
            # nearest_memory_samples_indices = indices[:, :n_samples]
            # selected_memory_hand = memory_hand[indices[:, :n_samples], :]
            # inp['memory'] = selected_memory_hand
            inp['memory_bps'] = memory_bps
            inp['memory_hand'] = memory_hand
            inp['memory_intent'] = memory_id
            ## end retrival memory
            # print(inp['obj_bps'].shape)
            # exit()
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test", callback=dumper)  # GrabNet, HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent, callback=dumper)  # IntGen
                # prd["Coarse.hand_verts"] = inp['hand_verts_obj']


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold","liftup"],
                        help="specify this variable when using the IntGen model.")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)
