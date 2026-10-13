import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
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
from lib.viztools.utils import ColorsMap as CMap
from lib.viztools.viz_o3d_utils import VizContext
import open3d as o3d
from sklearn.cluster import MiniBatchKMeans
import numpy as np


from oikit.oi_shape.utils import ALL_INTENT

from lib.datasets import create_dataset
from torch.utils.data.distributed import DistributedSampler

n_samples = 20

def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    dataset = builder.build_dataset(split_to_cfg[arg.data_split], data_preset=cfg.DATA_PRESET)
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
                             shuffle=False,
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


    # train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    # # train_sampler = DistributedSampler(train_data, shuffle=True)
    # memory_loader = DataLoader(train_data,
    #                           batch_size=32,
    #                           # shuffle=(train_sampler is None),
    #                           shuffle=False,
    #                           num_workers=int(arg.workers),
    #                           pin_memory=True,
    #                           drop_last=True,
    #                           # sampler=train_sampler,
    #                           worker_init_fn=worker_init_fn,
    #                           collate_fn=grasp_data_collate,
    #                           persistent_workers=(int(arg.workers) > 0))


    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model: GrabNet = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")


    ref_mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=cfg.DATA_PRESET.CENTER_IDX,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )



    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    logger.info("You are viz GrabNet's prediciton")
    model.eval()
    from lib.utils.pcd import point2point_signed
    with torch.no_grad():
        # K = 50  # 聚类中心的数量
        # samples_per_center = 15  # 每个聚类中心最近的样本数
        # epochs = 10  # 迭代次数

        # kmeans = MiniBatchKMeans(n_clusters=K, random_state=0)

        # # 用于收集每个聚类中心最近样本的列表
        # nearest_samples = {
        #     'bps': {i: [] for i in range(K)},
        #     'hand': {i: [] for i in range(K)},
        #     'id': {i: [] for i in range(K)},
        #     'ov': {i: [] for i in range(K)},
        #     'transl': {i: [] for i in range(K)},
        #     'anchor': {i: [] for i in range(K)}
        # }

        # # 假设 memory_loader 已定义并能正确加载数据
        # for epoch in range(epochs):
        #     print(f"Kmeans Epoch {epoch+1}/{epochs}")
        #     for m_bidx, m_batch in enumerate(memory_loader):
        #         m_batch = transform(m_batch)  # 假设 transform 函数已定义

        #         obj_bps = m_batch['obj_bps']
        #         hand_pose_obj = m_batch['hand_pose_obj']
        #         intent_id = m_batch['intent_id']
        #         obj_vert = m_batch['obj_verts_obj_ds']
        #         hand_transl = m_batch['hand_transl_obj'].unsqueeze(1)
        #         anchor = torch.cat((obj_bps, intent_id.unsqueeze(1)), dim=-1)

        #         kmeans.partial_fit(anchor.cpu())

        #         # 计算到各聚类中心的距离
        #         distances_to_centers = kmeans.transform(anchor.cpu())
        #         cluster_idxs = np.argmin(distances_to_centers, axis=1)

        #         for i, cluster_idx in enumerate(cluster_idxs):
        #             current_obj_bps = obj_bps[i].cpu()
        #             current_hand_pose_obj = hand_pose_obj[i].cpu()
        #             current_intent_id = intent_id[i].cpu()
        #             current_obj_vert = obj_vert[i].cpu()
        #             current_transl = hand_transl[i].cpu()



        #             current_anchor = torch.cat((current_obj_bps, current_intent_id.unsqueeze(0)), dim=-1)

        #             if len(nearest_samples['anchor'][cluster_idx]) < samples_per_center:
        #                 nearest_samples['bps'][cluster_idx].append(current_obj_bps)
        #                 nearest_samples['hand'][cluster_idx].append(current_hand_pose_obj)
        #                 nearest_samples['id'][cluster_idx].append(current_intent_id)
        #                 nearest_samples['ov'][cluster_idx].append(current_obj_vert)
        #                 nearest_samples['transl'][cluster_idx].append(current_transl)
        #                 nearest_samples['anchor'][cluster_idx].append(current_anchor)
        #             else:
        #                 distances = [np.linalg.norm(current_anchor - np.array(sample)) for sample in nearest_samples['anchor'][cluster_idx]]
        #                 max_distance_idx = np.argmax(distances)
        #                 if distances[max_distance_idx] > distances_to_centers[i, cluster_idx]:
        #                     nearest_samples['bps'][cluster_idx][max_distance_idx] = current_obj_bps
        #                     nearest_samples['hand'][cluster_idx][max_distance_idx] = current_hand_pose_obj
        #                     nearest_samples['id'][cluster_idx][max_distance_idx] = current_intent_id
        #                     nearest_samples['ov'][cluster_idx][max_distance_idx] = current_obj_vert
        #                     nearest_samples['transl'][cluster_idx][max_distance_idx] = current_transl
        #                     nearest_samples['anchor'][cluster_idx][max_distance_idx] = current_anchor

        # # 转换为张量
        # memory_bps = torch.stack([sample for center in range(K) for sample in nearest_samples['bps'][center]])
        # memory_hand = torch.stack([sample for center in range(K) for sample in nearest_samples['hand'][center]])
        # memory_id = torch.stack([sample for center in range(K) for sample in nearest_samples['id'][center]])
        # memory_ov =  torch.stack([sample for center in range(K) for sample in nearest_samples['ov'][center]])
        # memory_transl =  torch.stack([sample for center in range(K) for sample in nearest_samples['transl'][center]])

        # # 保存数据到 .npy 文件
        # np.save('./memory_kmeans/memory_bps.npy', memory_bps.numpy())
        # np.save('./memory_kmeans/memory_hand.npy', memory_hand.numpy())
        # np.save('./memory_kmeans/memory_id.npy', memory_id.numpy())
        # np.save('./memory_kmeans/memory_ov.npy', memory_ov.numpy())
        # np.save('./memory_kmeans/memory_transl.npy', memory_transl.numpy())

        # print("finish kmeans for gvit1")
        # exit()

        # memory_bps = memory_bps.clone()
        # memory_hand = memory_hand.clone()
        # memory_id = memory_id.clone()
        # memory_ov = memory_ov.clone()


        memory_bps = np.load('./memory_kmeans/memory_bps.npy')
        memory_hand = np.load('./memory_kmeans/memory_hand.npy')
        memory_id = np.load('./memory_kmeans/memory_id.npy')
        memory_ov = np.load('./memory_kmeans/memory_ov.npy')
        memory_transl = np.load('./memory_kmeans/memory_transl.npy')

        # 将NumPy数组转换为PyTorch Tensor
        memory_bps = torch.from_numpy(memory_bps)
        memory_hand = torch.from_numpy(memory_hand)
        memory_id = torch.from_numpy(memory_id)
        memory_ov = torch.from_numpy(memory_ov)
        memory_transl = torch.from_numpy(memory_transl)


        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
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
            # print(1111,arg.intent)
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test")  # GrabNet, HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)  # IntGen

            # print(cfg.MODEL)
            # print(123,prd.keys())
            # exit()

            h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
            # h_verts_r = inp["Refine.hand_verts"].detach().cpu().numpy()
            h_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()

            h_faces = mano_layer.th_faces.detach().cpu().numpy()
            o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach().cpu().numpy()
            o_norms_ds = inp[Queries.OBJ_NORMALS_OBJ_DS].detach().cpu().numpy()

            #####
            
            obj_bps = inp[Queries.OBJ_BPS]
            device = obj_bps.device
            x_intent = torch.tensor([int(ALL_INTENT[arg.intent])]).to(obj_bps.device)
            x_intent = x_intent.unsqueeze(1).expand_as(memory_bps)
            mem_intent_id = memory_id.unsqueeze(1).expand_as(memory_bps)

            ## weaken intent 
            # x_intent = x_intent.unsqueeze(1).expand(750,1)
            # mem_intent_id = memory_id.unsqueeze(1).expand(750,1)
            
            memory_bps = memory_bps.to(device)
            mem_intent_id = mem_intent_id.to(device)


            cos_sim = F.cosine_similarity(torch.cat((obj_bps.expand_as(memory_bps),x_intent),dim=-1),torch.cat((memory_bps,mem_intent_id),dim=-1), dim=1).unsqueeze(0)
            # cos_sim = F.cosine_similarity(obj_bps.expand_as(memory_bps),memory_bps, dim=1).unsqueeze(0)
            # cos_sim = torch.sqrt(torch.sum((obj_bps-memory_bps)**2,dim=-1)).unsqueeze(0)
            # cos_sim = F.cosine_similarity(obj_bps.unsqueeze(1).expand_as(mem_bps), mem_bps, dim=2).unsqueeze(1)
            cos_sim_squeezed = cos_sim  # [32, 384]
            # print(cos_sim_squeezed.shape)
            values, indices = torch.topk(cos_sim_squeezed, 5, dim=1)  # return index
            # print(234,indices,values)


            refence = indices.squeeze(0).detach().cpu().numpy()[0]
            reference_ov = memory_ov[refence].detach().cpu().numpy()
            reference_hand = memory_hand[refence]
            reference_intent = memory_id[refence].detach().cpu().numpy()
            reference_transl = memory_transl[refence].unsqueeze(0).detach().cpu().numpy()
            # print(333,reference_transl.shape)
            # print(reference_hand.shape)
            # exit()
            reference_mano = ref_mano_layer(reference_hand.unsqueeze(0))
            reference_mano_vert = reference_mano.verts + reference_transl
            # + reference_transl 

            # print(reference_hand.shape)
            # print(reference_mano.verts.shape)
            # print(reference_mano_vert.shape)
            # print(h_verts_c.shape)
            # exit()





            # ref_pose = reference_hand.reshape(1,48)
            # ref_tran = reference_hand[:3].reshape(1,3)
            # mano_out = mano_layer(ref_pose)
            # print(mano_out.shape)
            # exit()
            # res["hand_verts"] = mano_out.verts + ref_tran  # (B, 778, 3)

            ### my vis

            obj_faces = inp[Queries.OBJ_FACES].detach().cpu().numpy()
            # obj_mesh = o3d.geometry.TriangleMesh()
            # obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
            # obj_mesh.vertices = o3d.utility.Vector3dVector(o_verts_ds.squeeze(0))
            # obj_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 0.0]] * len(np.asarray(o_verts_ds.squeeze(0)))))
            # obj_mesh.compute_vertex_normals()
            # obj_mesh.compute_triangle_normals()
            # viz_ctx.add_geometry(obj_mesh)

            # print(inp.keys())
            # exit()
            if dataset.data_mode == "handover":
                alt_hand_verts = inp[Queries.ALT_HAND_VERTS_OBJ].detach().cpu().numpy()

            n_items = h_verts_c.shape[0]
            # viz_ofs = np.array([0.2, 0.0, 0.0])
            viz_ofs = np.array([0.6, 0.0, 0.0])
            for i in range(n_items):
                show_next = False
                # print(reference_intent)
                viz_ctx.update_by_mesh("hand_coarse", h_verts_r[i], h_faces, vcolors=CMap["deepskyblue"])
                viz_ctx.update_by_mesh("hand_refine", reference_mano_vert[i]+ viz_ofs, h_faces, vcolors=CMap["tomato"])
                #### viz_ctx.update_by_mesh("obj_mesh", o_norms_ds[i], obj_faces.squeeze(0), vcolors=CMap["deepskyblue"]) 我自己的，不对，点云数目和face不对齐
                # viz_ctx.update_by_mesh("hand_refine", h_verts_r[i] + viz_ofs, h_faces, vcolors=CMap["tomato"])
                ov = o_verts_ds[i]
                on = o_norms_ds[i]
                viz_ctx.update_by_pc("obj", ov, on, pcolors=CMap["lime"])
                # viz_ctx.update_by_pc("obj_ofs", ov + viz_ofs, on, pcolors=CMap["lime"])
                viz_ctx.update_by_pc("obj_ofs", reference_ov + viz_ofs, on, pcolors=CMap["lime"])

                if dataset.data_mode == "handover":
                    viz_ctx.update_by_mesh("alt_hand", alt_hand_verts[i], h_faces, vcolors=CMap["blue"])
                    viz_ctx.update_by_mesh("alt_hand_ofs", alt_hand_verts[i] + viz_ofs, h_faces, vcolors=CMap["blue"])

                while not show_next:
                    viz_ctx.step()


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold","liftup"],
                        help="specify when use IntGen model")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)
