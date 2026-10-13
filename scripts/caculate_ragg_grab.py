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

from oikit.oi_shape.utils import ALL_INTENT


import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
import open3d as o3d
# n_samples = 20


from torch.utils.data.dataloader import default_collate

import torch.nn as nn
from collections import defaultdict

# 假设name_to_rgb已定义在某处
name_to_rgb = {'yellow': [1, 1, 0]}  # 示例定义



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


class LoadData(Dataset):
    def __init__(self, dataset_dir, ds_name='train', dtype=torch.float32, only_params=False, load_on_ram=False, subjects=None, object_types=None, object_instances=None, object_actions=None):
        super().__init__()

        self.only_params = only_params
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path, 'grabnet_%s.npz' % ds_name))

        frame_names = np.load(os.path.join(dataset_dir, ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2] for name in self.frame_names])
        self.obj_names = [name.split('_')[0] for name in self.frame_objs]  # 提取并保存物体的名字（类型）

        # 过滤逻辑
        if subjects is not None and 'all' not in subjects:
            mask = np.isin(self.frame_sbjs, subjects)
            self.frame_names = self.frame_names[mask]
            self.frame_sbjs = self.frame_sbjs[mask]
            self.frame_objs = self.frame_objs[mask]
            self.obj_names = [self.obj_names[i] for i in range(len(self.obj_names)) if mask[i]]
            self.ds = self._filter_ds(mask)

        if object_types is not None or object_instances is not None or object_actions is not None:
            obj_types = np.asarray([name.split('_')[0] for name in self.frame_objs])
            obj_instances = np.asarray([name.split('_')[1] for name in self.frame_objs])
            obj_actions = np.asarray([name.split('_')[2] for name in self.frame_objs])

            mask = np.ones(len(self.frame_objs), dtype=bool)
            if object_types is not None and 'all' not in object_types:
                mask &= np.isin(obj_types, object_types)
            if object_instances is not None and 'all' not in object_instances:
                mask &= np.isin(obj_instances, object_instances)
            if object_actions is not None and 'all' not in object_actions:
                mask &= np.isin(obj_actions, object_actions)

            self.frame_names = self.frame_names[mask]
            self.frame_sbjs = self.frame_sbjs[mask]
            self.frame_objs = self.frame_objs[mask]
            self.obj_names = [self.obj_names[i] for i in range(len(self.obj_names)) if mask[i]]
            self.ds = self._filter_ds(mask)

        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        bps_fname = os.path.join(dataset_dir, 'bps.npz')
        self.bps = torch.from_numpy(np.load(bps_fname)['basis']).to(dtype)

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs = torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        self.load_on_ram = load_on_ram

        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

    def _np2torch(self, ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k: torch.tensor(data[k]) for k in data.files}
        return data_torch

    def _filter_ds(self, mask):
        filtered_ds = {}
        for key in self.ds.keys():
            filtered_ds[key] = self.ds[key][mask]
        return filtered_ds

    def load_disk(self, idx):
        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = [self._np2torch(f) for f in frame_names]
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]

    def __getitem__(self, idx):
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params and not self.load_on_ram:
            form_disk = self.load_disk(idx)
            data_out.update(form_disk)

        # 加载并变换网格数据
        objname = self.obj_names[idx]
        mesh_path = os.path.join('/data/fareal/EXTRACT/tools/object_meshes/contact_meshes/', objname + '.ply')
        try:
            obj_mesh = Mesh(filename=mesh_path, vc=name_to_rgb['yellow'])
            rot_mat = data_out['root_orient_obj_rotmat'].cpu().numpy().reshape(3, 3).T
            transl = data_out['trans_obj'].cpu().numpy()
            obj_mesh.rotate_vertices(rot_mat)
            obj_mesh.v += transl

            # 将网格数据存储为顶点和面信息，并转换类型
            data_out['mesh_vertices'] = torch.tensor(obj_mesh.v, dtype=torch.float32)
            data_out['mesh_faces'] = torch.tensor(obj_mesh.f.astype(np.int32), dtype=torch.int32)
        except Exception as e:
            print(f"Failed to load mesh for {objname}: {e}")

        # 添加 object_type 和 frame_names 到返回的数据中
        data_out['object_type'] = objname
        data_out['frame_names'] = self.frame_names[idx]

        return data_out



def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    dump_path = os.path.join(recorder.dump_path, "results")
    os.makedirs(dump_path, exist_ok=True)
    logger.info(f"Dumping results to {dump_path}")

    def dumper(prd, inp, step_idx, **kwargs):
        # 获取文件前缀
        frame_names = inp['frame_names']
        obj_prefix = [f"{os.path.basename(os.path.dirname(path)).split('_')[0]}_{os.path.basename(path).split('.')[0]}" for path in frame_names]
        
        # object_type = obj_prefix  # 从 LoadData 中获取对象类型

        obj_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach().cpu().numpy()
        obj_rotmat = inp[Queries.OBJ_ROTMAT].detach().cpu().numpy()
        if Queries.ALT_HAND_TRANSL_OBJ in inp:
            alt_hand_verts = inp[Queries.ALT_HAND_VERTS_OBJ].detach().cpu().numpy()

        hand_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
        hand_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
        n_samples = hand_verts_c.shape[0]  # batch

        n_data = inp['obj_bps'].shape[0]
        bps_object = inp['obj_bps']


        # values_np = torch.round(values * 1e8) / 1e8
        # values_np = values_np.cpu().numpy()
        np.set_printoptions(precision=8, suppress=True)

        for i in range(n_samples):
            # 计算 hand_verts_r 的 md5，并且只使用前10个字符
            grasp_hash = hashlib.md5(hand_verts_r[i].tobytes()).hexdigest()[:10]
            sample_fname = f"{obj_prefix[i]}.pkl"
            sample_content = {
                "obj_id": obj_prefix[i],
                "obj_rotmat": obj_rotmat[i].reshape(3,3),
                "obj_verts_ds": obj_verts_ds[i],
                "hand_verts_c": hand_verts_c[i],
                "hand_verts_r": hand_verts_r[i],
            }
            if Queries.ALT_HAND_TRANSL_OBJ in inp:
                sample_content["alt_hand_verts"] = alt_hand_verts[i]
            with open(os.path.join(dump_path, sample_fname), "wb") as f:
                pickle.dump(sample_content, f)
        return True


    # 使用自定义的 LoadData 类加载数据
    dataset_dir = '/data/fareal/EXTRACT/data/'
    ds_test = LoadData(
        dataset_dir=dataset_dir, 
        ds_name=arg.data_split,  # 使用命令行参数中的数据集划分
        subjects=['s1'], 
        object_types=['mug','camera','wineglass','toothpaste'], 
        # object_types=['camera'],
        object_instances=['pass'], 
        object_actions=['1']
    )

    data_loader = DataLoader(ds_test,
                             batch_size=arg.batch_size,
                             shuffle=True,  # 保持原有设置
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    # 保持 memory 加载部分不变
    train_data = builder.build_dataset(split_to_cfg['train'], data_preset=cfg.DATA_PRESET)

    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    model.eval()

    # 初始化分类器并加载权重
    classifier = ActionClassifier(input_channels=3)
    classifier.load_state_dict(torch.load("/home/robin/fareal/OakInk-Grasp-Generation-main/classification//best_classifier.pth"))  # 替换为你自己的路径
    classifier = DP(classifier).to(rank)
    classifier.eval()


    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    # 初始化每类的统计字典
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    memory_bps = np.load('./memory//memory_bps.npy')
    memory_hand = np.load('./memory//memory_hand.npy')
    memory_id = np.load('./memory//memory_id.npy')
    # memory_ov = np.load('./memory/memory_ov.npy')
    # memory_transl = np.load('./memory/memory_transl.npy')

    # 将NumPy数组转换为PyTorch Tensor
    memory_bps = torch.from_numpy(memory_bps)
    memory_hand = torch.from_numpy(memory_hand)
    memory_id = torch.from_numpy(memory_id)

    
    with torch.no_grad():
        from sklearn.cluster import MiniBatchKMeans
        # import numpy as np

        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
        for i, inp in enumerate(data_iter):
            # inp = transform(inp)
            # print(inp.keys())
            # print(inp['frame_names'])
            # exit()

            inp['memory_bps'] = memory_bps
            inp['memory_hand'] = memory_hand
            inp['memory_intent'] = memory_id

            inp['obj_bps'] = inp['bps_object'] 
            inp['obj_verts_obj_ds']=inp['verts_object']
            inp['hand_shape']=  torch.zeros(1, 10).to(inp['bps_object'].device)
            inp['obj_rotmat'] = inp['root_orient_obj_rotmat']

            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test")  # GrabNet, HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)  # IntGen


            # h_verts_r = prd["Coarse.hand_verts"].detach()  # [B, 778, 3]
            h_verts_r = prd["Refine.hand_verts"].detach()  # [B, 778, 3]
            # h_verts_r = inp[Queries.HAND_VERTS_OBJ].detach()  # [B, 778, 3]
            # o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach()  # [B, 4096, 3]
            o_verts_ds = inp['obj_verts_obj_ds'].detach().to(h_verts_r.device)

            o_verts_ds_doubled = torch.cat([o_verts_ds, o_verts_ds], dim=1)    # [B, 2N, 3]


            # print(o_verts_ds.shape)
            # exit()

            # print(h_verts_r.device,o_verts_ds.device)
            # exit()

            features = torch.cat([h_verts_r, o_verts_ds_doubled], dim=1)  # [B, 4874, 3]
            logits = classifier(features)
            _, preds = torch.max(logits, dim=1)

            labels_list = [intent_to_label[arg.intent]]
            labels = torch.tensor(labels_list, dtype=torch.long, device=logits.device)

            # 统计每一类的正确数
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()

                class_stats[label]['total'] += 1
                if pred == label:
                    class_stats[label]['correct'] += 1

        # 打印结果
        total_correct = sum(v['correct'] for v in class_stats.values())
        total_samples = sum(v['total'] for v in class_stats.values())

        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Correct predictions: {total_correct}")
        logger.info(f"Overall Accuracy: {total_correct / total_samples:.4f}")

        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            acc = stats['correct'] / stats['total']
            logger.info(f"Class [{label_to_intent[label]}] - Correct: {stats['correct']}, Total: {stats['total']}, Acc: {acc:.4f}")





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
