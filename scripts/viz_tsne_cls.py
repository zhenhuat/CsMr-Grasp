import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean

import torch.nn.functional as F

import sys
sys.path.append('/home/robin/fareal/icassp_grasp/')

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
import matplotlib.pyplot as plt
import seaborn as sns

from oikit.oi_shape.utils import ALL_INTENT

from lib.datasets import create_dataset
from torch.utils.data.distributed import DistributedSampler
import re

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

    memory_loader = DataLoader(train_data,
                             batch_size=128,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

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

    # viz_ctx = VizContext(non_block=True)
    # viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    # viz_ctx.register_key_callback("D", next_sample)
    logger.info("You are viz GrabNet's prediction")
    model.eval()
    from lib.utils.pcd import point2point_signed

    all_h_verts_c = []
    all_intent_ids = []

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

        for intent in ['hold', 'use']:
            arg.intent = intent
            data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0} - Intent: {intent}")
            for i, inp in enumerate(data_iter):
                inp = transform(inp)
                inp['memory_bps'] = memory_bps
                inp['memory_hand'] = memory_hand
                inp['memory_intent'] = memory_id

                obj_list = inp['sample_identifier'][0]
                obj_match = re.search(r'CATE_(.*?)_OBJ',obj_list)
                obj_name = obj_match.group(1)

                if arg.intent is None:
                    prd, _ = model(inp=inp, step_idx=i, mode="test")
                else:
                    prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)

                # h_verts_c = inp['hand_verts_obj'].detach().cpu().numpy()
                # h_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
                # h_verts_c = prd["Refine.hand_verts"].detach().cpu().numpy()
                # all_h_verts_c.append(h_verts_c)
                # all_intent_ids.extend([intent+' '+obj_name] * h_verts_c.shape[0])
                # print(inp.keys())
                # exit()

                ### for gt but samples is less
                if intent == inp['intent_name'][0]:
                    # h_verts_c = inp['hand_verts_obj'].detach().cpu().numpy()
                    h_verts_c = prd["Refine.hand_verts"].detach().cpu().numpy()
                    all_h_verts_c.append(h_verts_c)
                    all_intent_ids.extend([intent+' '+obj_name] * h_verts_c.shape[0])
                else:
                    continue


    all_h_verts_c = np.concatenate(all_h_verts_c, axis=0)
    # all_h_verts_c = np.tile(all_h_verts_c, (2, 1, 1))
    # all_intent_ids.extend(all_intent_ids)
    # print(all_h_verts_c.shape)
    # print(all_intent_ids.shape)
    # exit()
    if all_h_verts_c.shape[0]< 20:
        num_samples = all_h_verts_c.shape[0]-1
    else:
        num_samples = 9

    tsne = TSNE(n_components=2, random_state=0,perplexity=num_samples)
    tsne_results = tsne.fit_transform(all_h_verts_c.reshape(all_h_verts_c.shape[0], -1))

    plt.figure(figsize=(16, 8))

    # Use a refined color palette
    # palette = sns.color_palette("deep", len(np.unique(all_intent_ids)))
    palette = sns.color_palette("tab20", len(np.unique(all_intent_ids)))
    # intent_to_color = {intent: palette[i] for i, intent in enumerate(np.unique(all_intent_ids))}
    intent_to_color = {intent: palette[i] for i, intent in enumerate(np.unique(all_intent_ids))}



    # 手动分配颜色
    # colors = [intent_to_color[intent] for intent in all_intent_ids]

    # # Create the scatter plot with enhanced aesthetics
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s=100, alpha=0.5, edgecolor='w')  # 修改部分


    # Create the scatter plot with enhanced aesthetics
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1], hue=all_intent_ids,
        palette=intent_to_color, s=100, alpha=0.5, edgecolor='w'
    )

    # Calculate and plot the centers for each class
    unique_intents = np.unique(all_intent_ids)
    centers = {}
    for intent in unique_intents:
        indices = np.where(np.array(all_intent_ids) == intent)[0]
        center = np.mean(tsne_results[indices], axis=0)
        centers[intent] = center
        # plt.scatter(center[0], center[1], s=400, marker='*')
        plt.scatter(center[0], center[1], s=200, marker='*',color=intent_to_color[intent], label=f'{intent}')

    # Calculate distances between centers
    distances = []
    for i, intent1 in enumerate(unique_intents):
        for j, intent2 in enumerate(unique_intents):
            if i < j:
                distance = euclidean(centers[intent1], centers[intent2])
                distances.append(distance)
                print(f"Distance between {intent1} center and {intent2} center: {distance:.3f}")

    # Calculate and print the average distance
    if distances:
        average_distance = np.mean(distances)
        print(f"Average distance between centers: {average_distance:.3f}")

    # Title and labels with improved fonts and sizes
    # plt.title('t-SNE Visualization of SNS-Grasp', fontsize=20, weight='bold')
    # plt.xlabel('t-SNE component 1', fontsize=18, weight='bold')
    # plt.ylabel('t-SNE component 2', fontsize=18, weight='bold')

    # Add text with the average distance in the bottom left corner
    # plt.text(-0.1, -0.1, f'Average distance between centers: {average_distance:.3f}', 
    #      horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=18, color='black')

    # Customize the axes
    plt.tick_params(axis='both', which='major', labelsize=14, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=12, width=2)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    # Legend with improved placement and style
    # plt.legend(title='Intent', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=16)
    # plt.legend(
    # # title='Intent',
    # loc='upper center',      # 初始定位点（上方中心）
    # bbox_to_anchor=(0.5, -0.01),  # 向下偏移（x=水平居中，y=图片下方）
    # ncol=9,       # 列数=标签数（强制水平排列）
    # fontsize=10,
    # title_fontsize=16,
    # frameon=False           # 可选：去掉图例边框
    # )

    plt.legend(
        title='Intent',
        loc='upper right',          # 初始定位点（左上角）
        # bbox_to_anchor=(1, 1),     # 图例的锚点（1=右边界，1=顶部）
        ncol=1,                    # 单列显示
        fontsize=9,
        title_fontsize=12,
        frameon=True,             # 可选：去掉图例边框
        borderaxespad=0.5,          # 图例与坐标轴边框的间距（可选调整）
        framealpha=0.0
    )
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Remove borders and gridlines for a cleaner look
    sns.despine(left=False, bottom=False, top=False, right=False)
    plt.grid(False)

    # Save the plot as a file
    plt.savefig('./tsne_plot_CsMr-Grasp.pdf', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
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
