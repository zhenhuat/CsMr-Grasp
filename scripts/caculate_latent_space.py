import argparse
import os
from argparse import Namespace
from time import time
import sys
from collections import defaultdict

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

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

from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


def calculate_metrics(z_all, intent_labels_all):
    """
    计算潜空间的定量指标
    
    Args:
        z_all: 所有样本的潜变量 [N_samples, latentD]
        intent_labels_all: 对应的意图标签 [N_samples]
    
    Returns:
        dict: 包含所有指标的字典
    """
    results = {}
    n_intents = len(np.unique(intent_labels_all))
    
    # 先对潜变量进行 K-Means 聚类（无监督）
    kmeans = KMeans(n_clusters=n_intents, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z_all)
    
    # 1. Silhouette Score (轮廓系数)
    # 衡量簇内紧密度与簇间分离度的比值，范围 [-1, 1]，越高越好
    silhouette = silhouette_score(z_all, intent_labels_all)
    results['silhouette_score'] = silhouette
    
    # 2. Davies-Bouldin Index
    # 衡量簇内散度与簇间距离之比，越低越好
    davies_bouldin = davies_bouldin_score(z_all, intent_labels_all)
    results['davies_bouldin_index'] = davies_bouldin
    
    # 3. Normalized Mutual Information (NMI) - 修正版
    # 衡量聚类标签和真实意图标签的互信息，范围 [0, 1]，越高越好
    # 使用 K-Means 聚类结果与真实标签比较
    nmi = normalized_mutual_info_score(intent_labels_all, cluster_labels)
    results['nmi_clustering'] = nmi
    
    # 4. Adjusted Rand Index (ARI) - 新增指标
    # 衡量聚类标签和真实标签的相似度，范围 [-1, 1]，越高越好
    ari = adjusted_rand_score(intent_labels_all, cluster_labels)
    results['ari_clustering'] = ari
    
    # 额外的指标
    # 5. 计算每个意图簇的平均距离
    unique_labels = np.unique(intent_labels_all)
    intra_cluster_distances = {}
    inter_cluster_distances = {}
    
    for label in unique_labels:
        mask = intent_labels_all == label
        z_cluster = z_all[mask]
        if len(z_cluster) > 1:
            # 簇内平均距离
            intra_dist = np.mean(np.linalg.norm(
                z_cluster[:, np.newaxis, :] - z_cluster[np.newaxis, :, :], axis=2
            ))
            intra_cluster_distances[int(label)] = intra_dist
    
    # 簇间距离（簇中心之间的距离）
    cluster_centers = {}
    for label in unique_labels:
        mask = intent_labels_all == label
        cluster_centers[label] = np.mean(z_all[mask], axis=0)
    
    center_distances = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            dist = np.linalg.norm(cluster_centers[label1] - cluster_centers[label2])
            center_distances.append(dist)
            inter_cluster_distances[f'{label1}-{label2}'] = dist
    
    results['avg_intra_cluster_distance'] = np.mean(list(intra_cluster_distances.values()))
    results['avg_inter_cluster_distance'] = np.mean(center_distances) if center_distances else 0
    results['separation_ratio'] = results['avg_inter_cluster_distance'] / (results['avg_intra_cluster_distance'] + 1e-8)
    
    # K-Means 聚类质量
    results['kmeans_inertia'] = kmeans.inertia_  # 簇内平方和
    
    return results


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    dataset = builder.build_dataset(split_to_cfg[arg.data_split], data_preset=cfg.DATA_PRESET)
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model: GrabNet = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    model.eval()
    
    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")

    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    # 存储所有样本的潜变量和标签
    z_all = []
    intent_labels_all = []
    intent_names_all = []
    
    logger.info("=" * 60)
    logger.info("Latent Space Quantitative Analysis")
    logger.info("=" * 60)
    
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Collecting Latent")
        for step_idx, inp in enumerate(data_iter):
            inp = transform(inp)
            
            # 使用指定的 intent 或数据集的 intent
            if arg.intent is not None:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test", intent_name=arg.intent, return_latent=True)
            else:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test", intent_name=inp['intent_name'][0], return_latent=True)
            
            # 获取 encoder 输出的原始潜变量（未经调整）
            # 这就是 encoder 把测试集实例压缩成的 latent
            z = prd["Coarse.z_mean"].detach().cpu().numpy()  # [B, latentD]
            z_all.append(z)
            
            # 获取意图标签
            if arg.intent is not None:
                # 使用指定的 intent
                labels = np.full(z.shape[0], intent_to_label[arg.intent])
                names = [arg.intent] * z.shape[0]
            else:
                # 使用数据集的真实标签
                intent_vec = inp["intent_vec"].cpu().numpy()
                labels = np.argmax(intent_vec, axis=1)
                names = inp['intent_name']
            
            intent_labels_all.append(labels)
            intent_names_all.extend(names)
        
        # 合并所有 batch
        z_all = np.concatenate(z_all, axis=0)
        intent_labels_all = np.concatenate(intent_labels_all, axis=0)
        
        logger.info(f"Total samples collected: {len(z_all)}")
        logger.info(f"Latent dimension: {z_all.shape[1]}")
        logger.info(f"Number of intents: {len(np.unique(intent_labels_all))}")
        logger.info(f"Intent distribution: {np.bincount(intent_labels_all)}")
        
        logger.info(f"\nLatent Space Statistics:")
        logger.info(f"  Mean: {np.mean(z_all):.6f}")
        logger.info(f"  Std: {np.std(z_all):.6f}")
        logger.info(f"  Min: {np.min(z_all):.6f}")
        logger.info(f"  Max: {np.max(z_all):.6f}")
        
        # 按意图分别统计
        for label in np.unique(intent_labels_all):
            mask = intent_labels_all == label
            z_cluster = z_all[mask]
            logger.info(f"  Intent {label}: mean={np.mean(z_cluster):.6f}, std={np.std(z_cluster):.6f}")
        
        # 计算指标
        logger.info("\n" + "=" * 60)
        logger.info("Clustering Quality Metrics")
        logger.info("=" * 60)
        
        metrics = calculate_metrics(z_all, intent_labels_all)
        
        logger.info(f"\nSilhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        logger.info(f"NMI: {metrics['nmi_clustering']:.4f}")
        logger.info(f"ARI: {metrics['ari_clustering']:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Cluster Separation Metrics")
        logger.info("=" * 60)
        
        logger.info(f"\nAvg Intra-cluster Distance: {metrics['avg_intra_cluster_distance']:.4f}")
        logger.info(f"Avg Inter-cluster Distance: {metrics['avg_inter_cluster_distance']:.4f}")
        logger.info(f"Separation Ratio: {metrics['separation_ratio']:.4f}")
        
        # 可视化潜空间
        # (已移除可视化功能)
        
        # 保存结果到文件
        if arg.save_results:
            results_path = f"latent_space_metrics_{arg.intent if arg.intent else 'all'}.txt"
            with open(results_path, 'w') as f:
                f.write("Latent Space Quantitative Analysis Results\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total samples: {len(z_all)}\n")
                f.write(f"Latent dimension: {z_all.shape[1]}\n")
                f.write(f"Number of intents: {len(np.unique(intent_labels_all))}\n\n")
                
                f.write("Metrics:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Silhouette Score: {metrics['silhouette_score']:.4f}\n")
                f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}\n")
                f.write(f"NMI: {metrics['nmi_clustering']:.4f}\n")
                f.write(f"ARI: {metrics['ari_clustering']:.4f}\n")
                f.write(f"Avg Intra-cluster Distance: {metrics['avg_intra_cluster_distance']:.4f}\n")
                f.write(f"Avg Inter-cluster Distance: {metrics['avg_inter_cluster_distance']:.4f}\n")
                f.write(f"Separation Ratio: {metrics['separation_ratio']:.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Interpretation:\n")
                f.write("=" * 60 + "\n")
                f.write("- Silhouette Score: measures cluster cohesion vs separation (higher=better)\n")
                f.write("- Davies-Bouldin Index: ratio of intra-cluster scatter to inter-cluster distance (lower=better)\n")
                f.write("- NMI/ARI: agreement between clustering and ground truth (higher=better)\n")
                f.write("- Separation Ratio: inter-cluster / intra-cluster distance (higher=better)\n")
            
            logger.info(f"\nResults saved to {results_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Analysis Complete!")
        logger.info("=" * 60)


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
                        choices=["use", "hold"],
                        help="specify intent for conditional generation")
    parser.add_argument("--save_results", action="store_true", help="save metrics to file")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)
