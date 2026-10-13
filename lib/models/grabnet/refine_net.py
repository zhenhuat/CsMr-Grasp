from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.structures import Meshes

from lib.datasets.grasp_query import Queries
from lib.metrics.basic_metric import LossMetric
from lib.models.model_abc import ModelABC
from lib.utils.builder import MODEL
from lib.utils.net_utils import load_weights
from lib.utils.pcd import point2point_signed
from lib.utils.transform import aa_to_rotmat

from .grabnet_arch import parms_decode


from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.anchorutils import anchor_load_driver
from manotorch.utils.anchorutils import get_region_palm_mask

# from ..pointnet2 import pointnet2_sem_seg_msg
from .torchkan import KANLinear


# pointnetwork2 = pointnet2_sem_seg_msg.get_model(40)
# checkpoint = torch.load('/data/yanbin/TZH/OakInk-Grasp-Generation-main/lib/models/pointnet2/best_model.pth')
# model_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
# pointnetwork2.load_state_dict(model_state_dict)
# pointnetwork2.eval() 



class ResBlock(nn.Module):

    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.fc1 = KANLinear(Fin, n_neurons,10)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = KANLinear(n_neurons, Fout,10)
        self.bn2 = nn.BatchNorm1d(Fout)
        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))
        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)
        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout
        if final_nl:
            return self.ll(Xout)

        return Xout




@MODEL.register_module()
class RefineNet(ModelABC):

    def __init__(self, cfg, in_size=778 + 16 * 6 + 3, h_size=512, n_iters=3, strict=False):
        super(RefineNet, self).__init__()
        self.stage = "Refine"
        self.cfg = cfg
        self.n_iters = n_iters
        self.h_size = h_size
        self.in_size = in_size

        self.vpe = torch.from_numpy(np.load(cfg.VPE_PATH)).to(torch.long)
        self.v_weights = torch.from_numpy(np.load(cfg.C_WEIGHT_PATH)).to(torch.float32)
        self.v_weights2 = torch.pow(self.v_weights, 1.0 / 2.5)
        self.contact_v = self.v_weights > 0.8
        self.kl_coef = cfg.KL_COEF

        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

        self._build_layers()
        self._build_loss()
        self._build_evaluation()
        load_weights(self, pretrained=cfg.PRETRAINED, strict=strict)

        # self.pointnetwork2 = get_model(40)
        # checkpoint_path = '/data/yanbin/TZH/OakInk-Grasp-Generation-main/lib/models/pointnet2/best_model.pth'
        # checkpoint = torch.load(checkpoint_path, map_location='cuda')
        # model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        # self.pointnetwork2.load_state_dict(model_state_dict)
        # self.pointnetwork2.eval()




    def _build_layers(self):
        h_size = self.h_size
        in_size = self.in_size 

        self.bn1 = nn.BatchNorm1d(778)
        self.rb1 = ResBlock(in_size, h_size)
        self.rb2 = ResBlock(in_size + h_size, h_size)
        self.rb3 = ResBlock(in_size + h_size, h_size)
        self.out_p = nn.Linear(h_size, 16 * 6)
        self.out_t = nn.Linear(h_size, 3)
        self.dout = nn.Dropout(0.3)
        self.actvf = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        # self.pointnetwork2 = pointnetwork2

        self.merged_vertex_assignment = anchor_load_driver("./assets")[2]  # vertex that belongs to a certain palm region

    def _build_loss(self):
        self.has_loss = True
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.loss_metric = LossMetric(self.cfg)

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq

    def forward(self, inp, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inp, step_idx=step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inp, step_idx=step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inp, step_idx=step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")



    def pairwise_dist(self, x, y):
        # print(x.shape,y.shape)
        # exit()
        zz = torch.matmul(x, y.t())
        rx = torch.sum(x**2, dim=1, keepdim=True).repeat(1, y.shape[0])
        ry = torch.sum(y**2, dim=1, keepdim=False).unsqueeze(0).repeat(x.shape[0], 1)
        P = rx + ry - 2 * zz
        return P

    def region_disjointness_metric(self, hand_verts, obj_verts, hand_region_assignment):
        hand_verts = hand_verts.squeeze(0).float()
        obj_verts = obj_verts.squeeze(0).float()

        tip_regions = [2, 5, 8, 11, 14]
        tip_only_weight = torch.zeros(17, dtype=torch.float64)
        tip_only_weight[tip_regions] = 1.0
        all_regions = list(range(17))
        res_list = []
        for region_id in all_regions:
            region_mask = get_region_palm_mask(region_id, None, hand_region_assignment, None)
            hand_verts_region = hand_verts[region_mask]
            dist_mat = torch.abs(self.pairwise_dist(hand_verts_region, obj_verts))
            hand_to_object_dist = torch.min(dist_mat, dim=1).values
            hand_to_object_dist = torch.sqrt(hand_to_object_dist)
            min_value = torch.min(hand_to_object_dist)
            res_list.append(min_value.item())

        res = torch.tensor(res_list)
        tip_only = torch.sum(res * tip_only_weight)
        # print(res)
        # print(tip_only)
        return tip_only.item()


    def region_h2o(self, h2o_batch, hand_region_assignment):
        h2o_batch = h2o_batch.double()  # 确保输入数据为双精度浮点数类型
        tip_regions = [2, 5, 8, 11, 14]
        tip_only_weight = torch.zeros(17, dtype=torch.float64, device=h2o_batch.device)
        tip_only_weight[tip_regions] = 1.0
        
        all_regions = list(range(17))
        batch_size = h2o_batch.size(0)
        res_batch = torch.zeros(batch_size, len(tip_regions), device=h2o_batch.device, dtype=torch.float64)
        
        for region_id in tip_regions:
            i = 0
            region_mask = get_region_palm_mask(region_id, None, hand_region_assignment, None)
            region_mask = torch.tensor(region_mask, device=h2o_batch.device, dtype=torch.bool)
            h2o_region = h2o_batch[:, region_mask]
            min_values, _ = torch.min(h2o_region, dim=1)
            # top5_values, _ = torch.topk(h2o_region, 5, dim=1, largest=False)
            res_batch[:, i] = min_values
            i = i+1
        
        # tip_only = torch.sum(res_batch * tip_only_weight, dim=1)
        
        return res_batch.view(batch_size,-1)


    # def ssim(self,hand1, hand2, C1=0.01**2, C2=0.03**2):
    #     mu1 = hand1.mean(dim=1, keepdim=True)
    #     mu2 = hand2.mean(dim=1, keepdim=True)
        
    #     sigma1 = hand1.var(dim=1, unbiased=False, keepdim=True)
    #     sigma2 = hand2.var(dim=1, unbiased=False, keepdim=True)
        
    #     sigma12 = ((hand1 - mu1) * (hand2 - mu2)).mean(dim=1, keepdim=True)
        
    #     ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    #     ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
        
    #     ssim_map = ssim_num / ssim_den
    #     ssim_value = ssim_map.mean()
    #     return ssim_value


    def farthest_point_sampling(self,xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, -1)
        return centroids




    def ssim(self,hand1, hand2, C1=0.01**2, C2=0.03**2):
        mu1 = hand1.mean(dim=1, keepdim=True)
        mu2 = hand2.mean(dim=1, keepdim=True)
        
        sigma1 = hand1.var(dim=1, unbiased=False, keepdim=True)
        sigma2 = hand2.var(dim=1, unbiased=False, keepdim=True)
        
        sigma12 = ((hand1 - mu1) * (hand2 - mu2)).mean(dim=1, keepdim=True)
        
        ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
        ssim_map = ssim_num / ssim_den

        # ssim_mean = (2 * mu1 * mu2 + C1)/(mu1**2 + mu2**2 + C1)
        # ssim_sigma = (sigma12 + C1)/(sigma1.sqrt()*sigma2.sqrt()+ C1)
        # ssim_map = ssim_sigma

        ssim_value = ssim_map
        return ssim_value.mean()


    def msssim(self,hand1, hand2, C1=0.01**2, C2=0.03**2):
        res = 1.- self.ssim(hand1, hand2)

        for i in range(3):
            B,N,C = hand1.shape
            centroids1 = self.farthest_point_sampling(hand1, N//2)
            centroids2 = self.farthest_point_sampling(hand2, N//2)
            hand1 = hand1[torch.arange(B).view(-1, 1), centroids1]
            hand2 = hand2[torch.arange(B).view(-1, 1), centroids2]
            ssim_value = 1. - self.ssim(hand1, hand2)
            res = res + ssim_value

            return res




    def h2o_msssim(self, hand1,hand2, hand_region_assignment):
        hand1 = hand1.double()  # 确保输入数据为双精度浮点数类型
        hand2 = hand2.double()  # 确保输入数据为双精度浮点数类型

        tip_regions = [2,5,8,11,14,15,16]
        
        all_regions = list(range(17))
        ssim_map = 0

        for region_id in all_regions:
            region_mask = get_region_palm_mask(region_id, None, hand_region_assignment, None)
            region_mask = torch.tensor(region_mask, device=hand1.device, dtype=torch.bool)
            hand1_region = hand1[:, region_mask]
            hand2_region = hand2[:, region_mask]
            region_ssim = self.ssim(hand1_region,hand2_region)
            ssim_map = ssim_map + region_ssim

            # print(hand1_region.shape)


        return ssim_map



    # def h2o_ssim(self, hand1,hand2,obj,C1=0.01**2, C2=0.03**2):
    #     a = hand1.double()  # 确保输入数据为双精度浮点数类型
    #     b = hand2.double()  # 确保输入数据为双精度浮点数类型
    #     c = obj.double()

    #     centroid_a = a.mean(dim=1, keepdim=True)  # (B, 1, C)
    #     centroid_b = b.mean(dim=1, keepdim=True)  # (B, 1, C)
    #     centroid_c = c.mean(dim=1, keepdim=True)  # (B, 1, C)


    #     vector_ac = centroid_c - centroid_a  # (B, 1, C)
    #     vector_bc = centroid_c - centroid_b  # (B, 1, C)


    #     # 扩展向量以便广播
    #     vector_ac_expanded = vector_ac.expand_as(a)  # (B, N, C)
    #     vector_bc_expanded = vector_bc.expand_as(b)  # (B, N, C)

    #     # 计算点到向量的投影
    #     projection_a = torch.sum(a * vector_ac_expanded, dim=2, keepdim=True) / torch.sum(vector_ac_expanded * vector_ac_expanded, dim=2, keepdim=True)
    #     projection_a = projection_a * vector_ac_expanded  # 投影向量 (B, N, C)

    #     projection_b = torch.sum(b * vector_bc_expanded, dim=2, keepdim=True) / torch.sum(vector_bc_expanded * vector_bc_expanded, dim=2, keepdim=True)
    #     projection_b = projection_b * vector_bc_expanded  # 投影向量 (B, N, C)

    #     # 计算投影向量的大小
    #     magnitude_a = torch.norm(projection_a, dim=2, keepdim=True)  # (B, N, 1)
    #     magnitude_b = torch.norm(projection_b, dim=2, keepdim=True)  # (B, N, 1)



    #     mu1 = magnitude_a.mean(dim=1, keepdim=True)
    #     mu2 = magnitude_b.mean(dim=1, keepdim=True)
        
    #     sigma1 = magnitude_a.var(dim=1, unbiased=False, keepdim=True)
    #     sigma2 = magnitude_b.var(dim=1, unbiased=False, keepdim=True)
        
    #     sigma12 = ((magnitude_a - mu1) * (magnitude_b - mu2)).mean(dim=1, keepdim=True)
        
    #     ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    #     ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
        
    #     ssim_map = ssim_num / ssim_den
    #     ssim_value = ssim_map


    #     return ssim_map.mean()


    # def h2o_msssim(self, h2o,h2ogt,C1=0.01**2, C2=0.03**2):


    #     mu1 = h2o.mean(dim=1, keepdim=True)
    #     mu2 = h2ogt.mean(dim=1, keepdim=True)
        
    #     sigma1 = h2o.var(dim=1, unbiased=False, keepdim=True)
    #     sigma2 = h2ogt.var(dim=1, unbiased=False, keepdim=True)
        
    #     sigma12 = ((h2o - mu1) * (h2ogt - mu2)).mean(dim=1, keepdim=True)
        
    #     ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    #     ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
        
    #     ssim_map = ssim_num / ssim_den
    #     ssim_value = ssim_map


    #     return ssim_map.mean()



    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        res = {}
        # print(inp.keys())
        # exit()
        # train RefineNet with _f: fake inp data
        verts_rhand_f = inp[Queries.HAND_VERTS_OBJ + "_f"]  # (B, 778, 3)
        trans_rhand_f = inp[Queries.HAND_TRANSL_OBJ + "_f"]  # (B, 3)
        global_hand_pose_obj_f = inp[Queries.HAND_POSE_OBJ + "_f"][:, :3]  # (B, 3)
        fpose_hand_pose_obj_f = inp[Queries.HAND_POSE_OBJ + "_f"][:, 3:]  # (B, 45)

        bs = verts_rhand_f.shape[0]
        device = verts_rhand_f.device
        global_orient_rhand_rotmat_f = aa_to_rotmat(global_hand_pose_obj_f).unsqueeze(1)  # (B, 1, 3, 3)
        fpose_rhand_rotmat_f = aa_to_rotmat(fpose_hand_pose_obj_f.reshape(-1, 3)).reshape(bs, -1, 3, 3)  # (B, 15, 3, 3)
        verts_object = inp[Queries.OBJ_VERTS_OBJ_DS]  # (B, 10000, 3)
        rhfaces = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand_f, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed, h2o, _ = point2point_signed(verts_rhand_f, verts_object, rh_mesh)
        h2o_dist = h2o.abs()
        res["h2o_dist"] = h2o_dist
        hand_shape = inp[Queries.HAND_SHAPE]
        verts_rhand = inp[Queries.HAND_VERTS_OBJ]  # (B, 778, 3)
        rh_mesh_gt = Meshes(verts=verts_rhand, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed_gt, h2o_gt, _ = point2point_signed(verts_rhand, verts_object, rh_mesh_gt)
        h2o_dist_gt = h2o_gt.abs()

        inp["h2o_dist_gt"] = h2o_dist_gt  # for compute loss



        # tip sign distant 
        # o2h_signed, h2o, _ = point2point_signed(verts_rhand, obj_verts, rh_mesh)
        tip_dis = self.region_h2o(h2o,self.merged_vertex_assignment).abs()
        tip_dis_gt = self.region_h2o(h2o_gt,self.merged_vertex_assignment)
        inp["tip_dis_gt"] = tip_dis_gt  # for compute loss



        # exit()
        hand_param,hand_param1,hand_param2 = self._forward_impl(
            h2o_dist=h2o_dist,
            fpose_rhand_rotmat_f=fpose_rhand_rotmat_f,
            trans_rhand_f=trans_rhand_f,
            global_orient_rhand_rotmat_f=global_orient_rhand_rotmat_f,
            verts_object=verts_object,
            hand_shape=hand_shape,
            obj_bps = inp['obj_bps'],
        )

        res.update(hand_param)
        mano_pose = torch.cat([hand_param["global_orient"], hand_param["hand_pose"]], dim=1)  # (B, 48)
        mano_output: MANOOutput = self.mano_layer(mano_pose, hand_shape)
        verts_rhand = mano_output.verts + hand_param["transl"].unsqueeze(1)
        res["hand_verts"] = verts_rhand  # for visualize

        mano_pose1 = torch.cat([hand_param1["global_orient"], hand_param1["hand_pose"]], dim=1)  # (B, 48)
        mano_output1: MANOOutput = self.mano_layer(mano_pose1, hand_shape)
        verts_rhand1 = mano_output1.verts + hand_param1["transl"].unsqueeze(1)
        res["hand_verts1"] = verts_rhand1  # for visualize

        mano_pose2 = torch.cat([hand_param2["global_orient"], hand_param2["hand_pose"]], dim=1)  # (B, 48)
        mano_output2: MANOOutput = self.mano_layer(mano_pose2, hand_shape)
        verts_rhand2 = mano_output2.verts + hand_param2["transl"].unsqueeze(1)
        res["hand_verts2"] = verts_rhand2 # for visualize


        rnet_loss = self.compute_loss(res, inp, **kwargs)
        self.loss_metric.feed(rnet_loss)
        if step_idx % self.log_freq == 0:
            for k, v in rnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, rnet_loss

    def on_train_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"  # GrabNet-train
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def validation_step(self, inp, step_idx, mode="val", **kwargs):
        return self.training_step(inp, step_idx, mode=mode, **kwargs)

    def on_val_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"  # GrabNet-train
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def testing_step(self, inp, cnet_res, step_idx, **kwargs):
        res = {}

        # for k, v in cnet_res.items():
        verts_rhand_f = cnet_res[f"Coarse.hand_verts"]  # (B, 778, 3)
        trans_rhand_f = cnet_res[f"Coarse.transl"]  # (B, 3)
        global_hand_pose_obj_f = cnet_res[f"Coarse.global_orient"]  # (B, 3)
        fpose_hand_pose_obj_f = cnet_res[f"Coarse.hand_pose"]  # (B, 45)

        bs = verts_rhand_f.shape[0]
        device = verts_rhand_f.device
        global_orient_rhand_rotmat_f = aa_to_rotmat(global_hand_pose_obj_f).unsqueeze(1)  # (B, 1, 3, 3)
        fpose_rhand_rotmat_f = aa_to_rotmat(fpose_hand_pose_obj_f.reshape(-1, 3)).reshape(bs, -1, 3, 3)  # (B, 15, 3, 3)
        verts_object = inp[Queries.OBJ_VERTS_OBJ_DS]  # (B, 10000, 3)
        rhfaces = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand_f, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed, h2o, _ = point2point_signed(verts_rhand_f, verts_object, rh_mesh)
        h2o_dist = h2o.abs()
        res["h2o_dist"] = h2o_dist
        hand_shape = torch.zeros(bs, 10).to(device)


        tip_dis = self.region_h2o(h2o,self.merged_vertex_assignment).abs()

        hand_param,_,_ = self._forward_impl(
            h2o_dist=h2o_dist,
            fpose_rhand_rotmat_f=fpose_rhand_rotmat_f,
            trans_rhand_f=trans_rhand_f,
            global_orient_rhand_rotmat_f=global_orient_rhand_rotmat_f,
            verts_object=verts_object,
            hand_shape=hand_shape,
            obj_bps = inp['obj_bps'],
        )

        res.update(hand_param)
        mano_pose = torch.cat([hand_param["global_orient"], hand_param["hand_pose"]], dim=1)  # (B, 48)
        mano_output: MANOOutput = self.mano_layer(mano_pose, hand_shape)
        verts_rhand = mano_output.verts + hand_param["transl"].unsqueeze(1)
        res["hand_verts"] = verts_rhand  # for visualize

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, {}

    def _forward_impl(self, h2o_dist, fpose_rhand_rotmat_f, trans_rhand_f, global_orient_rhand_rotmat_f, verts_object,obj_bps,**kwargs):

        bs = h2o_dist.shape[0]
        init_pose = fpose_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_rpose = global_orient_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_pose = torch.cat([init_rpose, init_pose], dim=1)
        init_trans = trans_rhand_f
        obj_feature =obj_bps

        hand_parms_ori = parms_decode(init_pose, init_trans)

        for i in range(self.n_iters):
            if i != 0:
                hand_parms = parms_decode(init_pose, init_trans)
                mano_pose = torch.cat([hand_parms["global_orient"], hand_parms["hand_pose"]], dim=1)  # (B, 48)
                mano_shape = kwargs[Queries.HAND_SHAPE]  # (B, 10)
                mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)

                verts_rhand = mano_output.verts + hand_parms["transl"].unsqueeze(1)
                _, h2o_dist, _ = point2point_signed(verts_rhand, verts_object)

                # h2o_dist = self.region_h2o(h2o_dist,self.merged_vertex_assignment).abs()

            # print(h2o_dist)
            # print(h2o_dist.shape)
            # exit()
            h2o_dist = h2o_dist.to(torch.float32)
            h2o_dist = self.bn1(h2o_dist)
            X0 = torch.cat([h2o_dist, init_pose, init_trans], dim=1)
            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)

            pose = self.out_p(X)
            trans = self.out_t(X)

            init_trans = init_trans + trans
            init_pose = init_pose + pose
            if i == 0:
                hand_parms1 = parms_decode(init_pose, init_trans)
            elif i == 1:
                hand_parms2 = parms_decode(init_pose, init_trans)



        hand_parms = parms_decode(init_pose, init_trans)
        return hand_parms,hand_parms1,hand_parms2

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def compute_loss(self, prd, tgt, **kwargs):
        device = tgt[Queries.HAND_VERTS_OBJ].device
        dtype = tgt[Queries.HAND_VERTS_OBJ].dtype
        bs = tgt[Queries.HAND_VERTS_OBJ].shape[0]

        if self.v_weights.device != device:
            self.v_weights = self.v_weights.to(device)
            self.v_weights2 = self.v_weights2.to(device)
            self.vpe = self.vpe.to(device)

        verts_rhand = prd["hand_verts"]
        rh_f = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)
        obj_verts = tgt[Queries.OBJ_VERTS_OBJ_DS]
        o2h_signed, h2o, _ = point2point_signed(verts_rhand, obj_verts, rh_mesh)

        # verts_rhand_ori = prd["hand_verts2"]
        # dis = torch.norm(verts_rhand_ori-tgt[Queries.HAND_VERTS_OBJ],dim=-1)
        # refine_w = 1/(dis**2+1e-6)
        # refine_w = refine_w/torch.sum(refine_w,dim=1,keepdim=True)


        # w_h2o = tgt['h2o_dist_gt'].abs()
        # dis = torch.norm(verts_rhand_ori-tgt[Queries.HAND_VERTS_OBJ],dim=-1)
        # refine_w = 1/(w_h2o+1e-6)
        # refine_w = w_h2o/torch.sum(w_h2o,dim=1,keepdim=True)


        verts_rhand1= prd["hand_verts1"]
        rh_f = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)
        obj_verts = tgt[Queries.OBJ_VERTS_OBJ_DS]
        o2h_signed1, h2o1, _ = point2point_signed(verts_rhand1, obj_verts, rh_mesh)


        verts_rhand2= prd["hand_verts2"]
        rh_f = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)
        obj_verts = tgt[Queries.OBJ_VERTS_OBJ_DS]
        o2h_signed2, h2o2, _ = point2point_signed(verts_rhand2, obj_verts, rh_mesh)

        # tip_dis = torch.einsum('ij,j->ij', h2o.abs(), self.v_weights2)
        # tip_dis = self.region_h2o(h2o,self.merged_vertex_assignment)

        # ### ssim loss
        # verts_rhand_gt = tgt[Queries.HAND_VERTS_OBJ]  # (B, 778, 3)
        # rh_mesh_gt = Meshes(verts=verts_rhand_gt, faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)
        # print(rh_mesh.shape)
        # print(rh_mesh_gt.shape)
        # exit()
        # print(self.ssim(rh_mesh,rh_mesh_gt))
        # exit()
        # ssim_loss = (1.-self.region_ssim(rh_mesh,rh_mesh_gt,self.merged_vertex_assignment))
        # ssim_loss = -0.1*self.region_ssim(rh_mesh,rh_mesh_gt,self.merged_vertex_assignment)-1*self.ssim(rh_mesh,rh_mesh_gt)
        # print(h2o)
        # print(tgt['h2o_dist_gt'].abs())
        # exit()
        ssim_loss = -0.3*self.h2o_msssim(h2o.abs(),tgt['h2o_dist_gt'].abs(),self.merged_vertex_assignment)
        # ssim_loss = -0.1*self.ssim(verts_rhand,tgt[Queries.HAND_VERTS_OBJ])


        # h2o_dist_gt = torch.einsum('ij,j->ij', tgt['h2o_dist_gt'].abs(), self.v_weights2)
        # h2o_dist_gt = self.region_h2o(h2o_dist_gt,self.merged_vertex_assignment)

        # tip_dis = 5*tip_dis.abs().mean()
        # loss_dist_h = 35 * (1. - self.kl_coef) * torch.mean(torch.abs(h2o.abs() - h2o_dist_gt.abs()))

        ######### dist loss
        loss_dist_h = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ij,j->ij', torch.abs(h2o.abs() - tgt['h2o_dist_gt'].abs()), self.v_weights2))
        ########## verts loss
        loss_mesh_rec_w = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ijk,j->ijk', torch.abs(verts_rhand - tgt[Queries.HAND_VERTS_OBJ]), self.v_weights2))
        ########## edge loss
        loss_edge = 30 * (1. - self.kl_coef) * self.LossL1(self.edges_for(verts_rhand, self.vpe),
                                                           self.edges_for(tgt[Queries.HAND_VERTS_OBJ], self.vpe))
        ##########
        loss_dist_h1 = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ij,j->ij', torch.abs(h2o1.abs() - tgt['h2o_dist_gt'].abs()), self.v_weights2))
        loss_dist_h2 = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ij,j->ij', torch.abs(h2o2.abs() - tgt['h2o_dist_gt'].abs()), self.v_weights2))

        # loss_mesh_rec_w1 = 35 * (1. - self.kl_coef) * torch.mean(
        #     torch.einsum('ijk,j->ijk', torch.abs(verts_rhand1 - tgt[Queries.HAND_VERTS_OBJ]), self.v_weights2))
        # ########## edge loss
        # loss_edge1 = 30 * (1. - self.kl_coef) * self.LossL1(self.edges_for(verts_rhand1, self.vpe),
        #                                                    self.edges_for(tgt[Queries.HAND_VERTS_OBJ], self.vpe))


        loss_dict = {
            'loss_edge_r': loss_edge,
            'loss_mesh_rec_r': loss_mesh_rec_w,
            'loss_dist_h_r': loss_dist_h,
            'loss_dist_h_r1': loss_dist_h1,
            'loss_dist_h_r2': loss_dist_h2,
            # 'loss_edge_r1': loss_edge,
            # 'loss_mesh_rec_r1': loss_mesh_rec_w,
            # 'loss_tip':tip_dis,
            'loss_ssim': ssim_loss,
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total_r'] = loss_total
        loss_dict['loss'] = loss_total
        return loss_dict
