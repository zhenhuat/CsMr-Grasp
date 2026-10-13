from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from oikit.oi_shape.utils import ALL_INTENT

from lib.datasets.grasp_query import Queries
from lib.utils.builder import MODEL
from lib.utils.transform import aa_to_rotmat

from .coarse_net import CoarseNet
from .grabnet_arch import ResBlock, parms_decode
from lib.utils.transform import *


import math
import random
from typing import List
from collections import namedtuple
from einops import rearrange
# from timm.models.layers import DropPath
from timm.layers import DropPath

from pytorch3d.structures import Meshes
from lib.utils.pcd import point2point_signed

from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.anchorutils import anchor_load_driver
from manotorch.utils.anchorutils import get_region_palm_mask
import torch.nn.init as init

# from ..pointnet2 import pointnet2_sem_seg_msg
 


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    # print(t)
    tt = t.to(torch.int64)
    out = a.gather(-1, tt)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:,:x.shape[1]].repeat(x.shape[0], 1, 1)
        return self.dropout(x)



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise','pred_trans_noise', 'pred_x_start'])




@MODEL.register_module()
class CoarseIntentNetIntentEmbedEncode(CoarseNet):

    def __init__(self, cfg):
        self.n_intents = cfg.DATA_PRESET.N_INTENTS
        super(CoarseIntentNetIntentEmbedEncode, self).__init__(cfg)

        ###### my diff
        # build diffusion
        timesteps = 100
        #timesteps_eval = args.timestep_eval
        sampling_timesteps = 1
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        #self.num_timesteps_eval = int(timesteps_eval)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = 1.0
        self.box_renewal = True
        self.use_ensemble = True
        self.num_proposals =  3
        self.n_reference = 1

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

    def _build_layers(self):
        in_bps = self.in_bps
        in_pose = self.in_pose
        # n_neurons = self.n_neurons
        n_neurons = 256
        latentD = self.latentD + 80
        intentD = self.latentD


        self.intent_embed = nn.Embedding(self.n_intents, intentD)  # (3, 16)
        nn.init.uniform_(self.intent_embed.weight.data, 0, 1)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(n_neurons),
            nn.Linear(n_neurons, n_neurons),
            nn.GELU(),
            nn.Linear(n_neurons, n_neurons),
        )



        self.memory_hand_rot =  torch.randn(5,n_neurons).cuda()
        # self.memory_hand_rot = nn.Parameter(torch.ones(5,n_neurons))
        # init.orthogonal_(self.memory_hand_rot)


        self.SMA = SAMFormer(6,n_neurons,n_neurons,2,0.1,n_neurons)

        self.v_obj_emb = nn.Linear(3,n_neurons)
        self.intent_emb = nn.Linear(16,n_neurons)
        self.transl_emb = nn.Linear(3,n_neurons)
        self.pose_emb = nn.Linear(96,n_neurons)

        self.m_pose_emb = nn.Linear(n_neurons,n_neurons)
        self.cm_pose_emb = nn.Linear(96,n_neurons)

        self.g_obj_emb = nn.Linear(4096,n_neurons)

        self.pre_x = nn.Linear(n_neurons*2,n_neurons)

        self.gelu = nn.GELU()



        self.dec_pose = nn.Linear(n_neurons, 96)
        self.dec_trans = nn.Linear(n_neurons, 3)

        # self.pointnet = pointnet2.get_model()



        self.merged_vertex_assignment = anchor_load_driver("./assets")[2]  # vertex that belongs to a certain palm region

        # self.w = nn.Parameter(2)



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


    def region_h2o(self, h2o,hand_region_assignment):
        h2o = h2o.squeeze(0).float()
        tip_regions = [2, 5, 8, 11, 14]
        tip_only_weight = torch.zeros(17, dtype=torch.float64)
        tip_only_weight[tip_regions] = 1.0
        all_regions = list(range(17))
        res_list = []
        for region_id in all_regions:
            region_mask = get_region_palm_mask(region_id, None, hand_region_assignment, None)
            h2o_region = h2o[region_mask]
            # print(h2o_region)
            min_value = torch.min(h2o_region)
            # print(min_value)
            # exit()
            res_list.append(min_value.item())
        res = torch.tensor(res_list)
        tip_only = torch.sum(res * tip_only_weight)
        # print(res)
        # print(tip_only)
        return tip_only.item()


    def decode(self,obj_verts, Zin,Z_trans, bps_object,cond_hand, intent,t,trainning=False):
        bs = Zin.shape[0]

        Zin = rearrange(Zin,'b n c->b (n c)')
        # print(self.memory_hand_rot[:,:5])
        # exit()
        memory_hand_rot = self.memory_hand_rot.unsqueeze(0).repeat(bs,1,1)
        t = t.unsqueeze(1)
        # o_bps = self.dec_bn1(bps_object)

        Zin = self.pose_emb(Zin).unsqueeze(1)
        Z_trans = self.transl_emb(Z_trans).unsqueeze(1)
        X = torch.cat([Zin,Z_trans],-1)  #b,17,c
        # X = Zin + Z_trans
        X = self.gelu(X)
        X = self.pre_x(X)
        X = self.gelu(X)

        cond_bps = self.g_obj_emb(bps_object).unsqueeze(1)
        cond_intent = self.intent_emb(intent).unsqueeze(1)
        cond_time = self.time_mlp(t)
        # print(cond_bps.shape,cond_intent.shape,cond_time.shape)
        cond = torch.cat([cond_bps,cond_intent,cond_time],1)
        cond = self.gelu(cond)



        m_pose = self.m_pose_emb(memory_hand_rot)
        cm_pose = self.cm_pose_emb(cond_hand)
        m_pose = self.gelu(m_pose)
        cm_pose = self.gelu(cm_pose)

        # cond_hand = torch.cat((m_pose,cm_pose),1)
        # print(m_pose.shape)
        # exit()

        X_final = self.SMA(X,cond,m_pose,cm_pose).squeeze(1)
        # X_final = X.squeeze(1)

        # print(X_final.shape)
        # exit()
        # print(self.dec_pose.bias)
        pose = self.dec_pose(X_final)
        trans = self.dec_trans(X_final)


        # if trainning:
        #     pre_pose = pose.detatch()
        #     difference = torch.norm(self.memory_hand_rot.unsqueeze(0) - pre_pose.unsqueeze(1),dim=-1)
        #     # print(difference.shape)
        #     closest_indices = torch.argmin(difference.view(pre_pose.shape[0], -1), dim=1)
        #     # print(closest_indices.shape)
        #     for i in range(pose.shape[0]):
        #         index = closest_indices[i]
        #         average_values = (self.memory_hand_rot[index] + pre_pose[i]) / 2.0
        #         self.memory_hand_rot[index] = average_values

        results = parms_decode(pose, trans)
        # results['mom_k'] = self.memory_hand_rot
        # results["z"] = Zin
        results["ztrans"] = Z_trans
        return results


    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self,obj_verts, x,x_trans, bps_object,mano_shape,intent,cond_hand, t):
        bs = obj_verts.shape[0]
        # print('here')
        x_t = torch.clamp(x, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t = x_t / self.scale

        trans_t = torch.clamp(x_trans, min=-1.1 * self.scale, max=1.1*self.scale)
        trans_t = trans_t / self.scale


        # mano_pose = rot6d_to_aa(x_t).reshape(bs,-1) #B,48
        # mano_shape = mano_shape  # (B, 10)
        # mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)

        # verts_rhand = mano_output.verts + trans_t.unsqueeze(1)
        # rh_f = self.mano_layer.th_faces.expand(bs, -1, -1)
        # rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to('cuda').verts_normals_packed().view(-1, 778, 3)
        # o2h_signed, h2o, _ = point2point_signed(verts_rhand, obj_verts, rh_mesh)
        # h2o_dist = h2o.abs()


        pre_results = self.decode(obj_verts,x_t,trans_t,bps_object,cond_hand, intent, t)
        pred_pose = pre_results["pre_p16"].reshape(-1,16,6)
        pred_trans = pre_results["transl"].reshape(-1,3)

        x_start = pred_pose
        x_start = x_start * self.scale
        x_start = torch.clamp(x_start, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)


        trans_start = pred_trans
        trans_start = trans_start * self.scale
        trans_start = torch.clamp(trans_start, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_trans_noise = self.predict_noise_from_start(x_trans, t, trans_start)

        return ModelPrediction(pred_noise,pred_trans_noise, pre_results)



    @torch.no_grad()
    def ddim_sample(self,obj_verts, bps_object,mano_shape,cond_hand, intent, seed=None):
        batch = bps_object.shape[0]
        # shape = (batch, 96)
        # print(self.num_proposals)
        # exit()
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        # print(total_timesteps,sampling_timesteps)
        sampling_timesteps = 100
        index = 10
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # print(times)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


        device= intent.device
        dtype = bps_object.dtype
        img = np.random.normal(0.0, 1.0, size=(batch,16,6))
        img = torch.tensor(img, dtype=dtype).to(device)

        img_trans = np.random.normal(0.0, 1.0, size=(batch,3))
        img_trans = torch.tensor(img_trans, dtype=dtype).to(device)
        # print(img_trans)
        # exit()

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        preds_all=[]



        tip_dis_list = []
        previous_img_noise = []
        previous_trans_noise = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=dtype)
            preds = self.model_predictions(obj_verts,img,img_trans,bps_object,mano_shape,intent,cond_hand,time_cond)
            pred_noise,pred_trans_noise, x_start,trans_start = preds.pred_noise,preds.pred_trans_noise,preds.pred_x_start["pre_p16"].reshape(batch,16,6),preds.pred_x_start["transl"].reshape(batch,3)
            # if time == sampling_timesteps-1:
            #     print(x_start)

            previous_trans_noise.append(pred_trans_noise)
            previous_img_noise.append(pred_noise)
            if time + 1 ==  total_timesteps:
                previous_trans_noise.append(pred_trans_noise)
                previous_img_noise.append(pred_noise)

            mano_pose = rot6d_to_aa(x_start).reshape(batch,-1) #B,48
            mano_shape = mano_shape  # (B, 10)
            mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)
            verts_rhand = mano_output.verts + img_trans.unsqueeze(1)


            # ## tip distant
            tip_dis = self.region_disjointness_metric(verts_rhand,obj_verts,self.merged_vertex_assignment)


            ## tip sign distant 
            # rh_f = self.mano_layer.th_faces.expand(batch, -1, -1)
            # rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to('cuda').verts_normals_packed().view(-1, 778, 3)
            # o2h_signed, h2o, _ = point2point_signed(verts_rhand, obj_verts, rh_mesh)
            # # h2o = h2o.abs()
            # tip_dis = self.region_h2o(h2o,self.merged_vertex_assignment)
            # print(tip_dis)
            # exit()
  
            tip_dis_list.append(tip_dis)
            # print(total_timesteps,time+1)



            # pred_noise = 2*pred_noise - previous_img_noise[total_timesteps-time-1]
            # pred_trans_noise = 2*pred_trans_noise - previous_trans_noise[total_timesteps-time-1]


            preds_all.append(preds.pred_x_start)


            if time_next < 0:
                img = x_start
                img_trans = trans_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]


            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            noise_trans = torch.randn_like(img_trans)
            # print(noise_trans)
            

            # noise = (img_trans - alpha.sqrt()*trans_start)/((1-alpha).sqrt())
            # noise_trans = (img_trans - alpha.sqrt()*trans_start)/((1-alpha).sqrt())
            # print(noise.shape)
            # exit()


            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            # print(img.dtype)
            img = img.to(dtype)


            img_trans = trans_start * alpha_next.sqrt() + \
                  c * pred_trans_noise + \
                  sigma * noise_trans
            # print(img.dtype)
            img_trans = img_trans.to(dtype)

            if sampling_timesteps - time == index+1:
                break

        sum_dict = {}
        count_dict = {}

        final_dict = {}
        tip_dist_sum = sum(tip_dis_list)

        time_weights = [t+1 for t in range(len(preds_all))]
        normalized_weights = [w / sum(time_weights) for w in time_weights]

        # for i in range(len(preds_all)):
        #     for key,value in preds_all[i].items():
        #         if key in final_dict:
        #             final_dict[key] = final_dict[key] + (tip_dis_list[i]/tip_dist_sum)*value
        #         else:
        #             final_dict[key] = (tip_dis_list[i]/tip_dist_sum)*value


        # for dictionary in preds_all:
        #     for key, value in dictionary.items():
        #         if key in sum_dict:
        #             sum_dict[key] += value
        #             count_dict[key] += 1
        #         else:
        #             sum_dict[key] = value
        #             count_dict[key] = 1
        # average_dict = {}
        # for key in sum_dict:
        #     average_dict[key] = sum_dict[key] / count_dict[key]

        # preds_all[99]
        return preds_all[index]

    @torch.no_grad()
    def average_ddim_sample(self,n_sample,obj_verts, bps_object,mano_shape,cond_hand, intent, seed=None):
        average_preds = None
        batch = bps_object.shape[0]
        iter_dist = []
        iter_pre = []
        for i in range(n_sample):
            preds = self.ddim_sample(obj_verts, bps_object,mano_shape,cond_hand, intent, seed=None)
            iter_pre.append(preds)

            pred_pose  = preds["pre_p16"].reshape(batch,16,-1)
            pred_trans = preds["transl"]
            mano_pose = rot6d_to_aa(pred_pose).reshape(batch,-1) #B,48
            mano_shape = mano_shape  # (B, 10)
            mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)
            verts_rhand = mano_output.verts + pred_trans.unsqueeze(1)
            # ## tip distant
            tip_dis = self.region_disjointness_metric(verts_rhand,obj_verts,self.merged_vertex_assignment)
            iter_dist.append(tip_dis)
        # print(iter_dist)
        # exit()

        if average_preds is None:
            average_preds = {key: torch.zeros_like(value) for key, value in preds.items()}

        reciprocals = [dis for dis in iter_dist]
        total = sum(reciprocals)
        wight = [reciprocal/total for reciprocal in reciprocals]

        for i in range(n_sample):
            for key, value in preds.items():
                average_preds[key] = average_preds[key] + wight[i]*iter_pre[i][key]
        # exit()
        return average_preds




    def _forward_impl(self,obj_verts,bps_object, trans_rhand,h2o_dist, intent,x_poses,t,trainning=True,**kwargs):
        # print(bps_object.shape, trans_rhand.shape, global_orient_rhand_rotmat.shape, intent.shape)

        # z = self.encode(bps_object, trans_rhand, global_orient_rhand_rotmat, intent)
        # z_s = z.rsample()
        # print(z_s.shape, bps_object.shape, intent.shape,x_poses.shape)
        # exit()
        # z_s = x_poses.reshape(x_poses.shape[0],-1)
        hand_parms = self.decode(obj_verts,x_poses,trans_rhand, bps_object,h2o_dist, intent,t,trainning)
        # results = {"mean": z.mean, "std": z.scale}
        results = {}
        results.update(hand_parms)
        return results

    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        # print(111,inp['memory'].shape)
        # exit()
        bps_object = inp[Queries.OBJ_BPS]  # (B, 4096)
        bs = bps_object.shape[0]

        # print(inp[Queries.HAND_POSE_OBJ].shape)

        n_data = inp['memory_bps'].shape[0]
        mem_bps = inp['memory_bps'].unsqueeze(0).repeat(bs,1,1)
        memory_hand = inp['memory_hand'].unsqueeze(0).repeat(bs,1,1)
        mem_intent_id = inp['memory_intent']
        x_intent = inp['intent_id']
        x_intent = x_intent.unsqueeze(1).unsqueeze(1).expand_as(mem_bps)
        mem_intent_id = mem_intent_id.unsqueeze(0).unsqueeze(2).expand_as(mem_bps)
        # print(mem_bps.shape,memory_hand.shape)
        # cos_sim = F.cosine_similarity(bps_object.unsqueeze(1).expand_as(mem_bps), mem_bps, dim=2).unsqueeze(1)
        cos_sim = F.cosine_similarity(torch.cat((bps_object.unsqueeze(1).expand_as(mem_bps),x_intent),dim=-1),torch.cat((mem_bps,mem_intent_id),dim=-1), dim=2).unsqueeze(1)
        # sim_weight = F.softmax(cos_sim,-1)
        # cond_hand_global = sim_weight@memory_hand
        cos_sim_squeezed = cos_sim.squeeze(1)  # 去掉中间的维度，变为[32, 384]
        values, indices = torch.topk(cos_sim_squeezed, self.n_reference, dim=1)  # 返回最大的5个值和它们的索引
        cond_hand = torch.gather(memory_hand, 1, indices.unsqueeze(2).expand(-1, -1, 48))
        # cond_hand = torch.cat((cond_hand_global,cond_hand),1)
        # print(cond_hand.shape)
        # exit()
        ccond_hand = []
        for i in range(cond_hand.shape[1]):  # 遍历b维度
            c_hand = aa_to_rot6d(cond_hand[:,i].reshape(-1,16,3)).reshape(bs,-1)  
            ccond_hand.append(c_hand)
        cond_hand = torch.stack(ccond_hand,1)


        obj_verts = inp[Queries.OBJ_VERTS_OBJ_DS] #B,4096,3
        trans_rhand = inp[Queries.HAND_TRANSL_OBJ]  # (B, 3)
        global_hand_pose_obj = inp[Queries.HAND_POSE_OBJ][:, :3]  # (B, 3)
        global_orient_rhand_rotmat = aa_to_rotmat(global_hand_pose_obj).unsqueeze(1)  # (B, 1, 3, 3)
        intent_vec = inp[Queries.INTENT_VEC]  # (B, INTENT_D)
        bs = bps_object.shape[0]


        gt_pose = inp[Queries.HAND_POSE_OBJ]  # (B, 16*3)
        gt_pose = gt_pose.reshape(-1,16,3)
        gt_pose = aa_to_rot6d(gt_pose)      # (B, 16,6)
        # gt_pose = rot6d_to_rotmat(gt_pose)     # (B, 16,3,3)   
        # print(gt_pose.shape)   
        # gt_pose = rearrange(gt_pose,'b n f->b (n f)') # b,16,6


        x_poses,x_trans,noise_pose,noise_trans,t = self.prepare_targets(gt_pose,trans_rhand)
        x_poses = x_poses.float()
        x_trans = x_trans.float()


        # mano_pose = rot6d_to_aa(x_poses).reshape(bs,-1) #B,48
        # mano_shape = inp[Queries.HAND_SHAPE]  # (B, 10)
        # mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)

        # verts_rhand = mano_output.verts + x_trans.unsqueeze(1)
        # rh_f = self.mano_layer.th_faces.expand(bs, -1, -1)
        # rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to('cuda').verts_normals_packed().view(-1, 778, 3)
        # o2h_signed, h2o, _ = point2point_signed(verts_rhand, obj_verts, rh_mesh)
        # h2o_dist = h2o.abs()



        t = t.squeeze(-1)

        # intent_vec has shape: (B, 3), the last dim is a one hot vector that indicates the intent ID
        # self.intent_embed.weight has shape: (3, 16), each row is a vector that represents an intent embedding
        # now use the intent_vec to index self.intent_embed
        # intent has shape: (B, 16)
        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)


        res = self._forward_impl(obj_verts,bps_object, x_trans,cond_hand,intent,x_poses,t)
        cnet_loss = self.compute_loss(prd=res, tgt=inp, **kwargs)
        if step_idx % self.log_freq == 0:
            for k, v in cnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, cnet_loss

    def testing_step(self,inp, step_idx, intent_name=None, **kwargs):
        obj_bps = inp[Queries.OBJ_BPS]
        bs = obj_bps.shape[0]


        n_data = inp['memory_bps'].shape[0]
        mem_bps = inp['memory_bps'].unsqueeze(0).repeat(bs,1,1)
        memory_hand = inp['memory_hand'].unsqueeze(0).repeat(bs,1,1)
        mem_intent_id = inp['memory_intent']

        x_intent = torch.tensor([int(ALL_INTENT[intent_name])]).to(obj_bps.device)
        x_intent = x_intent.unsqueeze(1).unsqueeze(1).expand_as(mem_bps)
        mem_intent_id = mem_intent_id.unsqueeze(0).unsqueeze(2).expand_as(mem_bps)
        # print(mem_bps.shape,memory_hand.shape)
        # cos_sim = F.cosine_similarity(obj_bps.unsqueeze(1).expand_as(mem_bps), mem_bps, dim=2).unsqueeze(1)
        cos_sim = F.cosine_similarity(torch.cat((obj_bps.unsqueeze(1).expand_as(mem_bps),x_intent),dim=-1),torch.cat((mem_bps,mem_intent_id),dim=-1), dim=2).unsqueeze(1)
        sim_weight =F.softmax(cos_sim,-1)
        cond_hand_global = sim_weight@memory_hand
        cos_sim_squeezed = cos_sim.squeeze(1)  # 去掉中间的维度，变为[32, 384]
        values, indices = torch.topk(cos_sim_squeezed, self.n_reference, dim=1)  # 返回最大的5个值和它们的索引
        cond_hand = torch.gather(memory_hand, 1, indices.unsqueeze(2).expand(-1, -1, 48))
        # cond_hand = torch.cat((cond_hand_global,cond_hand),1)
        # print(cond_hand.shape)
        # exit()
        ccond_hand = []
        for i in range(cond_hand.shape[1]):  # 遍历b维度
            c_hand = aa_to_rot6d(cond_hand[:,i].reshape(-1,16,3)).reshape(bs,-1)  
            ccond_hand.append(c_hand)
        cond_hand = torch.stack(ccond_hand,1)


        # intent_vec = inp[Queries.INTENT_VEC]  # (B, INTENT_D)
        if intent_name is not None:
            intent_id = int(ALL_INTENT[intent_name])
            intent_vec = torch.zeros((bs, self.n_intents), dtype=torch.float32)
            intent_vec[torch.arange(bs), intent_id - 1] = 1.0  # intent id starts from 1
            intent_vec = intent_vec.to(obj_bps.device)
            inp[Queries.INTENT_VEC] = intent_vec  # @OVERWRITE
        else:
            intent_vec = inp[Queries.INTENT_VEC]

        obj_verts = inp[Queries.OBJ_VERTS_OBJ_DS] #B,4096,3
        mano_shape = inp[Queries.HAND_SHAPE]  # (B, 10)
        # print(inp[Queries.OBJ_VERTS_OBJ].shape)
        # print(inp[Queries.OBJ_NORMALS_OBJ_DS].shape)
        # exit()

        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)
        res = self.ddim_sample(obj_verts,obj_bps,mano_shape,cond_hand, intent=intent, seed=kwargs.get("seed", None))
        # res = self.average_ddim_sample(20,obj_verts,obj_bps,mano_shape, intent=intent, seed=kwargs.get("seed", None))

        res["mano_pose"] = torch.cat([res["global_orient"], res["hand_pose"]], dim=1)
        mano_out = self.mano_layer(res["mano_pose"])
        res["hand_verts"] = mano_out.verts + res[f"transl"].unsqueeze(1)  # (B, 778, 3)
        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, {}


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def prepare_diffusion_concat(self, pose_3d,pose_trans):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise_pose = torch.randn(16,6, device='cuda')
        noise_trans = torch.randn(3, device='cuda')
        # print(t)
        # exit()

        x_start_pose = pose_3d
        x_start_pose = x_start_pose * self.scale
        x_pose = self.q_sample(x_start=x_start_pose, t=t, noise=noise_pose)
        x_pose = torch.clamp(x_pose, min= -1.1 * self.scale, max= 1.1*self.scale)
        x_pose = x_pose / self.scale


        x_start_trans = pose_trans
        x_start_trans = x_start_trans * self.scale
        x_trans = self.q_sample(x_start=x_start_trans, t=t, noise=noise_trans)
        x_trans = torch.clamp(x_trans, min= -1.1 * self.scale, max= 1.1*self.scale)
        x_trans = x_trans / self.scale

        return x_pose,x_trans, noise_pose,noise_trans, t

    def prepare_targets(self, target_pose,target_trans):
        diffused_poses = []
        diffused_trans = []
        noise_pose = []
        noise_trans = []
        ts = []
        for i in range(0,target_pose.shape[0]):
            targets_pose_per_sample = target_pose[i]
            targets_trans_per_sample = target_trans[i]
            # print(111,targets_per_sample.shape)

            d_poses,d_trans, d_poses_noise,d_trans_noise, d_t = self.prepare_diffusion_concat(targets_pose_per_sample,targets_trans_per_sample)
            # print(2222,d_noise.shape)
            diffused_poses.append(d_poses)
            diffused_trans.append(d_trans)
            noise_pose.append(d_poses_noise)
            noise_trans.append(d_trans_noise)
            ts.append(d_t)

        return torch.stack(diffused_poses),torch.stack(diffused_trans),torch.stack(noise_pose),torch.stack(noise_trans),torch.stack(ts)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
# from timm.vision_transformer import PatchEmbed, Attention, Mlp

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.5)
        self.mlp = My_Mlp(hidden_size, hidden_size * 4, hidden_size,nn.GELU,0.5)

        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


        self.adp_c = nn.Linear(hidden_size*8,hidden_size)
        self.CA = CrossAttention(hidden_size,heads=num_heads)
        self.SE = SENet(hidden_size)
        self.act = nn.Sigmoid()


    def forward(self, x, cond,memory):
        condition = torch.cat((cond,memory),1)
        # c = self.attn(condition)
        # c = c + self.mlp(c)
        c = condition.mean(dim=1)
        shift_msa, scale_msa, gate_msa,shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        gate_msa = self.act(gate_msa)
        gate_mlp = self.act(gate_mlp)
        # shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        # shift_mca, scale_mca = self.adaLN_modulation2(x.squeeze(1)).chunk(2, dim=1)

        # condition = modulate(self.norm1(condition), shift_mca, scale_mca)
        # x = x + gate_msa.unsqueeze(1) *self.SE(modulate(self.norm2(x), shift_msa, scale_msa))

        ## spayer ok camera just soso
        # x = x + gate_msa.unsqueeze(1)*modulate(self.norm1(self.CA(condition,condition,x)), shift_msa, scale_msa)
        # x = x + gate_mlp.unsqueeze(1)*modulate(self.norm2(self.mlp(x)),shift_mlp, scale_mlp)

        ### spayer terrible
        # x = modulate(self.norm1(x + gate_msa.unsqueeze(1)*self.CA(condition,condition,x)), shift_msa, scale_msa)
        # x = modulate(self.norm2(x + gate_mlp.unsqueeze(1)*self.mlp(x)),shift_mlp, scale_mlp)

        ### test 0408(1)
        # x = x + gate_msa.unsqueeze(1)*modulate(self.norm1(self.CA(condition,condition,x)), shift_msa, scale_msa)
        # x = x + self.norm2(self.mlp(x))

        ### spayer terrible,camera ok
        # x = x + gate_msa.unsqueeze(1)*self.CA(condition,condition,modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1)*self.mlp(modulate(self.norm2(x),shift_mlp, scale_mlp))

        ### test 0408(2)
        x = (1-gate_msa.unsqueeze(1))*x + gate_msa.unsqueeze(1)*modulate(self.norm1(self.CA(condition,condition,x)), shift_msa, scale_msa)
        x = (1-gate_mlp.unsqueeze(1))*x + gate_mlp.unsqueeze(1)*modulate(self.norm2(self.mlp(x)),shift_mlp, scale_mlp)
        # x = x + self.norm2(self.mlp(x))
        return x




# class SENet(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
       
#         self.calibrator1 = nn.Linear(hidden_size,hidden_size)
#         # self.calibrator2 = nn.Linear(hidden_size,hidden_size)
#         self.sig = nn.Sigmoid()
#         # self.act = nn.SiLU()

#     def forward(self, x):

#         w = self.calibrator1(x)
#         # w = self.act(w)
#         # w = self.calibrator2(w)
#         w = self.sig(w)

#         return w*x

class SENet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
       
        self.calibrator1 = nn.Conv1d(1,1,kernel_size=3,padding=1,bias=True)
        # self.calibrator2 = nn.Linear(hidden_size,hidden_size)
        self.sig = nn.Sigmoid()
        # self.act = nn.SiLU()

    def forward(self, x):
        # x = rearrange(x,'b n c->b c n')
        w = self.calibrator1(x)
        # print(w.shape)
        # w = self.act(w)
        # w = self.calibrator2(w)
        w = self.sig(w)
        x = w*x
        # x = rearrange(x,'b c n->b n c')

        return w*x






class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # self.values = nn.Linear(self.embed_size*2, self.embed_size, bias=False)
        # self.keys = nn.Linear(self.embed_size*2, self.embed_size, bias=False)
        # self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

        # self.weight = nn.Parameter(torch.rand(self.embed_size,5))
        

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        # print(value.shape)
        # exit()
        values = self.values(value).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)



        # Einsum does all the heavy lifting
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Scale
        energy = energy / (self.embed_size ** (1 / 2))

        attention = torch.softmax(energy, dim=3)
        # print(111,attention)
        # exit()

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out



# class PE(nn.Module):
#     def __init__(self, embed_size, max_len=1000):
#         super(PE, self).__init__()
#         pe = torch.zeros(max_len,embed_size)
#         position = torch.arange(0,max_len,dtype=torch.float64).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0,embed_size,2,dtype=torch.float64)*-(np.log(1000.0)/embed_size))
#         pe[:,0::2] = torch.sin(position*div_term)
#         pe[:,1::2] = torch.cos(position*div_term)
#         pe = pe.unsqueeze(0).transpose(0,1)
#         self.register_buffer('pe',pe)
#     def forward(self, x):
#         x = x + self.pe[x.size(1),:]
#         return x




class SemanticsModulatedAttention(nn.Module):

    def __init__(self, latent_dim,
                       num_heads,
                       dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)

        self.query = nn.Linear(latent_dim, latent_dim)


        self.key_retr = nn.Linear(latent_dim, latent_dim)
        self.value_retr = nn.Linear(latent_dim, latent_dim)

        self.key_pre = nn.Linear(latent_dim, latent_dim)
        self.value_pre = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=self.num_heads, dim_feedforward=latent_dim*4, dropout=dropout)
        # self.CA = CrossAttention(latent_dim,heads=self.num_heads)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim*4, act_layer=approx_gelu, drop=dropout)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=self.num_heads, dim_feedforward=latent_dim*4, dropout=dropout,batch_first=True,norm_first=True)



    
    def forward(self, x, condition,cm_pose):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        H = self.num_heads 
        # N = x.shape[1] + condition.shape[1] + cm_pose.shape[1]
         # m_pose.shape[1]+

        # B, T, 
        condition = torch.cat([condition,cm_pose],1)

        # x = self.norm1(x+self.query(x))
        y = self.decoder_layer(x,condition)
        # z = self.norm3(y+self.mlp(y))

        return y



# class SAM_BLOCK(nn.Module):
#     def __init__(self, latent_dim,
#                        text_latent_dim,
#                        num_heads,
#                        dropout,
#                        time_embed_dim):
#         super().__init__()

#         self.layer_norm = nn.LayerNorm(latent_dim)

#         self.mlp = Mlp(latent_dim, latent_dim * 4, latent_dim,nn.GELU,dropout)

#         self.sam_att = SemanticsModulatedAttention(latent_dim,
#                        text_latent_dim,
#                        num_heads,
#                        dropout,
#                        time_embed_dim)
#         self.drop = DropPath(dropout)

#         self.mlp1 = Mlp(latent_dim, latent_dim * 4, latent_dim,nn.GELU,dropout)

#         self.sam_att1 = SemanticsModulatedAttention(latent_dim,
#                        text_latent_dim,
#                        num_heads,
#                        dropout,
#                        time_embed_dim)




#     def forward(self, x,cond,m_pose,cm_pose):
#         # x = self.sam_att1(x,cm_pose,cm_pose,cm_pose)
#         # x = x + self.mlp1(self.layer_norm(x))

#         x = self.sam_att(x,cond,m_pose,cm_pose)
#         x = x + self.mlp(self.layer_norm(x))

#         return x


class My_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size,num_heads):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
    def forward(self, x, cond,memory):
        c = torch.cat((cond,memory),1)
        c = self.attn(self.norm(c))
        c = c.mean(dim=1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class SAMFormer(nn.Module):
    def __init__(self, num_block,latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout,
                       time_embed_dim):
        super(SAMFormer, self).__init__()

        self.num_block = num_block
        self.sam_block = []
        for l in range(self.num_block):
            print(l)
            # self.sam_block.append(SAM_BLOCK(latent_dim,
            #            text_latent_dim,
            #            num_heads,
            #            dropout,
            #            time_embed_dim))
            self.sam_block.append(DiTBlock(latent_dim,num_heads))

        self.sam_block = nn.ModuleList(self.sam_block)
        # self.final = FinalLayer(latent_dim,num_heads)



    def forward(self, X,cond,m_pose,cm_pose):
        for i in range(self.num_block):
            X = self.sam_block[i](X,cond,cm_pose)
        # X = self.final(X,cond,cm_pose)
        return X