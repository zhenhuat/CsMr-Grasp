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
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.structures import Meshes


class ActionClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_dims=[64, 256, 64], token_compression_ratios=[4, 4, 4, 64]):
        super().__init__()
        
        # 初始通道扩展 (3 → 64)
        self.initial_conv = nn.Conv1d(input_channels, hidden_dims[0], kernel_size=1)
        
        # 通道先升后降的模块序列
        self.channel_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.channel_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )
        
        # Token 压缩模块序列（通过跨步卷积）
        self.token_compressors = nn.ModuleList()
        for ratio in token_compression_ratios:
            self.token_compressors.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=ratio, stride=ratio),
                    nn.BatchNorm1d(hidden_dims[-1]),
                    nn.ReLU()
                )
            )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            # nn.Sigmoid()  # 二分类输出
        )

    def forward(self, x):
        # x: (B, 4874, 3)
        x = x.permute(0, 2, 1)  # (B, 3, 4874)
        
        # 初始通道扩展
        x = self.initial_conv(x)  # (B, 64, 4874)
        
        # 通道变换：64 → 256 → 64
        for block in self.channel_blocks:
            x = block(x)
        
        # Token 压缩：4874 → 1218 → 304 → 76 → 1
        for compressor in self.token_compressors:
            x = compressor(x)
        
        # 全局平均 pooling (B, 64, 1) → (B, 64)
        x = x.squeeze(-1)
        
        # 二分类输出
        return self.classifier(x)



class MultiScaleResBlock(nn.Module):
    """使用全连接层实现多尺度"""
    def __init__(self, Fin, Fout, scales=[1,2,4], n_neurons=256):
        super(MultiScaleResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.scales = scales
        
        # 多尺度全连接分支
        self.scale_branches = nn.ModuleList()
        for scale in scales:
            # 使用不同的隐藏层大小模拟多尺度
            hidden_size = n_neurons // (scale * 2)
            branch = nn.Sequential(
                nn.Linear(Fin, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.scale_branches.append(branch)
        
        # 融合层
        total_features = sum([n_neurons // (scale * 2) for scale in scales])
        self.fusion = nn.Sequential(
            nn.Linear(total_features, n_neurons),
            nn.BatchNorm1d(n_neurons),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(n_neurons, Fout)
        
        # shortcut连接
        if Fin != Fout:
            self.shortcut = nn.Linear(Fin, Fout)
        else:
            self.shortcut = nn.Identity()
            
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x: [B, Fin]
        
        # 多尺度处理
        scale_features = []
        for branch in self.scale_branches:
            feat = branch(x)
            scale_features.append(feat)
        
        # 融合多尺度特征
        fused = torch.cat(scale_features, dim=-1)
        fused = self.fusion(fused)
        
        # 输出投影
        out = self.output_proj(fused)
        
        # 残差连接
        shortcut = self.shortcut(x)
        out = out + shortcut
        
        return self.activation(out)


@MODEL.register_module(name="CoarseIntentNetCS")
class CoarseIntentNetIntentEmbedEncode(CoarseNet):

    def __init__(self, cfg):
        self.n_intents = cfg.DATA_PRESET.N_INTENTS
        super(CoarseIntentNetIntentEmbedEncode, self).__init__(cfg)

    def _build_layers(self):
        in_bps = self.in_bps
        in_pose = self.in_pose
        n_neurons = self.n_neurons
        latentD = self.latentD
        intentD = self.latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + intentD + in_pose)
        self.enc_rb1 = MultiScaleResBlock(in_bps + intentD + in_pose, n_neurons)
        self.enc_rb2 = MultiScaleResBlock(n_neurons + in_bps + intentD + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)  # normalize the bps_torch for object
        self.dec_rb1 = MultiScaleResBlock(latentD + intentD + in_bps, n_neurons)
        self.dec_rb2 = MultiScaleResBlock(n_neurons + latentD + intentD + in_bps, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

        self.intent_embed = nn.Embedding(self.n_intents, intentD)  # (3, 16)
        nn.init.uniform_(self.intent_embed.weight.data, 0, 1)


        self.classifier = ActionClassifier()

        # 加载训练好的分类器权重
        # self.classifier.load_state_dict(torch.load("/home/robin/fareal/OakInk-Grasp-Generation-main/best_action_classifier_verts.pth"))
        self.classifier.load_state_dict(torch.load("/home/zhenhua/grasp/OakInk-Grasp-Generation-main//classification/best_classifier.pth"))
        self.classifier.eval()  # 设置为评估模式

        self.center_idx = 9
        self.mano_layer = ManoLayer(rot_mode="axisang",
                                    center_idx=self.center_idx,
                                    mano_assets_root="assets/mano_v1_2",
                                    use_pca=False,
                                    flat_hand_mean=True)

    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat, intent):
        bs = bps_object.shape[0]
        X = torch.cat([intent, bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)
        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0)
        X = self.enc_rb2(torch.cat([X0, X], dim=1))

        mean = self.enc_mu(X)
        std = F.softplus(self.enc_var(X))

        # mean = torch.clamp(mean, min=-10, max=10)  # 限制均值范围
        std = torch.clamp(std, min=1e-9, max=1.0)  # 限制标准差范围
        # log_prob = -0.5 * ((z - mean) ** 2 / (std ** 2) + torch.log(std ** 2) + np.log(2 * np.pi))
        return torch.distributions.normal.Normal(mean, std)

    def decode(self, Zin, bps_object, intent,intent_vec):
        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)
        X0 = torch.cat([Zin, intent, o_bps], dim=1)
        X = self.dec_rb1(X0)
        X = self.dec_rb2(torch.cat([X0, X], dim=1))
        pose = self.dec_pose(X)
        trans = self.dec_trans(X)
        results = parms_decode(pose, trans)
        results["z"] = Zin

        return results

    def project_to_gaussian(self,z, eps=1e-8):
        z_mean = z.mean(dim=1, keepdim=True)
        z_std = z.std(dim=1, keepdim=True)
        z_std = torch.max(z_std, torch.tensor(eps, device=z.device))
        return (z - z_mean) / z_std


    def adjust_latent_space(self,  classifier, decoder,  bps_object, intent, intent_vec,  z_s, tgt,  alpha=0.0001, beta=0.001, gamma=0.001,step_scale=0.01):
        batch_size = bps_object.shape[0]
        device = bps_object.device
        with torch.enable_grad():  # 启用梯度计算
            z_s.requires_grad_(True)
            z_s.retain_grad()

            # 解码生成样本
            x_gen = decoder(z_s, bps_object, intent, intent_vec)

            # 使用 MANO 层生成手部网格
            mano_pose = torch.cat([x_gen["global_orient"], x_gen["hand_pose"]], dim=1)  # (B, 48)
            mano_shape = tgt[Queries.HAND_SHAPE]  # (B, 10)
            mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)
            verts_rhand = mano_output.verts + x_gen["transl"].unsqueeze(1)  # (B, 778, 3)
            obj_verts = tgt[Queries.OBJ_VERTS_OBJ_DS]  # (B, 4096, 3)

            # 分类器输入特征
            features = torch.cat([verts_rhand, obj_verts], dim=1)  # (B, 778 + 4096, 3)
            logits = classifier(features)  # 分类器预测

            # 计算分类损失
            cls_labels = torch.argmax(tgt[Queries.INTENT_VEC], dim=1)  # (batch_size,)
            loss = F.cross_entropy(logits, cls_labels)

            # 反向传播计算梯度（保留计算图）
            loss.backward(retain_graph=True)

            # 获取潜在变量的梯度
            if z_s.grad is None:
                raise RuntimeError("Gradient for z is None. Ensure z participates in the computation graph.")
            grad_z = z_s.grad.data  # (B, latent_dim)

            # 计算分类器的置信度（softmax概率）
            prob = F.softmax(logits, dim=1)  # (batch_size, num_classes)
            confidence = prob.gather(1, cls_labels.unsqueeze(1))  # (batch_size, 1)
            confidence = confidence.view(-1, *([1] * (z_s.dim() - 1)))  # 广播到 grad_z 的形状


            # 2. 以置信度作为步幅（高置信度样本移动幅度更大）
            # step_size = alpha * confidence  # (B, 1, ...)
            z_adjusted = z_s - alpha * grad_z + beta * (z_s - z_s.mean())  # 沿梯度方向移动
            z_adjusted = z_adjusted.detach()  # 断开梯度连接

            # # 1. 计算梯度方向（单位向量）
            # grad_direction = grad_z / (grad_z.norm(dim=-1, keepdim=True) + 1e-8)  # (B, latent_dim)

            # # 2. 以置信度作为步幅（高置信度样本移动幅度更大）
            # step_size = step_scale * confidence  # (B, 1, ...)
            # z_adjusted = z_s - step_size * grad_direction  # 沿梯度方向移动

            # 3. 可选：仍然保留方差缩放（如果希望同时调整方差）
            # z_variance_scale = 1.0 / (1.0 + gamma * confidence * grad_z.abs())
            # z_adjusted = z_adjusted * z_variance_scale

            # 4. 投影回高斯分布（可选）
            # z_adjusted = self.project_to_gaussian(z_adjusted)

        return z_adjusted




    def sample_poses(self, bps_object, intent,intent_vec,inp, seed=None):
        bs = bps_object.shape[0]

        if seed is not None:
            np_rand_state = np.random.get_state()
            np.random.seed(seed)

        dtype = bps_object.dtype
        device = bps_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0.0, 1.0, size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen, dtype=dtype).to(device)

            Zgen = self.adjust_latent_space(self.classifier, self.decode, bps_object, intent, intent_vec, Zgen, inp)

        if seed is not None:
            np.random.set_state(np_rand_state)

        result = self.decode(Zgen, bps_object, intent,intent_vec)
        return result

    def _forward_impl(self, bps_object, trans_rhand, global_orient_rhand_rotmat, intent, intent_vec, inp, **kwargs):
        z_dist = self.encode(bps_object, trans_rhand, global_orient_rhand_rotmat, intent)  # 返回 Normal 分布
        z_s = z_dist.rsample()  # 从分布中采样潜在变量
        log_probs = z_dist.log_prob(z_s).sum(dim=-1)  # 计算对数概率
        z_adjusted = self.adjust_latent_space(self.classifier, self.decode, bps_object, intent, intent_vec, z_s, inp)
        hand_parms = self.decode(z_adjusted, bps_object, intent, intent_vec)
        results = {"mean": z_dist.mean, "std": z_dist.scale, 'probs': log_probs}
        results.update(hand_parms)
        return results

    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        bps_object = inp[Queries.OBJ_BPS]  # (B, 4096)
        trans_rhand = inp[Queries.HAND_TRANSL_OBJ]  # (B, 3)
        global_hand_pose_obj = inp[Queries.HAND_POSE_OBJ][:, :3]  # (B, 3)
        global_orient_rhand_rotmat = aa_to_rotmat(global_hand_pose_obj).unsqueeze(1)  # (B, 1, 3, 3)
        intent_vec = inp[Queries.INTENT_VEC]  # (B, INTENT_D)
        bs = bps_object.shape[0]

        # intent_vec has shape: (B, 3), the last dim is a one hot vector that indicates the intent ID
        # self.intent_embed.weight has shape: (3, 16), each row is a vector that represents an intent embedding
        # now use the intent_vec to index self.intent_embed
        # intent has shape: (B, 16)
        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)

        res = self._forward_impl(bps_object, trans_rhand, global_orient_rhand_rotmat, intent,intent_vec,inp)
        cnet_loss = self.compute_loss(prd=res, tgt=inp, **kwargs)
        if step_idx % self.log_freq == 0:
            for k, v in cnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, cnet_loss

    def testing_step(self, inp, step_idx, intent_name=None, return_latent=False, **kwargs):
        """
        Testing step with optional latent space analysis
        
        Args:
            return_latent: if True, return original and adjusted latent variables
        """
        obj_bps = inp[Queries.OBJ_BPS]
        bs = obj_bps.shape[0]
        
        if intent_name is not None:
            intent_id = int(ALL_INTENT[intent_name])
            intent_vec = torch.zeros((bs, self.n_intents), dtype=torch.float32)
            intent_vec[torch.arange(bs), intent_id - 1] = 1.0
            intent_vec = intent_vec.to(obj_bps.device)
            inp[Queries.INTENT_VEC] = intent_vec
        else:
            intent_vec = inp[Queries.INTENT_VEC]

        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)
        
        # 如果是分析潜空间，使用 encoder 的输出而不是随机噪声
        if return_latent:
            # 使用训练时的 forward 但不计算损失
            trans_rhand = inp[Queries.HAND_TRANSL_OBJ]
            global_hand_pose_obj = inp[Queries.HAND_POSE_OBJ][:, :3]
            global_orient_rhand_rotmat = aa_to_rotmat(global_hand_pose_obj).unsqueeze(1)
            
            # encoder 输出
            z_dist = self.encode(obj_bps, trans_rhand, global_orient_rhand_rotmat, intent)
            z_mean = z_dist.mean  # encoder 输出的均值（确定性部分）
            z_original = z_dist.rsample()  # 从分布采样（用于对比）
            
            # 调整后的潜变量
            z_adjusted = self.adjust_latent_space(
                self.classifier, self.decode, obj_bps, intent, intent_vec, z_original, inp
            )
            
            # decode 调整后的 z
            result = self.decode(z_adjusted, obj_bps, intent, intent_vec)
            result["z_mean"] = z_mean  # encoder 输出的均值
            result["z_original"] = z_original  # 从 encoder 分布采样的 z
            result["z_adjusted"] = z_adjusted  # 调整后的 z
            result["z_std"] = z_dist.scale
        else:
            # 原来的逻辑：从随机噪声开始
            res = self.sample_poses(obj_bps, intent=intent, intent_vec=intent_vec, inp=inp, seed=kwargs.get("seed", None))
            result = res
        
        result["mano_pose"] = torch.cat([result["global_orient"], result["hand_pose"]], dim=1)
        mano_out = self.mano_layer(result["mano_pose"])
        result["hand_verts"] = mano_out.verts + result["transl"].unsqueeze(1)
        result = {f"{self.stage}.{k}": v for k, v in result.items()}
        return result, {}

