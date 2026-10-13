import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------
# 项目路径
# ----------------------------
sys.path.append('/home/robin/fareal/icassp_grasp/')

from einops import rearrange
from manotorch.manolayer import ManoLayer, MANOOutput
from lib.utils.builder import MODEL
from lib.utils.transform import rotmat_to_aa, aa_to_rotmat
from collections import namedtuple
from lib.utils.transform import *

# ----------------------------
# 1. 共享工具函数（保留原始 parms_decode）
# ----------------------------

def parms_decode(pose, trans):
    bs = trans.shape[0]
    pre_p16 = pose
    pose_full = CRot2rotmat(pose)
    pose = pose_full.view([bs, 1, -1, 3, 3])
    pose = rotmat_to_aa(pose).view(bs, -1)
    global_orient = pose[:, :3]
    hand_pose = pose[:, 3:]
    pose_full = pose_full.view([bs, -1, 3, 3])
    hand_parms = {"global_orient": global_orient, "hand_pose": hand_pose, "transl": trans, "fullpose": pose_full, 'pre_p16': pre_p16}
    return hand_parms

def CRot2rotmat(pose):
    reshaped_input = pose.view(-1, 3, 2)
    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)

# ----------------------------
# 2. ActionClassifier（完全保留）
# ----------------------------

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
        for block in self.channel_blocks: x = block(x)
        for compressor in self.token_compressors: x = compressor(x)
        return self.classifier(x.squeeze(-1))

# ----------------------------
# 3. 新的生成模型组件（来自你提供的新模型）
# ----------------------------

class MultiScaleResBlock(nn.Module):
    """使用全连接层实现多尺度"""
    def __init__(self, Fin, Fout, scales=[1,2,4], n_neurons=256):
        super(MultiScaleResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.scales = scales
        
        self.scale_branches = nn.ModuleList()
        for scale in scales:
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
        
        total_features = sum([n_neurons // (scale * 2) for scale in scales])
        self.fusion = nn.Sequential(
            nn.Linear(total_features, n_neurons),
            nn.BatchNorm1d(n_neurons),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output_proj = nn.Linear(n_neurons, Fout)
        
        if Fin != Fout:
            self.shortcut = nn.Linear(Fin, Fout)
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        scale_features = []
        for branch in self.scale_branches:
            feat = branch(x)
            scale_features.append(feat)
        fused = torch.cat(scale_features, dim=-1)
        fused = self.fusion(fused)
        out = self.output_proj(fused)
        shortcut = self.shortcut(x)
        out = out + shortcut
        return self.activation(out)

# ----------------------------
# 4. CoarseNet 基类（简化版，仅定义必要属性）
# ----------------------------

class CoarseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latentD = 16
        self.in_bps = 4096
        self.in_pose = 96
        self.n_neurons = 256
        self.center_idx = 9  # 注意：新模型使用 center_idx=9
        self.stage = "coarse"
        self.log_freq = 100

# ----------------------------
# 5. 主模型：CoarseIntentNetIntentEmbedEncode（完全替换为新结构）
# ----------------------------

@MODEL.register_module()
class CoarseIntentNetIntentEmbedEncode(CoarseNet):

    def __init__(self, cfg):
        # 注意：新模型从 cfg.DATA_PRESET.N_INTENTS 读取
        self.n_intents = cfg.get("N_INTENTS", cfg.get("DATA_PRESET", {}).get("N_INTENTS", 2))
        super(CoarseIntentNetIntentEmbedEncode, self).__init__(cfg)
        self._build_layers()

    def _build_layers(self):
        in_bps = self.in_bps
        in_pose = self.in_pose
        n_neurons = self.n_neurons
        latentD = self.latentD
        intentD = latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + intentD + in_pose)
        self.enc_rb1 = MultiScaleResBlock(in_bps + intentD + in_pose, n_neurons)
        self.enc_rb2 = MultiScaleResBlock(n_neurons + in_bps + intentD + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = MultiScaleResBlock(latentD + intentD + in_bps, n_neurons)
        self.dec_rb2 = MultiScaleResBlock(n_neurons + latentD + intentD + in_bps, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

        self.intent_embed = nn.Embedding(self.n_intents, intentD)
        nn.init.uniform_(self.intent_embed.weight.data, 0, 1)

        # 初始化分类器
        self.classifier = ActionClassifier()
        try:
            self.classifier.load_state_dict(torch.load("/home/robin/fareal/icassp_grasp/classification/best_classifier.pth", map_location='cpu'))
        except Exception as e:
            print(f"Warning: Failed to load classifier weights: {e}")
        self.classifier.eval()

        # MANO Layer
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat, intent):
        bs = bps_object.shape[0]
        X = torch.cat([intent, bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)
        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0)
        X = self.enc_rb2(torch.cat([X0, X], dim=1))
        mean = self.enc_mu(X)
        std = F.softplus(self.enc_var(X))
        std = torch.clamp(std, min=1e-9, max=1.0)
        return torch.distributions.normal.Normal(mean, std)

    def decode(self, Zin, bps_object, intent, intent_vec):
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

    def project_to_gaussian(self, z, eps=1e-8):
        z_mean = z.mean(dim=1, keepdim=True)
        z_std = z.std(dim=1, keepdim=True)
        z_std = torch.max(z_std, torch.tensor(eps, device=z.device))
        return (z - z_mean) / z_std

    def adjust_latent_space(self, classifier, decoder, bps_object, intent, intent_vec, z_s, tgt,
                            alpha=0.0001, beta=0.001, gamma=0.001, step_scale=0.01):
        batch_size = bps_object.shape[0]
        device = bps_object.device
        with torch.enable_grad():
            z_s.requires_grad_(True)
            z_s.retain_grad()

            x_gen = decoder(z_s, bps_object, intent, intent_vec)

            mano_pose = torch.cat([x_gen["global_orient"], x_gen["hand_pose"]], dim=1)
            mano_shape = tgt.get("mano_shape", torch.zeros(batch_size, 10, device=device))  # fallback
            obj_verts = tgt.get("obj_verts", torch.randn(batch_size, 4096, 3, device=device))  # fallback

            mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)
            verts_rhand = mano_output.verts + x_gen["transl"].unsqueeze(1)
            features = torch.cat([verts_rhand, obj_verts], dim=1)
            logits = classifier(features)

            cls_labels = torch.argmax(intent_vec, dim=1)
            loss = F.cross_entropy(logits, cls_labels)
            loss.backward(retain_graph=True)

            if z_s.grad is None:
                raise RuntimeError("Gradient for z is None.")
            grad_z = z_s.grad.data
            prob = F.softmax(logits, dim=1)
            confidence = prob.gather(1, cls_labels.unsqueeze(1))
            confidence = confidence.view(-1, *([1] * (z_s.dim() - 1)))

            z_adjusted = z_s - alpha * grad_z + beta * (z_s - z_s.mean())
            return z_adjusted.detach()

    def sample_poses(self, bps_object, intent, intent_vec, inp, seed=None):
        bs = bps_object.shape[0]
        if seed is not None:
            state = torch.get_rng_state()
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        dtype = bps_object.dtype
        device = bps_object.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.randn(bs, self.latentD, dtype=dtype, device=device)
            # 可选：启用 SNS 调整（当前注释掉以匹配原始 profiling 行为）
            Zgen = self.adjust_latent_space(self.classifier, self.decode, bps_object, intent, intent_vec, Zgen, inp)

        if seed is not None:
            torch.set_rng_state(state)

        result = self.decode(Zgen, bps_object, intent, intent_vec)
        return result

    def testing_step(self, inp, step_idx, intent_name=None, **kwargs):
        obj_bps = inp["obj_bps"]
        obj_verts = inp.get("obj_verts", None)
        mano_shape = inp.get("mano_shape", None)
        intent_vec = inp["intent_vec"]
        bs = obj_bps.shape[0]

        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)
        
        # 构造 tgt 用于 adjust_latent_space（如果启用）
        fake_tgt = {
            "mano_shape": mano_shape if mano_shape is not None else torch.zeros(bs, 10, device=obj_bps.device),
            "obj_verts": obj_verts if obj_verts is not None else torch.randn(bs, 4096, 3, device=obj_bps.device),
            "intent_vec": intent_vec
        }

        res = self.sample_poses(obj_bps, intent=intent, intent_vec=intent_vec, inp=fake_tgt, seed=kwargs.get("seed", None))

        res["mano_pose"] = torch.cat([res["global_orient"], res["hand_pose"]], dim=1)
        mano_out = self.mano_layer(res["mano_pose"])
        res["hand_verts"] = mano_out.verts + res["transl"].unsqueeze(1)
        return res, {}

# ----------------------------
# 6. Profiling 主函数（适配新模型）
# ----------------------------

def count_params(module):
    return sum(p.numel() for p in module.parameters())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建模型
    cfg = {"N_INTENTS": 2}
    model = CoarseIntentNetIntentEmbedEncode(cfg).to(device)
    model.eval()

    # 参数量统计
    total = count_params(model)
    print(f"\n=== Parameter Count ===")
    print(f"Total: {total / 1e6:.2f}M")
    print(f"  ├── Intent Embed: {count_params(model.intent_embed) / 1e3:.1f}K")
    print(f"  ├── Encoder: {count_params(model.enc_rb1) + count_params(model.enc_rb2) + count_params(model.enc_mu) + count_params(model.enc_var):.0f}")
    print(f"  ├── Decoder: {count_params(model.dec_rb1) + count_params(model.dec_rb2) + count_params(model.dec_pose) + count_params(model.dec_trans):.0f}")
    print(f"  └── Classifier: {count_params(model.classifier) / 1e6:.3f}M")

    # 构造 dummy input
    B = 1
    dummy_inp = {
        "obj_bps": torch.randn(B, 4096, device=device, dtype=torch.float32),
        "obj_verts": torch.randn(B, 4096, 3, device=device, dtype=torch.float32),
        "mano_shape": torch.zeros(B, 10, device=device, dtype=torch.float32),
        "intent_vec": torch.eye(2, device=device, dtype=torch.float32)[0:1],
    }

    # 推理时间测量
    print(f"\n=== Inference Latency (VAE Sampling) ===")
    with torch.no_grad():
        # Warm-up
        for _ in range(3):
            _ = model.testing_step(dummy_inp, 0)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = model.testing_step(dummy_inp, 0)
        torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / 10
    print(f"Average inference time over 10 runs: {avg_time:.4f} seconds")
    print(f"Approximate FPS: {1.0 / avg_time:.2f}")

if __name__ == "__main__":
    main()