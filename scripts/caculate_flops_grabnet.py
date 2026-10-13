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
from manotorch.manolayer import ManoLayer
from lib.utils.builder import MODEL
from lib.utils.transform import rotmat_to_aa

# ----------------------------
# 1. 工具函数：parms_decode + CRot2rotmat（必须保留）
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
# 2. ResBlock 定义（根据 GrabNet 风格复现）
# ----------------------------

class ResBlock(nn.Module):
    def __init__(self, Fin, Fout, n_neurons=None):
        super(ResBlock, self).__init__()
        if n_neurons is None:
            n_neurons = Fout
        self.Fin = Fin
        self.Fout = Fout
        self.bn1 = nn.BatchNorm1d(Fin)
        self.linear1 = nn.Linear(Fin, n_neurons)
        self.bn2 = nn.BatchNorm1d(n_neurons)
        self.linear2 = nn.Linear(n_neurons, Fout)
        if Fin != Fout:
            self.shortcut = nn.Linear(Fin, Fout)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, use_bn=True):
        if use_bn:
            x_in = self.bn1(x)
        else:
            x_in = x
        x_in = self.act(self.linear1(x_in))
        if use_bn:
            x_in = self.bn2(x_in)
        x_in = self.linear2(x_in)
        x_out = self.shortcut(x) + x_in
        return self.act(x_out)

# ----------------------------
# 3. CoarseNet 基类（简化）
# ----------------------------

class CoarseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latentD = 16
        self.in_bps = 4096
        self.in_pose = 96
        self.n_neurons = 256
        self.stage = "coarse"
        self.log_freq = 100
        self.center_idx = 9
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

# ----------------------------
# 4. 主模型：完全对齐你提供的版本
# ----------------------------

@MODEL.register_module()
class CoarseIntentNetIntentEmbedEncode(CoarseNet):

    def __init__(self, cfg):
        self.n_intents = cfg.get("N_INTENTS", 2)  # 兼容直接传 N_INTENTS
        super(CoarseIntentNetIntentEmbedEncode, self).__init__(cfg)
        self._build_layers()

    def _build_layers(self):
        in_bps = self.in_bps
        in_pose = self.in_pose
        n_neurons = self.n_neurons
        latentD = self.latentD
        intentD = self.latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + intentD + in_pose)
        self.enc_rb1 = ResBlock(in_bps + intentD + in_pose, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps + intentD + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = ResBlock(latentD + intentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + intentD + in_bps, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

        self.intent_embed = nn.Embedding(self.n_intents, intentD)
        nn.init.uniform_(self.intent_embed.weight.data, 0, 1)

    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat, intent):
        bs = bps_object.shape[0]
        X = torch.cat([intent, bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)
        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)
        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object, intent):
        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)
        X0 = torch.cat([Zin, intent, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        pose = self.dec_pose(X)
        trans = self.dec_trans(X)
        results = parms_decode(pose, trans)
        results["z"] = Zin
        return results

    def sample_poses(self, bps_object, intent, seed=None):
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

        if seed is not None:
            torch.set_rng_state(state)

        result = self.decode(Zgen, bps_object, intent)
        return result

    def testing_step(self, inp, step_idx, intent_name=None, **kwargs):
        obj_bps = inp["obj_bps"]
        bs = obj_bps.shape[0]
        intent_vec = inp["intent_vec"]

        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)
        res = self.sample_poses(obj_bps, intent=intent, seed=kwargs.get("seed", None))

        res["mano_pose"] = torch.cat([res["global_orient"], res["hand_pose"]], dim=1)
        mano_out = self.mano_layer(res["mano_pose"])
        res["hand_verts"] = mano_out.verts + res["transl"].unsqueeze(1)
        return res, {}

# ----------------------------
# 5. Profiling 主函数
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
    print(f"  ├── Encoder (ResBlocks + mu/var): {count_params(model.enc_rb1) + count_params(model.enc_rb2) + count_params(model.enc_mu) + count_params(model.enc_var):.0f}")
    print(f"  ├── Decoder (ResBlocks + pose/trans): {count_params(model.dec_rb1) + count_params(model.dec_rb2) + count_params(model.dec_pose) + count_params(model.dec_trans):.0f}")
    print(f"  └── MANO Layer: Non-trainable (ignored)")

    # Dummy input (B=1)
    B = 1
    dummy_inp = {
        "obj_bps": torch.randn(B, 4096, device=device, dtype=torch.float32),
        "intent_vec": torch.eye(2, device=device, dtype=torch.float32)[0:1],  # shape: [1, 2]
    }

    # Inference latency
    print(f"\n=== Inference Latency (VAE Sampling, B={B}) ===")
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