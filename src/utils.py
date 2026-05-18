# -*- coding: utf-8 -*-
# 工具模块：全局模型工厂、随机种子、可微增广（DiffAugment）、切片 Wasserstein（SWDLoss）、
# 服务端非对称监督对比损失（SupervisedContrastiveLoss）。
# 被 main.py / client.py / server.py / baselines 引用；增广默认 aug_mode='S'（每步随机选一个算子块）。
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import ConvNet, Projector


class SupervisedContrastiveLoss(nn.Module):
    # 非对称监督对比：锚点为 batch 嵌入经 Projector 与 L2 归一化后的向量；
    # 正样本由各类「上一通信轮末统计的」原型 prev_syn_proto 按标签 target 索引得到（与 server.py 中 asymmetric=True 一致）。
    # proto 与 batch 的类间软关系由 proto 两两相似度经温度缩放得到（soft_relation_mask）。

    def __init__(self, num_classes, device, temperature=0.07, z_dim=10):
        super().__init__()
        self.device = device
        self.head = Projector(input_dim=z_dim, output_dim=z_dim).to(self.device)
        self.head.train()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, x, y, proto, asymmetric=True):
        # asymmetric=False 时不过 Projector；当前工程内 server 恒传 asymmetric=True，对称分支未使用。
        if asymmetric:
            x = self.head(x)
        x = F.normalize(x, dim=1)
        sim_matrix = torch.exp(torch.matmul(x, proto.t()) / self.temperature)
        proto_norm = F.normalize(proto, dim=1)
        batch_sim = torch.matmul(proto_norm, proto_norm.t())
        soft_relation_mask = torch.exp(batch_sim / 0.5)
        soft_relation_mask = soft_relation_mask / soft_relation_mask.max(dim=1, keepdim=True)[0]
        sample_soft_mask = soft_relation_mask[y]
        mask = F.one_hot(y, num_classes=sim_matrix.shape[1]).float()
        numerator = (mask * sim_matrix).sum(1)
        denominator = (sample_soft_mask * sim_matrix).sum(1) - numerator + 1e-8
        return -torch.log(numerator / (numerator + denominator)).mean()


def get_model(model_name, dataset_info):
    # 仅 ConvNet / ConvNetBN：后者为 BatchNorm，前者为 InstanceNorm，其余超参相同。
    if model_name == "ConvNet":
        return ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='instancenorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size'],
        )
    if model_name == "ConvNetBN":
        return ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='batchnorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size'],
        )
    raise NotImplementedError("仅支持 ConvNet / ConvNetBN（与 run.sh 一致）")


def setup_seed(seed):
    # 固定 PyTorch / CUDA / NumPy / Python random，并关闭 cudnn benchmark 以保证可复现（略降卷积性能）。
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class ParamDiffAug:
    # DiffAugment 的全局超参容器；rand_* 从此读 ratio / prob。
    # aug_mode：'S' 每步从 strategy 里随机抽一个算子名（如 color），再执行其函数列表；'M' 为逐名全套串行（当前未启用，见 DiffAugment）。
    def __init__(self):
        self.aug_mode = 'S'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        # self.ratio_noise = 0.05  # 预留；AUGMENT_FNS 中无噪声算子，当前未使用
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    # 在 Siamese 模式下按 param.latestseed 固定随机流，使同 batch 内各样本共享同一几何/颜色变换。
    if param.latestseed == -1:
        return
    torch.random.manual_seed(param.latestseed)
    param.latestseed += 1


def DiffAugment(x, strategy='', seed=-1, param=None):
    # 对 batch 图像 x 施加可微增广；strategy 为下划线拼接的算子名序列，如 color_crop_cutout_flip_scale_rotate。
    # seed!=-1 时 param.Siamese=True，双分支（真/合成）共用同一随机性（client/server 传入毫秒种子）。
    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True
    param.latestseed = seed
    if strategy == 'None' or strategy == 'none':
        return x
    if strategy:
        # ----- aug_mode 'M'：对 strategy 中每个名字依次执行其全部子算子（串联多增广）-----
        # 当前 ParamDiffAug 默认 aug_mode='S'，主实验与 run.sh 均未使用 M；若需启用，取消下行注释并设 param.aug_mode='M'。
        # if param.aug_mode == 'M':
        #     for p in strategy.split('_'):
        #         for f in AUGMENT_FNS[p]:
        #             x = f(x, param)
        if param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s' % param.aug_mode)
        x = x.contiguous()
    return x


def rand_scale(x, param):
    # 各样本独立随机缩放（affine grid）；Siamese 时用 batch 内第一份变换复制到全体。
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0], device=x.device, dtype=x.dtype) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0], device=x.device, dtype=x.dtype) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=x.dtype, device=x.device)
    if param.Siamese:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape)
    return F.grid_sample(x, grid)


def rand_rotate(x, param):
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0], device=x.device, dtype=x.dtype) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
              [torch.sin(theta[i]), torch.cos(theta[i]), 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=x.dtype, device=x.device)
    if param.Siamese:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape)
    return F.grid_sample(x, grid)


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        randb[:] = randb[0].clone()
    return x + (randb - 0.5) * ratio


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        rands[:] = rands[0].clone()
    return (x - x_mean) * (rands * ratio) + x_mean


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        randc[:] = randc[0].clone()
    return (x - x_mean) * (randc + ratio) + x_mean


def rand_crop(x, param):
    # 整图 pad 后按随机平移重采样，等价带边界的随机平移裁剪。
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


# strategy 中每个 token 对应一串可微算子；DiffAugment 在 aug_mode='S' 下每步随机抽一个 token 执行其列表。
AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


class SWDLoss(nn.Module):
    # Sliced Wasserstein：随机投影到 num_projections 条方向，对投影后排序向量做 L2，再对投影取平均。
    # 当两类样本数 N!=M 时，对排序后的序列做 1D 线性插值重采样到同一长度再比较。

    def __init__(self, num_projections=50, device='cuda'):
        super().__init__()
        self.num_projections = num_projections
        self.device = device

    def forward(self, X, Y):
        d = X.shape[1]
        dev = X.device
        dt = X.dtype
        projections = torch.randn(self.num_projections, d, device=dev, dtype=dt)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)
        X_proj = torch.matmul(X, projections.T)
        Y_proj = torch.matmul(Y, projections.T)
        X_proj_sorted, _ = torch.sort(X_proj, dim=0)
        Y_proj_sorted, _ = torch.sort(Y_proj, dim=0)
        N, M = X.shape[0], Y.shape[0]
        if N != M:
            X_proj_sorted = F.interpolate(
                X_proj_sorted.T.unsqueeze(1), size=max(N, M), mode='linear', align_corners=False
            ).squeeze(1).T
            Y_proj_sorted = F.interpolate(
                Y_proj_sorted.T.unsqueeze(1), size=max(N, M), mode='linear', align_corners=False
            ).squeeze(1).T
        return torch.mean((X_proj_sorted - Y_proj_sorted) ** 2)
