# -*- coding: utf-8 -*-
# 全局分类网络与对比学习投影头定义。
# - ConvNet：卷积骨干 + 线性分类头；embed() 输出展平后的特征向量，供客户端 SWD、服务端对比项使用。
# - ConvNet / ConvNetBN 均由 src/utils.py 的 get_model() 按 dataset_info 实例化（差异主要在 net_norm）。
# - Projector：MLP，将 embed 维度映射到对比损失所用低维空间（见 server 中非对称监督对比支路）。
import torch.nn as nn


class ConvNet(nn.Module):
    # 通用 VGG 风格小卷积网络：net_depth 个「Conv → (Norm) → Act → (Pool)」块，最后接 Linear(num_classes)。

    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
        super().__init__()
        # ----- 激活 -----
        if net_act == 'sigmoid':
            self.net_act = nn.Sigmoid()
        elif net_act == 'relu':
            self.net_act = nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

        # ----- 空间下采样：每块末尾可选 2×2 池化 -----
        if net_pooling == 'maxpooling':
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            self.net_pooling = None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

        # features：卷积堆叠；classifier：将展平特征映射到 num_classes 维 logits
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        # 标记是否曾处于 train 模式（forward 内使用，便于外部逻辑区分）
        self.modified = False

    def forward(self, x, train=False, mode='dummy', normalize='dummy'):
        # train=True 时同时返回展平特征 inter_out 与 logits，供少数需要「特征+分类」一并算损失的路径。
        # mode / normalize 为历史接口占位，当前未使用。
        if self.training:
            self.modified = True
        out = self.features(x)
        inter_out = out.reshape(out.size(0), -1)
        out = self.classifier(inter_out)
        if train:
            return inter_out, out
        return out

    def embed(self, x):
        # 仅骨干特征、不接分类头：输出 shape (N, num_feat)，客户端 SWD 与服务端对比正则均调用此方法。
        out = self.features(x)
        return out.reshape(out.size(0), -1)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = [C, H, W]，用于构造与特征图通道/空间尺寸匹配的归一化层。
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        if net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        if net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        if net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        if net_norm == 'none':
            return None
        exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        # 逐块搭建 Conv2d → (Norm) → ReLU → (Pool)，并同步更新 shape_feat 供后续 Linear 输入维数计算。
        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        # MedMNIST 等 28×28 输入：逻辑上先视为 32×32，首层用 padding=3 将 28 扩到 32，再与 CIFAR 共用后续块结构。
        if im_size[0] == 28:
            shape_feat = [in_channels, 32, 32]
        for d in range(net_depth):
            # 首层对单通道或 28 宽的三通道使用 padding=3，使特征图空间尺寸与 shape_feat 规划一致。
            layers += [nn.Conv2d(
                in_channels, net_width, kernel_size=3,
                padding=3 if (channel == 1 or (channel == 3 and im_size[0] == 28)) and d == 0 else 1,
            )]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self.net_act]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat


class Projector(nn.Module):
    # 多层感知机：input_dim → hidden_dim（× num_hidden 段）→ output_dim，用于对比学习中的特征投影与归一化前空间。

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_hidden=1, bn='batchnorm', activation='relu'):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dim)]
        if bn == 'batchnorm':
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        if activation == 'relu':
            self.layers.append(nn.ReLU(inplace=True))
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if bn == 'batchnorm':
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
