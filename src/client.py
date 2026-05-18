# -*- coding: utf-8 -*-
# 联邦学习客户端（数据浓缩 / Data Condensation）
# - 每个 Client 对应一个参与方，只在本地真实数据上工作；不上传原始样本，
#   而是优化「可学习合成图像」像素，使合成图在冻结骨干下的嵌入分布
#   与当前批真实图接近（SWD 损失，见 train_swd_condense）。
# - 全局分类模型 self.global_model 由服务端下发（receive_model）；浓缩时
#   冻结其全部参数，仅对 synthetic_images 反传。
# - 真实样本的抽样顺序由 PerLabelDatasetNonIID.cal_loss + pre_sample 决定：
#   用预测熵构造权重，再在每类上做有放回抽样（贯穿 dc_iterations+1 步）。
import copy
import logging
import math
import time

import torch
import torch.nn as nn
from dataset.data.dataset import PerLabelDatasetNonIID
from src.utils import DiffAugment, ParamDiffAug, get_model, SWDLoss


def get_gpu_mem_info(gpu_id=0):
    # 查询指定 GPU 显存占用（调试用，依赖 pynvml）。
    # gpu_id 从 device 字符串末位解析（如 cuda:0 -> 0），多卡时需与 CUDA_VISIBLE_DEVICES 一致。
    import pynvml
    pynvml.nvmlInit()
    gpu_id = int(str(gpu_id)[-1])
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        logging.info('gpu_id %s does not exsit!', gpu_id)
        return 0, 0, 0
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    logging.info("total %sMB, used %sMB, free %sMB", total, used, free)
    return total, used, free


class Client:
    # 单客户端：维护本地 PerLabelDatasetNonIID、可学习合成张量 synthetic_images；
    # 每联邦轮由 Server 调用 train_swd_condense，返回 detach 后的合成图及各类特征/logit 原型供聚合。

    def __init__(
        self,
        cid: int,
        train_set: PerLabelDatasetNonIID,
        classes: list[int],
        dataset_info: dict,
        ipc: int,
        compression_ratio: float,
        dc_iterations: int,
        real_batch_size: int,
        image_lr: float,
        image_momentum: float,
        image_weight_decay: float,
        dsa: bool,
        dsa_strategy: str,
        init: str,
        clip_norm: float,
        gamma: float,
        logit_lambda: float,
        b: float,
        save_root_path: str,
        device: torch.device,
    ):
        # ----- 身份与数据 -----
        self.cid = cid  # 客户端编号，与划分 JSON 中顺序一致
        self.train_set = train_set  # 已按 Non-IID 子集构造，仅含本客户端出现的类别
        self.classes = classes  # 该客户端拥有的类别列表（可能远少于全局 C）
        self.dataset_info = dataset_info  # channel, im_size, num_classes 等

        # ----- 浓缩超参（与 config.py / main.py 传入一致）-----
        self.ipc = ipc  # compression_ratio=0 时每类合成条数常数
        self.compression_ratio = compression_ratio  # >0 时按真实量×ratio 上取整得每类合成数，且每类至少 5 张
        self.dc_iterations = dc_iterations  # 外层循环次数实为 dc_iterations+1（与 pre_sample 长度对齐）
        self.real_batch_size = real_batch_size  # 每步从每类真实池中抽的批大小（对应 dc_batch_size）
        self.image_lr = image_lr
        self.image_momentum = image_momentum
        self.image_weight_decay = image_weight_decay

        # round：第几次进入浓缩；-1 表示尚未开始，receive_model 后仍为 -1 直至第一次 train_swd_condense
        self.round = -1
        self.dsa = dsa  # 是否对真图、合成图同步做可微增广 DiffAugment
        self.dsa_strategy = dsa_strategy  # 增广算子名字符串，见 utils.ParamDiffAug
        self.model_name = None  # 骨干名，首次 receive_model 时写入
        self.global_model = None  # 当前轮下发的全局模型
        self.prev_global_model = None  # 上一轮（或初始）全局模型，供 cal_loss 里双模型 logits 融合
        self.dsa_param = ParamDiffAug()  # 可微增广的随机参数字典（与 Server 侧各自一份）
        self.init = init  # 合成图初始化方式，当前仅支持 'real'
        self.clip_norm = clip_norm  # 仅裁剪 synthetic_images 的梯度范数
        self.gamma = gamma  # 传入 cal_loss，与 logit_lambda、b 一起参与融合 logits（见 dataset.py）
        self.logit_lambda = logit_lambda
        self.b = b
        self.save_root_path = save_root_path  # 传入 cal_loss 占位，当前损失计算未使用
        self.device = device

        if len(self.classes) > 0:
            # 每类合成数量 ipc_dict[c]：压缩模式用真实量 * ratio 上取整且下限 5；否则每类固定 ipc
            if self.compression_ratio > 0.0:
                self.ipc_dict = {
                    c: max(5, int(math.ceil(len(self.train_set.indices_class[c]) * self.compression_ratio)))
                    for c in self.classes
                }
            else:
                self.ipc_dict = {c: self.ipc for c in self.classes}
            # 所有类合成图拼成一条长向量 synthetic_images，accumulate_num_syn_imgs 为类边界下标
            num_synthetic_images = sum(self.ipc_dict.values())
            self.accumulate_num_syn_imgs = [0]
            for i, c in enumerate(self.classes):
                self.accumulate_num_syn_imgs.append(self.accumulate_num_syn_imgs[-1] + self.ipc_dict[c])
            # 先用高斯随机初始化，initialization() 里若 init=='real' 会覆盖为真实子样本
            self.synthetic_images = torch.randn(
                size=(
                    num_synthetic_images,
                    dataset_info['channel'],
                    dataset_info['im_size'][0],
                    dataset_info['im_size'][1],
                ),
                dtype=torch.float,
                requires_grad=True,
                device=self.device,
            )

    def train_swd_condense(self):
        # 客户端 SWD 浓缩主循环（由 Server.fit 每轮统一调用）。
        # 流程概要：
        # 1) round 自增；init=='real' 时用真实图覆盖合成初值（initialization）。
        # 2) 第 0 轮在客户端内 get_model 另建一份 global_model（与 receive_model 权重或略有差异，见论文）。
        # 3) get_feature_prototype / get_logit_prototype：各类真实样本在 embed / logits 上的类均值（供服务端）。
        # 4) cal_loss：双模型融合 logits → 熵 → 每样本权重；pre_sample：按权重为每类生成整段浓缩用下标。
        # 5) 冻结 global_model，仅优化 synthetic_images；每步 SWD(embed(真), embed(合成))，可选 DiffAugment。
        self.round += 1
        self.initialization()
        # 优化器只挂在 synthetic_images 上：浓缩目标不学全局 θ，只学合成像素
        optimizer_image = torch.optim.SGD(
            [self.synthetic_images], lr=self.image_lr, momentum=self.image_momentum, weight_decay=self.image_weight_decay
        )
        optimizer_image.zero_grad()
        logging.info("client %s have real samples %s", self.cid, [len(self.train_set.indices_class[c]) for c in self.classes])
        logging.info("client %s will condense %s samples for each class it owns", self.cid, self.ipc_dict)

        # 首轮：客户端内部重新实例化一份与 model_name 一致的骨干（论文中注明与 receive_model 的细微差异）
        if self.round == 0:
            self.global_model = get_model(self.model_name, self.dataset_info).to(self.device)
        # 基于「当前」global_model 在全体本地真实图上的类均值特征 / logits（大块数据时分 batch 累加）
        prototypes = self.get_feature_prototype()
        logit_prototypes = self.get_logit_prototype()

        # 用双模型融合 logits 算归一化熵，得到 loss_all；再 pre_sample 得到各类 sample_indices
        self.train_set.cal_loss(
            copy.deepcopy(self.global_model),
            copy.deepcopy(self.prev_global_model),
            logit_lambda=self.logit_lambda,
            gamma=self.gamma,
            b=self.b,
            rounds=self.round,
            cid=self.cid,
            save_root_path=self.save_root_path,
        )
        self.train_set.pre_sample(it=self.dc_iterations + 1, bs=self.real_batch_size)

        total_loss = 0.0
        swd_criterion = SWDLoss(device=self.device)
        self.global_model.train()
        for param in list(self.global_model.parameters()):
            param.requires_grad = False

        # 外层 dc_iterations+1 步：与 pre_sample 为每类预采样的长度一致
        for dc_iteration in range(self.dc_iterations + 1):
            loss = torch.tensor(0.0, device=self.device)
            images_real_all = []
            images_syn_all = []
            num_real_image = [0]  # 前缀和：cat 后按类切片 real_feature 时用
            for i, c in enumerate(self.classes):
                # 从该类预采样序列中截取当前步的 real_batch_size 张真实图索引
                real_image = self.train_set.images_all[
                    self.train_set.sample_indices[c][dc_iteration : dc_iteration + self.real_batch_size]
                ]
                num_real_image.append(num_real_image[-1] + real_image.shape[0])
                # 取出该类对应的合成图块并 reshape 为 (ipc_dict[c], C, H, W)
                synthetic_image = self.synthetic_images[
                    self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i + 1]
                ].reshape(
                    (
                        self.ipc_dict[c],
                        self.dataset_info['channel'],
                        self.dataset_info['im_size'][0],
                        self.dataset_info['im_size'][1],
                    )
                )
                if self.dsa:
                    # 同一 seed 对 real / syn 同步增广，保持两路可微对齐的随机性一致
                    seed = int(time.time() * 1000) % 100000
                    real_image = DiffAugment(real_image, self.dsa_strategy, seed=seed, param=self.dsa_param)
                    synthetic_image = DiffAugment(synthetic_image, self.dsa_strategy, seed=seed, param=self.dsa_param)
                images_real_all.append(real_image)
                images_syn_all.append(synthetic_image)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)
            self.global_model.train()
            # 真实分支不反传（detach）；合成分支需要 embed 的梯度回传到 synthetic_images
            real_feature = self.global_model.embed(images_real_all).detach()
            self.global_model.eval()
            synthetic_feature = self.global_model.embed(images_syn_all)

            # 按类在特征维上分别算 SWD，再求和为一步的 loss
            for i, c in enumerate(self.classes):
                loss += swd_criterion(
                    real_feature[num_real_image[i] : num_real_image[i + 1]],
                    synthetic_feature[self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i + 1]],
                )

            total_loss += loss.item()
            optimizer_image.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([self.synthetic_images], max_norm=self.clip_norm)
            optimizer_image.step()

            if dc_iteration % 200 == 0 or dc_iteration == self.dc_iterations:
                logging.info(
                    "client %s, data condensation %s, total loss = %s, avg loss = %s",
                    self.cid,
                    dc_iteration,
                    loss.item(),
                    loss.item() / len(self.classes),
                )

        synthetic_labels = torch.cat([torch.ones(self.ipc_dict[c]) * c for c in self.classes])
        return (
            copy.deepcopy(self.synthetic_images.detach()),
            copy.deepcopy(synthetic_labels),
            total_loss / (len(self.classes) * self.dc_iterations),
            self.ipc_dict,
            self.accumulate_num_syn_imgs,
            prototypes,
            logit_prototypes,
        )

    def get_feature_prototype(self):
        # 每类 c：在 embed 空间对该类全部真实样本求均值（及样本数 tot_num_c）。
        # 样本数 >500 时分 batch 前向再累加，避免显存峰值。
        # 返回 prototypes[c] = (mean_feature_vector, count)，供服务端按样本数加权平均。
        logging.info("get_feature_prototype")
        prototypes = {c: None for c in self.classes}
        self.global_model.eval()
        for param in list(self.global_model.parameters()):
            param.requires_grad = False
        for c in self.classes:
            tot_num_c = len(self.train_set.indices_class[c])
            if tot_num_c > 500:
                real_feature_c = []
                for it in range(0, tot_num_c, 500):
                    if it + 500 >= tot_num_c:
                        real_feature_c_batch = self.global_model.embed(
                            self.train_set.images_all[self.train_set.indices_class[c][it:tot_num_c]]
                        ).detach()
                    else:
                        real_feature_c_batch = self.global_model.embed(
                            self.train_set.images_all[self.train_set.indices_class[c][it : it + 500]]
                        ).detach()
                    real_feature_c.append(torch.sum(real_feature_c_batch, dim=0))
                real_feature_c = torch.vstack(real_feature_c)
                real_feature_c = torch.sum(real_feature_c, dim=0) / tot_num_c
                prototypes[c] = (real_feature_c, tot_num_c)
                del real_feature_c
            else:
                real_images_c = self.train_set.get_all_images(c)
                real_feature_c = self.global_model.embed(real_images_c)
                prototypes[c] = (torch.mean(real_feature_c, dim=0), tot_num_c)
                del real_feature_c, real_images_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return prototypes

    def get_logit_prototype(self):
        # 与 get_feature_prototype 类似，但在完整 logits（forward 的 C 维输出）上按类求均值。
        # 服务端用于 relation_class、logit 原型聚合等；浓缩循环本身不直接使用本返回值。
        logging.info("get_logit_prototype")
        prototypes = {c: None for c in self.classes}
        self.global_model.eval()
        for param in list(self.global_model.parameters()):
            param.requires_grad = False
        for c in self.classes:
            tot_num_c = len(self.train_set.indices_class[c])
            if tot_num_c > 500:
                real_logit_c = []
                for it in range(0, tot_num_c, 500):
                    if it + 500 >= tot_num_c:
                        real_logit_c_batch = self.global_model(
                            self.train_set.images_all[self.train_set.indices_class[c][it:tot_num_c]]
                        ).detach()
                    else:
                        real_logit_c_batch = self.global_model(
                            self.train_set.images_all[self.train_set.indices_class[c][it : it + 500]]
                        ).detach()
                    real_logit_c.append(torch.sum(real_logit_c_batch, dim=0))
                real_logit_c = torch.vstack(real_logit_c)
                real_logit_c = torch.sum(real_logit_c, dim=0) / tot_num_c
                prototypes[c] = (real_logit_c, tot_num_c)
                del real_logit_c
            else:
                real_images_c = self.train_set.get_all_images(c)
                real_logit_c = self.global_model(real_images_c)
                prototypes[c] = (torch.mean(real_logit_c, dim=0), tot_num_c)
                del real_logit_c, real_images_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return prototypes

    def receive_model(self, model_name, global_model=None):
        # 每联邦轮开始由 Server 调用：记录骨干名，深拷贝全局权重。
        # round==-1（首次）：prev_global 与当前都设为本次下发模型。
        # 否则：prev_global <- 上轮留在本地的 global_model，再写入本轮新 global_model。
        self.model_name = model_name
        if global_model is not None:
            if self.round == -1:
                self.prev_global_model = copy.deepcopy(global_model)
            else:
                self.prev_global_model = copy.deepcopy(self.global_model)
            self.global_model = copy.deepcopy(global_model)
            self.global_model.eval()

    def initialization(self):
        # 从每类真实数据中随机抽 ipc_dict[c] 张，像素级拷贝到 synthetic_images 对应段，作为浓缩初值
        if self.init != 'real':
            raise ValueError("仅支持 init=real（与 run.sh 一致）")
        logging.info("initialized by real images")
        for i, c in enumerate(self.classes):
            self.synthetic_images.data[
                self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i + 1]
            ] = self.train_set.get_images(c, self.ipc_dict[c]).detach().data
