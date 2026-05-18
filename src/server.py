# -*- coding: utf-8 -*-
# 联邦学习服务端：编排多轮通信、聚合各客户端浓缩结果、在「历史合成缓存」上训练全局模型，
# 并在 con_beta>0 且轮次>0 时叠加基于 prev_syn_proto 的非对称监督对比（见 SupervisedContrastiveLoss）。
# 与 main.py 构造参数一致；客户端固定经 train_swd_condense 做 SWD 嵌入空间浓缩。
import logging
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.client import Client
from src.utils import DiffAugment, ParamDiffAug, SupervisedContrastiveLoss


def get_gpu_mem_info(gpu_id=0):
    # 查询 GPU 显存（pynvml）；gpu_id 取 device 字符串最后一位，与 client 模块中实现一致。
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


class Server:
    # 维护全局模型、测试集 DataLoader、以及跨轮状态（如 prev_syn_proto、多轮合成缓存）。

    def __init__(
        self,
        train_set,
        ipc,
        dataset_info,
        global_model_name: str,
        global_model: nn.Module,
        clients: list[Client],
        communication_rounds: int,
        join_ratio: float,
        batch_size: int,
        model_epochs: int,
        lr_server: float,
        momentum_server: float,
        weight_decay_server: float,
        lr_head: float,
        weight_decay_head: float,
        con_beta: float,
        con_temp: float,
        topk: int,
        dsa: bool,
        dsa_strategy: str,
        preserve_all: bool,
        eval_gap: int,
        test_loader: DataLoader,
        device: torch.device,
        model_identification: str,
        save_root_path: str,
    ):
        self.train_set = train_set  # 服务端持有的「全类」PerLabelDataset，部分逻辑/占位用
        self.ipc = ipc  # 与客户端一致传入，本类 fit 内主要用 dataset_info
        self.dataset_info = dataset_info
        self.global_model_name = global_model_name
        self.global_model = global_model.to(device)
        self.clients = clients
        self.communication_rounds = communication_rounds  # 联邦通信轮数 T
        self.join_ratio = join_ratio  # 1.0 表示每轮全体客户端参与浓缩
        self.batch_size = batch_size  # 服务端在合成 TensorDataset 上的批大小
        self.model_epochs = model_epochs  # 外层训练循环为 range(model_epochs+1)，即默认 1001 次
        self.lr_server = lr_server
        self.momentum_server = momentum_server
        self.weight_decay_server = weight_decay_server
        self.lr_head = lr_head  # 对比支路 Projector（在 SupervisedContrastiveLoss 内）的 Adam 学习率
        self.weight_decay_head = weight_decay_head
        self.con_beta = con_beta  # 对比项权重；在 epoch == model_epochs//2 处乘 0.1，且跨后续通信轮不重置
        self.con_temp = con_temp  # 对比温度
        self.topk = topk  # 由各类 logit 原型构造 relation_class（每行 top-k 列索引），供日志与自检
        self.dsa = dsa
        self.dsa_strategy = dsa_strategy
        self.dsa_param = ParamDiffAug()  # 服务端对合成 batch 的可微增广参数（与 Client 侧独立）
        self.preserve_all = preserve_all  # True 则永不丢弃 all_synthetic_data 前端缓存
        self.eval_gap = eval_gap  # 每 eval_gap 轮在轮末追加一次 evaluate 并记录曲线
        self.test_loader = test_loader
        self.device = device
        self.model_identification = model_identification  # 结果目录名片段，日志外可用
        self.save_root_path = save_root_path

    def fit(self):
        # ---------- 跨通信轮累积量 ----------
        evaluate_acc = 0.0
        round_list = []
        evaluate_acc_list = []
        img_syn_loss = {idx: [] for idx in range(len(self.clients))}  # 各客户端浓缩平均损失历史
        all_synthetic_data = []  # 每轮拼接后的「全类」合成大图列表（可能滑动截断）
        all_synthetic_label = []
        # 按类 c 保存「每一轮、各客户端贡献」的合成张量列表，用于轮末写 prev_syn_proto（历轮纵向拼接再 embed）
        all_syn_imgs_c = {c: [] for c in range(0, self.dataset_info['num_classes'])}
        prev_syn_proto = None  # 形状 (C, z_dim)，供下一轮对比项作为 positive 原型；第 0 轮末写入、第 1 轮起使用

        for rounds in range(self.communication_rounds):
            logging.info(' ====== round %s ======', rounds)
            start_time = time.time()
            logging.info('---------- client training ----------')

            selected_clients = self.select_clients()
            selected_clients_id = [c.cid for c in selected_clients]
            logging.info('selected clients: %s', selected_clients_id)

            # ----- 本轮聚合用临时容器（全局 C 类）-----
            server_prototypes = {c: 0 for c in range(0, self.dataset_info['num_classes'])}
            server_proto_tensor = []
            server_logit_prototypes = {c: 0 for c in range(0, self.dataset_info['num_classes'])}
            server_logit_proto_tensor = []
            num_samples = {c: 0 for c in range(0, self.dataset_info['num_classes'])}
            syn_imgs_all = {c: [] for c in range(0, self.dataset_info['num_classes'])}  # 每类：各客户端 tensor 列表
            syn_imgs_num_cur = {c: 0 for c in range(0, self.dataset_info['num_classes'])}  # 本轮每类收到的合成条数

            # -----（1）客户端浓缩与上行：合成图 + 特征/logit 原型 -----
            for client in selected_clients:
                print(f"Round {rounds}, client {client.cid} start training...")
                get_gpu_mem_info(self.device)
                client.receive_model(self.global_model_name, self.global_model)
                condense_st_time = time.time()
                imgs, labels, syn_loss, ipc_dict, accumulate_num_syn_imgs, prototypes, logit_prototypes = (
                    client.train_swd_condense()
                )
                condense_ed_time = time.time()
                logging.info(
                    "Round %s, client %s condense time: %s", rounds, client.cid, condense_ed_time - condense_st_time
                )

                img_syn_loss[client.cid].append(syn_loss)
                # 按客户端本地类别顺序，把 imgs 切成每类张量，挂到全局类索引 c 上
                for i, c in enumerate(client.classes):
                    synthetic_image_c = imgs[accumulate_num_syn_imgs[i] : accumulate_num_syn_imgs[i + 1]].reshape(
                        (
                            ipc_dict[c],
                            self.dataset_info['channel'],
                            self.dataset_info['im_size'][0],
                            self.dataset_info['im_size'][1],
                        )
                    )
                    syn_imgs_all[c].append(synthetic_image_c)
                    syn_imgs_num_cur[c] += ipc_dict[c]

                # 原型按客户端样本数加权累加，后面统一除以 num_samples[c]
                for i, c in enumerate(client.classes):
                    logging.info("client %s, class %s have %s samples", client.cid, c, prototypes[c][1])
                    server_prototypes[c] += prototypes[c][0] * prototypes[c][1]
                    num_samples[c] += prototypes[c][1]
                    server_logit_prototypes[c] += logit_prototypes[c][0] * logit_prototypes[c][1]

                print(f"Round {rounds}, client {client.cid} finish training...")
                get_gpu_mem_info(self.device)

            logging.info("server receives %s condensed samples for each class", syn_imgs_num_cur)

            # -----（2）全局 C 类上的原型加权平均 → 矩阵与 relation_class -----
            for c in range(self.dataset_info['num_classes']):
                server_prototypes[c] /= num_samples[c]
                server_logit_prototypes[c] /= num_samples[c]
                server_proto_tensor.append(server_prototypes[c])
                server_logit_proto_tensor.append(server_logit_prototypes[c])

            server_proto_tensor = torch.vstack(server_proto_tensor).to(self.device).detach()
            server_proto_tensor = F.normalize(server_proto_tensor, dim=1)
            server_logit_proto_tensor = torch.vstack(server_logit_proto_tensor).to(self.device).detach()
            logging.info("logit_proto before softmax: %s", server_logit_proto_tensor)
            _, relation_class = self.get_mask(server_logit_proto_tensor, k=self.topk)
            if rounds > 0:
                for c in range(self.dataset_info['num_classes']):
                    if c not in relation_class[c]:
                        logging.info("class %s not in relation_class, manually added", c)
                        relation_class[c][-1] = c

            logging.info("shape of prototypes in tensor: %s", server_proto_tensor.shape)
            logging.info("shape of logit prototypes in tensor: %s", server_logit_proto_tensor.shape)
            logging.info("relation tensor: %s", relation_class)

            # 每类多客户端 tensor 沿 batch 维拼接
            for c in range(0, self.dataset_info['num_classes']):
                syn_imgs_all[c] = torch.vstack(syn_imgs_all[c])

            synthetic_data = []
            synthetic_label = []
            for c in range(0, self.dataset_info['num_classes']):
                all_syn_imgs_c[c].append(syn_imgs_all[c])  # 历轮追加，供文末 prev_syn_proto 统计
                synthetic_data.append(syn_imgs_all[c])
                synthetic_label.append(torch.ones(syn_imgs_all[c].shape[0]) * c)

            synthetic_data = torch.vstack(synthetic_data)
            synthetic_label = torch.cat(synthetic_label, dim=0)

            # -----（3）写入多轮缓存；可选滑动窗口丢弃旧轮 -----
            logging.info('---------- update global model ----------')
            all_synthetic_data.append(synthetic_data)
            all_synthetic_label.append(synthetic_label)
            logging.info(len(synthetic_data))

            preserve_thres = max(10, self.communication_rounds // 2)
            logging.info("preserve threshold: %s", preserve_thres)
            if (not self.preserve_all) and (len(all_synthetic_data) > preserve_thres):
                all_synthetic_data = all_synthetic_data[-preserve_thres:]
                all_synthetic_label = all_synthetic_label[-preserve_thres:]

            logging.info(len(all_synthetic_data))
            # 把缓存中多轮合成纵向拼成一大张训练集（论文中的滑动窗口缓存）
            all_syn_cat = torch.cat(all_synthetic_data, dim=0)
            all_syn_lab_cat = torch.cat(all_synthetic_label, dim=0)
            if self.device.type == 'cuda':
                all_synthetic_data_eval = all_syn_cat.to(self.device, non_blocking=True)
                all_synthetic_label_eval = all_syn_lab_cat.to(self.device, non_blocking=True)
                syn_loader_workers = 0
            else:
                all_synthetic_data_eval = all_syn_cat.cpu()
                all_synthetic_label_eval = all_syn_lab_cat.cpu()
                syn_loader_workers = 2
            synthetic_dataset = TensorDataset(all_synthetic_data_eval, all_synthetic_label_eval)
            logging.info("Round %s: # synthetic sample: %s", rounds, len(synthetic_dataset))
            synthetic_dataloader = DataLoader(
                synthetic_dataset, self.batch_size, shuffle=True, num_workers=syn_loader_workers, pin_memory=False
            )

            # -----（4）服务端优化器：全局骨干 SGD + 对比头 Adam -----
            self.global_model.train()
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=self.lr_server,
                weight_decay=self.weight_decay_server,
                momentum=self.momentum_server,
            )
            model_optimizer.zero_grad()
            lr_schedule = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=(self.model_epochs // 2), gamma=0.1)
            loss_function = torch.nn.CrossEntropyLoss()
            z_dim = server_proto_tensor.shape[1]
            relation_sup_con_loss = SupervisedContrastiveLoss(
                num_classes=self.dataset_info['num_classes'],
                device=self.device,
                temperature=self.con_temp,
                z_dim=z_dim,
            )
            mlp_head_optimizer = torch.optim.Adam(
                relation_sup_con_loss.head.parameters(),
                lr=self.lr_head,
                weight_decay=self.weight_decay_head,
            )
            mlp_head_optimizer.zero_grad()
            head_lr_schedule = torch.optim.lr_scheduler.StepLR(
                mlp_head_optimizer, step_size=(self.model_epochs // 2), gamma=0.1
            )

            print(f"Round {rounds}, global model start training...")
            get_gpu_mem_info(self.device)

            # 本轮服务端训练开始前：先评一次试集（与论文「round t evaluation」首条对应）
            acc, test_loss = self.evaluate()
            logging.info('round %s evaluation: test acc is %.4f, test loss = %.6f', rounds, acc, test_loss)
            self.global_model.train()
            for param in list(self.global_model.parameters()):
                param.requires_grad = True

            # -----（5）在合成集上训练 model_epochs+1 个「epoch」：交叉熵为主，可选非对称对比 -----
            for epoch in range(self.model_epochs + 1):
                total_loss = 0.0
                total_con_loss = 0.0
                total_sample = 0
                for x, target in synthetic_dataloader:
                    n_sample = target.shape[0]
                    x, target = x.to(self.device), target.to(self.device)
                    features = None
                    # 对比项用「未 DSA」的 batch 提一次特征（与 pred 路径可不一致，见下）
                    if self.con_beta > 0.0:
                        features, _ = self.global_model(x, train=True)
                    if self.dsa:
                        x = DiffAugment(x, self.dsa_strategy, param=self.dsa_param)
                    target = target.long()
                    # 分类损失在（可选）增广后的 x 上计算
                    _, pred = self.global_model(x, train=True)
                    loss = loss_function(pred, target)
                    total_loss += loss.item() * n_sample

                    if self.con_beta > 0.0 and rounds > 0 and x.shape[0] > 1 and features is not None:
                        assert prev_syn_proto is not None
                        # 每样本的正样本原型 = 上一通信轮末统计的、该类在历轮合成上的 embed 均值（行归一化后）
                        positive_proto = prev_syn_proto[target, :]
                        loss_con = relation_sup_con_loss(features, target, positive_proto, asymmetric=True)
                        total_con_loss += loss_con.item() * n_sample
                        loss = loss + self.con_beta * loss_con

                    model_optimizer.zero_grad()
                    loss.backward()
                    model_optimizer.step()
                    total_sample += n_sample
                    if self.con_beta > 0.0 and rounds > 0:
                        mlp_head_optimizer.step()

                total_loss /= total_sample
                total_con_loss /= total_sample
                lr_schedule.step()
                if self.con_beta > 0.0 and rounds > 0:
                    head_lr_schedule.step()
                if epoch == (self.model_epochs // 2):
                    logging.info("At epoch %s, decay the con_beta with 0.1 factor", epoch)
                    self.con_beta *= 0.1

                if epoch % 100 == 0 or epoch == self.model_epochs:
                    acc, test_loss = self.evaluate()
                    self.global_model.train()
                    logging.info(
                        "epoch %s, train loss avg now = %.6f, train contrast loss now = %.6f, test acc now = %.4f, test loss now = %.6f",
                        epoch,
                        total_loss,
                        total_con_loss,
                        acc,
                        test_loss,
                    )

            round_time = time.time() - start_time
            logging.info('epoch avg loss = %s, total time = %s', total_loss / self.model_epochs, round_time)

            print(f"Round {rounds}, global model finish training...")
            get_gpu_mem_info(self.device)

            # -----（6）通信轮末：用「自第 0 轮起累积的」各类合成 all_syn_imgs_c 更新 prev_syn_proto，供下一轮对比 -----
            logging.info("Round %s finish, update the prev_syn_proto", rounds)
            prev_syn_proto = torch.zeros_like(server_proto_tensor).to(self.device)
            self.global_model.eval()
            with torch.no_grad():
                for c in range(0, self.dataset_info['num_classes']):
                    all_syn_cat_c = torch.cat(all_syn_imgs_c[c], dim=0)
                    logging.info("%s", all_syn_cat_c.shape)
                    if all_syn_cat_c.shape[0] > 128:
                        for it in range(0, all_syn_cat_c.shape[0], 128):
                            if it + 128 >= all_syn_cat_c.shape[0]:
                                prev_syn_proto[c, :] += torch.sum(
                                    self.global_model.embed(all_syn_cat_c[it:]).detach(), dim=0
                                )
                            else:
                                prev_syn_proto[c, :] += torch.sum(
                                    self.global_model.embed(all_syn_cat_c[it : it + 128]).detach(), dim=0
                                )
                        prev_syn_proto[c, :] /= all_syn_cat_c.shape[0]
                    else:
                        prev_syn_proto[c, :] = torch.mean(self.global_model.embed(all_syn_cat_c).detach(), dim=0)
                prev_syn_proto = F.normalize(prev_syn_proto, dim=1).detach()
                logging.info("shape of prev_syn_proto: %s", prev_syn_proto.shape)

            if rounds % self.eval_gap == 0:
                acc, test_loss = self.evaluate()
                logging.info('round %s evaluation: test acc is %.4f, test loss = %.6f', rounds, acc, test_loss)
                evaluate_acc = acc
                round_list.append(rounds)
                evaluate_acc_list.append(evaluate_acc)

        logging.info("%s", evaluate_acc_list)
        logging.info("%s", img_syn_loss)

    def get_mask(self, matrix, k=3, largest=True):
        # 对 matrix 每一行取 top-k 列索引；返回 one-hot 风格 bool mask 与 min_idx（实为 topk 下标张量）。
        _, min_idx = torch.topk(matrix, k=k, dim=-1, largest=largest)
        mask = torch.zeros_like(matrix)
        rows = torch.arange(min_idx.size(0)).unsqueeze(1)
        mask[rows, min_idx] = 1
        mask = mask.bool()
        return mask, min_idx

    def select_clients(self):
        # join_ratio=1.0 时全选；否则无放回随机抽 floor 比例个客户端。
        return (
            self.clients
            if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def evaluate(self):
        # 在 test_loader 上累计交叉熵与 Top-1；prediction_matrix 记录真实类→预测类的计数，写入日志。
        prediction_matrix = {
            c: {c2: 0 for c2 in range(self.dataset_info['num_classes'])}
            for c in range(self.dataset_info['num_classes'])
        }
        self.global_model.eval()
        with torch.no_grad():
            correct, total, test_loss = 0, 0, 0.0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = self.global_model(x)
                test_loss += F.cross_entropy(pred, target, reduction='sum').item()
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                for i in range(target.shape[0]):
                    prediction_matrix[target[i].item()][pred_label[i].item()] += 1
        logging.info("%s", prediction_matrix)
        return correct / float(total), test_loss / float(total)
