import json
import logging
import os
import sys
import numpy as np
import torch
from torch.utils.data import Subset
from config import parser
from dataset.data.dataset import PerLabelDatasetNonIID, get_dataset
from src.client import Client
from src.server import Server
from src.utils import ParamDiffAug, get_model, setup_seed


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn *= s
        pp += nn
    return pp


def main():
    args = parser.parse_args()  #读取命令行参数
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy == 'None' else True

    if args.partition != 'dirichlet':#只允许 partition=dirichlet 这种数据划分方式，否则直接退出。
        raise SystemExit('仅支持 partition=dirichlet（与 run.sh 一致）')

    split_file = f'/{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
    args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file" + split_file)
    if args.compression_ratio > 0.0:
        model_identification = (
            f'{args.dataset}_alpha{args.alpha}_{args.client_num}clients/'
            f'{args.model}_{100 * args.compression_ratio}%_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        )
    else:
        model_identification = (
            f'{args.dataset}_alpha{args.alpha}_{args.client_num}clients/'
            f'{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        )
#把运行结果（模型、日志文件）存在 results/ 文件夹下，并且如果日志已存在会报错（防止覆盖）。
    args.save_root_path = os.path.join(os.path.dirname(__file__), 'results/', model_identification)
    os.makedirs(args.save_root_path, exist_ok=True)
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    log_path = os.path.join(args.save_root_path, 'log.txt')
    print(log_path)
    if os.path.exists(log_path):
        raise RuntimeError('log file already exists!')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
#设置随机种子，保证每次运行结果一致。
    setup_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
#加载数据集，返回数据集信息、训练集、测试集、测试集加载器。
    dataset_info, train_set, _test_set, test_loader = get_dataset(args.dataset, args.dataset_root, args.batch_size)
    print("load data: done")
    with open(args.split_file, 'r') as file:#读取数据划分文件，返回客户端索引、客户端类别。
        file_data = json.load(file)
    client_indices, client_classes = file_data['client_idx'], file_data['client_classes']
    if len(client_indices) != args.client_num:
        raise ValueError(
            f'划分 JSON 中客户端数 {len(client_indices)} 与 --client_num {args.client_num} 不一致，请核对 split_file'
        )

    if args.dataset == 'CIFAR10':
        labels = np.array(train_set.targets, dtype='int64')
    else:
        labels = train_set.labels
#统计每个客户端的类别分布。
    net_cls_counts = {}
    dict_users = {i: idcs for i, idcs in enumerate(client_indices)}
    for net_i, dataidx in dict_users.items():
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        net_cls_counts[net_i] = {unq[i]: unq_cnt[i] for i in range(len(unq))}

    logging.info('Data statistics: %s', net_cls_counts)
    logging.info('client classes: %s', client_classes)
#创建每个客户端的训练集子集。
    train_sets = [Subset(train_set, indices) for indices in client_indices]
    global_model = get_model(args.model, dataset_info)
    logging.info(global_model)
    logging.info(get_n_params(global_model))
    logging.info(args.__dict__)
#创建每个客户端的模型。
    client_list = [
        Client(
            cid=i,
            train_set=PerLabelDatasetNonIID(train_sets[i], client_classes[i], dataset_info['channel'], device),
            classes=client_classes[i],
            dataset_info=dataset_info,
            ipc=args.ipc,
            compression_ratio=args.compression_ratio,
            dc_iterations=args.dc_iterations,
            real_batch_size=args.dc_batch_size,
            image_lr=args.image_lr,
            image_momentum=args.image_momentum,
            image_weight_decay=args.image_weight_decay,
            dsa=args.dsa,
            dsa_strategy=args.dsa_strategy,
            init=args.init,
            clip_norm=args.clip_norm,
            gamma=args.gamma,
            logit_lambda=args.logit_lambda,
            b=args.b,
            save_root_path=args.save_root_path,
            device=device,
        )
        for i in range(args.client_num)
    ]
#创建服务端。
    server = Server(
        train_set=PerLabelDatasetNonIID(
            train_set,
            range(0, dataset_info['num_classes']),
            dataset_info['channel'],
            device,
        ),
        ipc=args.ipc,
        dataset_info=dataset_info,
        global_model_name=args.model,
        global_model=global_model,
        clients=client_list,
        communication_rounds=args.communication_rounds,
        join_ratio=args.join_ratio,
        batch_size=args.batch_size,
        model_epochs=args.model_epochs,
        lr_server=args.lr_server,
        momentum_server=args.momentum_server,
        weight_decay_server=args.weight_decay_server,
        lr_head=args.lr_head,
        weight_decay_head=args.weight_decay_head,
        con_beta=args.con_beta,
        con_temp=args.con_temp,
        topk=args.topk,
        dsa=args.dsa,
        dsa_strategy=args.dsa_strategy,
        preserve_all=args.preserve_all,
        eval_gap=args.eval_gap,
        test_loader=test_loader,
        device=device,
        model_identification=model_identification,
        save_root_path=args.save_root_path,
    )
    print('Server and Clients have been created.')
    server.fit()#开始训练。


if __name__ == "__main__":
    main()
