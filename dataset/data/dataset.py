import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from medmnist import OrganSMNIST, PathMNIST


def get_dataset(dataset, dataset_root, batch_size, pin_memory=None):
    """run.sh 仅用 CIFAR10 / PathMNIST / OrganSMNIST。"""
    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.CIFAR10(dataset_root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(dataset_root, train=False, download=True, transform=transform)
        class_names = trainset.classes
    elif dataset == 'PathMNIST':
        channel = 3
        im_size = (28, 28)
        num_classes = 9
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = PathMNIST(split="train", download=True, root=dataset_root, transform=transform)
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = PathMNIST(split="test", download=True, root=dataset_root, transform=transform)
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    elif dataset == 'OrganSMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 11
        mean = [0.5]
        std = [0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = OrganSMNIST(split="train", download=True, root=dataset_root, transform=transform)
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = OrganSMNIST(split="test", download=True, root=dataset_root, transform=transform)
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    else:
        exit(f'未知数据集: {dataset}（run.sh 仅支持 CIFAR10 / PathMNIST / OrganSMNIST）')

    dataset_info = {
        'name': dataset,
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'classes_names': class_names,
        'mean': mean,
        'std': std,
    }
    _pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=_pin,
    )
    return dataset_info, trainset, testset, testloader


class PerLabelDatasetNonIID:
    def __init__(self, dst_train, classes, channel, device):
        self.device = device
        self.images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        self.labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        self.indices_class = {c: [] for c in classes}
        for i, lab in enumerate(self.labels_all):
            if lab not in classes:
                continue
            self.indices_class[lab].append(i)
        if len(self.images_all) > 0:
            self.images_all = torch.cat(self.images_all, dim=0).to(device)
            self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=device)
            self.loss_all = None
            self.sample_prob = {c: [] for c in classes}
            self.sample_indices = {c: [] for c in classes}

    def __len__(self):
        return self.images_all.shape[0]

    def get_images(self, c, n):
        idx_c = torch.tensor(self.indices_class[c], dtype=torch.long, device=self.device)
        if idx_c.numel() >= n:
            sub = idx_c[torch.randperm(idx_c.numel(), device=self.device)[:n]]
        else:
            sub = idx_c
        return self.images_all[sub]

    def get_all_images(self, c):
        idx = self.indices_class[c]
        if len(idx) == 0:
            return self.images_all[:0]
        ii = torch.tensor(idx, dtype=torch.long, device=self.device)
        return self.images_all[ii]

    def cal_loss(self, model, prev_model, logit_lambda=0.5, gamma=1.0, b=0.7, rounds=None, cid=None, save_root_path=None):
        # gamma / b / rounds / cid / save_root_path 保留以兼容 Client 调用，当前未参与计算
        model.eval()
        prev_model.eval()
        with torch.no_grad():
            if self.images_all.shape[0] > 500:
                all_preds = []
                all_preds_prev = []
                batch_size = 500
                total_num = self.images_all.shape[0]
                for idx in range(0, total_num, batch_size):
                    batch_ed = min(idx + batch_size, total_num)
                    all_preds.append(model(self.images_all[idx:batch_ed]))
                    all_preds_prev.append(prev_model(self.images_all[idx:batch_ed]))
                all_preds = torch.cat(all_preds, dim=0)
                all_preds_prev = torch.cat(all_preds_prev, dim=0)
                all_preds = (1 - logit_lambda) * all_preds + logit_lambda * all_preds_prev
            else:
                all_preds = model(self.images_all)
                all_preds_prev = prev_model(self.images_all)
                all_preds = (1 - logit_lambda) * all_preds + logit_lambda * all_preds_prev
            probs = F.softmax(all_preds, dim=1).clamp(min=1e-12)
            log_k = math.log(max(probs.shape[1], 2))
            H = -(probs * probs.log()).sum(dim=1)
            H_norm = (H / log_k).clamp(0.0, 1.0)
            self.loss_all = (4.0 * H_norm * (1.0 - H_norm)).type(torch.float64)
        del all_preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def pre_sample(self, it, bs):
        for c in self.indices_class.keys():
            idx_c = torch.tensor(self.indices_class[c], dtype=torch.long, device=self.device)
            sp = F.softmax(self.loss_all[idx_c], dim=0)
            self.sample_prob[c] = sp
            pick = torch.multinomial(sp, num_samples=it * bs, replacement=True)
            self.sample_indices[c] = idx_c[pick].cpu().tolist()
            hist, bin_edges = np.histogram(self.sample_prob[c].detach().cpu().numpy(), bins=10)
            logging.info(
                f"class {c} have {len(self.indices_class[c])} samples, histogram: {hist}, bin edged: {bin_edges}"
            )
