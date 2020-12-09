import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from network.vgg_block import vgg_stock, vgg_bw, cfgs, split_block
from utils.data import get_dataset, get_transformer
from utils.metric import test
from utils.logger import Logger
from utils.nn_tool import init_conv

from dataset import CIFAR10Few, CIFAR100Few

logger = None

block_graft_ids = [3, 4, 5]
blocks_s_len = [1, 1, 1]


def train_epoch(args, teacher, blocks_s, blocks_s_len, adaptions,
                block_id, train_loader, optimizers):

    for block in blocks_s:
        block.train()

    if block_id < 0 and block_id >= len(blocks_s):
        RuntimeError("block_id out of range: {}".format(block_id))

    adaptions_t2s, adaptions_s2t = adaptions
    optimizers_s, optimizers_adapt_t2s, optimizers_adapt_s2t = optimizers

    optimizer_s = optimizers_s[block_id]

    block_s = blocks_s[block_id]
    block_s_len = blocks_s_len[block_id]

    if block_id > 0 and block_id < len(blocks_s) - 1:
        adaption_t2s = adaptions_t2s[block_id-1]
        adaption_s2t = adaptions_s2t[block_id]

        optimizer_adapt_t2s = optimizers_adapt_t2s[block_id-1]
        optimizer_adapt_s2t = optimizers_adapt_s2t[block_id]

        block_s = nn.Sequential(adaption_t2s, block_s, adaption_s2t)
    elif block_id == 0:
        adaption_s2t = adaptions_s2t[block_id]

        optimizer_adapt_s2t = optimizers_adapt_s2t[block_id]

        block_s = nn.Sequential(block_s, adaption_s2t)
    elif block_id == len(blocks_s) - 1:
        adaption_t2s = adaptions_t2s[block_id-1]

        optimizer_adapt_t2s = optimizers_adapt_t2s[block_id-1]
        block_s = nn.Sequential(adaption_t2s, block_s)

    loss_value = 0.0

    for i, data in enumerate(train_loader):
        data = data.cuda()
        teacher.reset_scion()

        logits_t = teacher(data).detach()

        teacher.set_scion(block_s, block_graft_ids[block_id], block_s_len)

        logits_s = teacher(data)

        if args.norm_loss:
            loss = F.mse_loss(F.normalize(logits_s), F.normalize(logits_t), reduction='sum')
        else:
            loss = F.mse_loss(logits_s, logits_t, reduction='mean')

        if block_id > 0 and block_id < len(blocks_s) - 1:
            optimizer_s.zero_grad()
            optimizer_adapt_s2t.zero_grad()
            optimizer_adapt_t2s.zero_grad()

            loss.backward()

            optimizer_s.step()
            optimizer_adapt_s2t.step()
            optimizer_adapt_t2s.step()
        elif block_id == 0:
            optimizer_s.zero_grad()
            optimizer_adapt_s2t.zero_grad()

            loss.backward()

            optimizer_s.step()
            optimizer_adapt_s2t.step()
        elif block_id == len(blocks_s) - 1:
            optimizer_s.zero_grad()
            optimizer_adapt_t2s.zero_grad()

            loss.backward()

            optimizer_s.step()
            optimizer_adapt_t2s.step()

        loss_value += loss.item()
        if logger:
            logger.write('Loss-B{}'.format(block_id), loss.item())

    loss_value /= len(train_loader.dataset)

    return loss_value


def graft_block(args):
    os.makedirs('log', exist_ok=True)
    global logger
    logger = Logger('log/graft_block_{}_{}_num_per_class_{}.txt'.\
                format(args.dataset, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()), 
                       args.num_per_class))

    cfg_t = cfgs['vgg16']
    cfg_s = cfgs['vgg16-graft']

    cfg_blocks_t = split_block(cfg_t)
    cfg_blocks_s = split_block(cfg_s)

    num_block = len(block_graft_ids)
    # ---------------------- Adaption ----------------------
    adaptions_t2s = [nn.Conv2d(cfg_blocks_t[block_graft_ids[i]][-2],
                               cfg_blocks_s[block_graft_ids[i]][-2],
                               kernel_size=1).cuda()
                     for i in range(0, num_block - 1)]

    for m in adaptions_t2s:
        init_conv(m)

    adaptions_s2t = [nn.Conv2d(cfg_blocks_s[block_graft_ids[i]][-2],
                               cfg_blocks_t[block_graft_ids[i]][-2],
                               kernel_size=1).cuda()
                     for i in range(0, num_block - 1)]

    for m in adaptions_s2t:
        init_conv(m)
        
    # ---------------------- Network ----------------------
    teacher = vgg_stock(cfg_t, args.dataset, args.num_class)
    student = vgg_bw(cfg_s, True, args.dataset, args.num_class)

    params_t = torch.load(args.ckpt)

    teacher.cuda().eval()
    teacher.load_state_dict(params_t)

    params_s = {}
    for key in params_t.keys():
        key_split = key.split('.')
        if key_split[0] == 'features' and \
                key_split[1] in ['0', '1', '2']:
            params_s[key] = params_t[key]

    student.cuda().train()
    student.load_state_dict(params_s, strict=False)

    blocks_s = [student.features[i] for i in block_graft_ids[:-1]]
    blocks_s += [nn.Sequential(nn.Flatten().cuda(), student.classifier)]
    
    # ---------------------- Optimizer ----------------------
    optimizers_s = [optim.Adam(blocks_s[i].parameters(), lr=args.lrs_s[i])
                    for i in range(0, num_block)]

    optimizers_adapt_t2s = [optim.Adam(adaptions_t2s[i].parameters(),
                                       lr=args.lrs_adapt_t2s[i])
                            for i in range(0, num_block - 1)]

    optimizers_adapt_s2t = [optim.Adam(adaptions_s2t[i].parameters(),
                                       lr=args.lrs_adapt_s2t[i])
                            for i in range(0, num_block - 1)]
    # ---------------------- Datasets ----------------------
    if args.dataset == 'CIFAR10':
        train_loader = DataLoader(
        CIFAR10Few(args.data_path, args.num_per_class,
                   transform=get_transformer(args.dataset,
                                             cropsize=32, crop_padding=4,
                                             hflip=True)),
        batch_size=args.batch_size, num_workers=4, shuffle=True)
    elif args.dataset == 'CIFAR100':
        train_loader = DataLoader(
        CIFAR100Few(args.data_path, args.num_per_class,
                    transform=get_transformer(args.dataset,
                                              cropsize=32, crop_padding=4,
                                              hflip=True)),
        batch_size=args.batch_size, num_workers=4, shuffle=True)

    test_loader = DataLoader(get_dataset(args, train_flag=False),
                             batch_size=256, num_workers=4, shuffle=False)

    # ---------------------- Training ----------------------
    os.makedirs('./ckpt/student', exist_ok=True)
    params_s_best = OrderedDict()

    for block_id in range(len(blocks_s)):
        best_accuarcy = 0.0
        for epoch in range(args.num_epoch[block_id]):
            if logger: logger.write('Epoch', epoch)
            loss_value = train_epoch(args, teacher, blocks_s, blocks_s_len,
                                     [adaptions_t2s, adaptions_s2t],
                                     block_id, train_loader,
                                     [optimizers_s, optimizers_adapt_t2s, optimizers_adapt_s2t])

            accuracy = test(teacher, test_loader)

            if best_accuarcy < accuracy:
                best_accuarcy = accuracy
            
            if epoch == args.num_epoch[block_id] - 1:
                block_warp = warp_block(blocks_s, block_id, adaptions_t2s, adaptions_s2t)
                params_s_best['block-{}'.format(block_id)] \
                    = block_warp.cpu().state_dict().copy()  # deep copy !!!

            if logger:
                logger.write('Accuracy-B{}'.format(block_id), accuracy)

    for block_id in range(len(blocks_s)):
        block = warp_block(blocks_s, block_id, adaptions_t2s, adaptions_s2t)
        block.load_state_dict(params_s_best['block-{}'.format(block_id)])
        block.cuda()
        teacher.set_scion(block, block_graft_ids[block_id], 1)
        accuracy = test(teacher, test_loader)
        if logger:
            logger.write('Test-Best-Accuracy-B{}'.format(block_id), accuracy)

    if logger:
        logger.close()
    with open('ckpt/student/vgg16-student-graft-block-{}-{}perclass.pth'.\
              format(args.dataset, args.num_per_class), 'bw') as f:
        torch.save(params_s_best, f)


def warp_block(blocks_s, block_id, adaptions_t2s, adaptions_s2t):
    block_s = blocks_s[block_id]

    if block_id > 0 and block_id < len(blocks_s) - 1:
        adaption_t2s = adaptions_t2s[block_id-1]
        adaption_s2t = adaptions_s2t[block_id]

        block_s = nn.Sequential(adaption_t2s, block_s, adaption_s2t)
    elif block_id == 0:
        adaption_s2t = adaptions_s2t[block_id]

        block_s = nn.Sequential(block_s, adaption_s2t)
    elif block_id == len(blocks_s) - 1:
        adaption_t2s = adaptions_t2s[block_id-1]

        block_s = nn.Sequential(adaption_t2s, block_s)
    return block_s


def run_graft_block(dataset, num_per_class):
    if dataset.lower() == 'cifar10':
        from utils.parser_cifar10 import parse
    elif dataset.lower() == 'cifar100':
        from utils.parser_cifar100 import parse

    args = parse(num_per_class)
    graft_block(args)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_per_class = 5
    run_graft_block('CIFAR100', num_per_class)


