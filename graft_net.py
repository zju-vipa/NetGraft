import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from network.vgg_block import vgg_stock, vgg_bw, cfgs, split_block
from utils.data import get_dataset, get_transformer
from utils.metric import test
from utils.logger import Logger

from dataset import CIFAR10Few, CIFAR100Few

from graft_block import warp_block

logger_net = None

block_graft_ids = [3, 4, 5]
blocks_s_len = [1, 1, 1]

def train_epoch(args, teacher, block_scion, scion_len, train_loader, optimizer):

    teacher.eval()
    block_scion.train()

    loss_value = 0.0

    for data in train_loader:
        data = data.cuda()

        teacher.reset_scion()
        logits_t = teacher(data).detach()

        teacher.set_scion(block_scion, block_graft_ids[0], scion_len)

        logits_s = teacher(data)

        if args.norm_loss:
            loss = F.mse_loss(F.normalize(logits_s), F.normalize(logits_t), reduction='sum')
        else:
            loss = F.mse_loss(logits_s, logits_t, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value += loss.item()

        if logger_net:
            logger_net.write('Loss-length-{}'.format(scion_len), loss.item())

    loss_value /= len(train_loader.dataset)

    return loss_value


def graft_net(args):
    global logger_net
    logger_net = Logger('log/graft_net_{}_{}_{}perclass.txt'.\
                    format(args.dataset, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
                           args.num_per_class))
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
                             batch_size=args.batch_size,
                             num_workers=4, shuffle=False)

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

    adaptions_s2t = [nn.Conv2d(cfg_blocks_s[block_graft_ids[i]][-2],
                               cfg_blocks_t[block_graft_ids[i]][-2],
                               kernel_size=1).cuda()
                     for i in range(0, num_block - 1)]

    # ---------------------- Teacher ----------------------
    teacher = vgg_stock(cfg_t, args.dataset, args.num_class)

    params_t = torch.load(args.ckpt)

    teacher.cuda().eval()
    teacher.load_state_dict(params_t)
    
    # ---------------------- Blocks ----------------------
    params_s = {}
    for key in params_t.keys():
        key_split = key.split('.')
        if key_split[0] == 'features' and \
                key_split[1] in ['0', '1', '2']:
            params_s[key] = params_t[key]

    student = vgg_bw(cfg_s, True, args.dataset, args.num_class)
    student.cuda().train()
    student.load_state_dict(params_s, strict=False)

    blocks_s = [student.features[i] for i in block_graft_ids[:-1]]
    blocks_s += [nn.Sequential(nn.Flatten().cuda(), student.classifier)]

    blocks = []

    for block_id in range(num_block):
        blocks.append(
            warp_block(blocks_s, block_id, adaptions_t2s, adaptions_s2t).cuda()
        )

    params = torch.load('ckpt/student/vgg16-student-graft-block-{}-{}perclass.pth'.\
                        format(args.dataset, args.num_per_class))

    for block_id in range(num_block):
        blocks[block_id].load_state_dict(
            params['block-{}'.format(block_id)]
        )

    for i in range(num_block - 1):
        block = nn.Sequential(*blocks[:(i + 2)])
        optimizer = optim.Adam(block.parameters(), lr=0.0001)

        scion_len = sum(blocks_s_len[:(i + 2)])

        accuracy_best_block = 0.0
        params_best_save = None

        for epoch in range(args.num_epoch[i]):
            if logger_net: logger_net.write('Epoch', epoch)
            loss_value = train_epoch(args, teacher, block, scion_len,
                                     train_loader, optimizer)

            accuracy = test(teacher, test_loader)

            if accuracy > accuracy_best_block:
                accuracy_best_block = accuracy
                params_tmp = block.cpu().state_dict()
                params_best_save = params_tmp.copy()
                block.cuda()
                
            if epoch == (args.num_epoch[i] - 1) and \
                i == (num_block - 2):
                block.load_state_dict(params_best_save)

            if logger_net:
                logger_net.write('Accuracy-length-{}'.format(scion_len), accuracy)

    if logger_net:
        logger_net.write('Student Best Accuracy', accuracy_best_block)

    with open('ckpt/student/vgg16-student-graft-net-{}-{}perclass.pth'\
                          .format(args.dataset, args.num_per_class), 'wb') as f:
        torch.save(block.state_dict(), f)
    if logger_net:
        logger_net.close()
    return accuracy_best_block


def run_graft_net(dataset, num_per_class):
    if dataset.lower() == 'cifar10':
        from utils.parser_cifar10 import parse_net
    elif dataset.lower() == 'cifar100':
        from utils.parser_cifar100 import parse_net

    args = parse_net(num_per_class)
    accuracy = graft_net(args)
    return accuracy


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_per_class = 10
    run_graft_net('CIFAR10', num_per_class)





