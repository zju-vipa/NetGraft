import argparse

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from network.vgg_block import cfgs, split_block, vgg_stock, vgg_bw
from utils.metric import test
from utils.data import get_dataset
from graft_block import warp_block
from graft_net import block_graft_ids

def test_whole_net(args):
    cfg_t = cfgs['vgg16']
    cfg_s = cfgs['vgg16-graft']

    cfg_blocks_t = split_block(cfg_t)
    cfg_blocks_s = split_block(cfg_s)

    num_block = len(block_graft_ids)

    # ---------------------- Network ----------------------
    teacher = vgg_stock(cfg_t, args.dataset, args.num_class)

    params_t = torch.load(args.ckpt)

    teacher.cuda().eval()
    teacher.load_state_dict(params_t)

    adaptions_t2s = [nn.Conv2d(cfg_blocks_t[block_graft_ids[i]][-2],
                               cfg_blocks_s[block_graft_ids[i]][-2],
                               kernel_size=1).cuda()
                     for i in range(0, num_block - 1)]

    adaptions_s2t = [nn.Conv2d(cfg_blocks_s[block_graft_ids[i]][-2],
                               cfg_blocks_t[block_graft_ids[i]][-2],
                               kernel_size=1).cuda()
                     for i in range(0, num_block - 1)]

    cfg_s = cfgs['vgg16-graft']
    student = vgg_bw(cfg_s, True, args.dataset, args.num_class)
    student.cuda()

    params_s = {}
    for key in params_t.keys():
        key_split = key.split('.')
        if key_split[0] == 'features' and \
                key_split[1] in ['0', '1', '2']:
            params_s[key] = params_t[key]
    
    student.load_state_dict(params_s, strict=False)

    blocks_s = [student.features[i] for i in block_graft_ids[:-1]]
    blocks_s += [nn.Sequential(nn.Flatten().cuda(), student.classifier)]

    blocks = []

    for block_id in range(num_block):
        blocks.append(
            warp_block(blocks_s, block_id, adaptions_t2s, adaptions_s2t).cuda()
        )

    block = nn.Sequential(*blocks)
    block.load_state_dict(
        torch.load('ckpt/student/vgg16-student-graft-net-{}-{}perclass.pth'\
                          .format(args.dataset, args.num_per_class))
    )

    test_loader = DataLoader(get_dataset(args, train_flag=False),
                             batch_size=args.batch_size,
                             num_workers=4, shuffle=False)

    block = nn.Sequential(student.features[:3], block)
    
    print('Test Accuracy: ', test(block, test_loader))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nshot', type=int, default=10, help='number of samples per class (default: 10)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='dataset name (default: CIFAR10)')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    if args.dataset.lower() == 'cifar10':
        from utils.parser_cifar10 import parse_net
    elif args.dataset.lower() == 'cifar100':
        from utils.parser_cifar100 import parse_net
    args = parse_net(args.nshot)
    test_whole_net(args)