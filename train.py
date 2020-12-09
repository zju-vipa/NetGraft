import os
import argparse

from utils.logger import Logger

from graft_block import run_graft_block
from graft_net import run_graft_net


def run_various_num_samples(dataset, idx):
    os.makedirs('./log/', exist_ok=True)

    nums = list(range(1,11))
    nums += [20, 50]

    logger_acc = Logger('log/accuracy-for-various-num_sample-{}.txt'.format(idx))

    for i in nums:
        logger_acc.write('Num-of-Samples', i)
        run_graft_block(dataset, i)
        acc = run_graft_net(dataset, i)
        logger_acc.write('Accuracy', acc)
        logger_acc.flush()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='dataset name (default: CIFAR10)')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    args = parse()
    for iteration in range(1, 6): # run five times
        run_various_num_samples(args.dataset, iteration)