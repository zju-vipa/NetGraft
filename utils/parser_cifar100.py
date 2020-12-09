import argparse
from utils.nn_tool import fix_seed


def parse(num_per_class=10):
    args = argparse.Namespace()

    args.dataset = 'CIFAR100'
    args.num_class = 100
    args.data_path = 'data/'
    
    args.num_per_class = num_per_class
    factor = args.num_per_class / 10

    args.batch_size = int(64*factor)

    args.ckpt = 'ckpt/teacher/vgg16-blockwise-cifar{}.pth'\
        .format(args.num_class)

    args.norm_loss = True

    args.lrs_s = [0.00025, 0.001, 0.001] 
    args.lrs_adapt_t2s = [0.001, 0.001]
    args.lrs_adapt_s2t = [0.001, 0.001]

    args.lrs_s = [item*factor for item in args.lrs_s]
    args.lrs_adapt_t2s = [item*factor for item in args.lrs_adapt_t2s]
    args.lrs_adapt_s2t = [item*factor for item in args.lrs_adapt_s2t]

    args.num_epoch = [1000, 200, 200]

    return args

def parse_net(num_per_class=10):
    args = argparse.Namespace()

    args.dataset = 'CIFAR100'
    args.num_class = 100
    args.data_path = 'data/'
    
    args.num_per_class = num_per_class
    factor = args.num_per_class / 10

    args.batch_size = int(64*factor)

    args.ckpt = 'ckpt/teacher/vgg16-blockwise-cifar{}.pth'\
        .format(args.num_class)

    args.norm_loss = True

    args.lrs_s = [0.00005, 0.00005] 
    args.lrs_adapt_t2s = [0.001, 0.001]
    args.lrs_adapt_s2t = [0.001, 0.001]

    args.lrs_s = [item*factor for item in args.lrs_s]
    args.lrs_adapt_t2s = [item*factor for item in args.lrs_adapt_t2s]
    args.lrs_adapt_s2t = [item*factor for item in args.lrs_adapt_s2t]

    args.num_epoch = [150, 150]

    return args
