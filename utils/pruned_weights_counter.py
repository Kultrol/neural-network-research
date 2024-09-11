import torch


def pruned_weights_counter(tensor):
    print(28 * 28 * 10 - torch.count_nonzero(tensor))
