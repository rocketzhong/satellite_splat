# -----------------------------------------------------------------------------
# This file includes code adapted from the following paper:
#
#   Junha Hyung, Susung Hong, Sungwon Hwang, Jaeseong Lee,
#   Jin-Hwa Kim, and Jaegul Choo.
#   "Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting."
#   In: Conference on Neural Information Processing Systems (NeurIPS), 2024.
#   https://github.com/junhahyung/erank_gs
#
# Parts of this implementation are adapted or modified from the above work.
# -----------------------------------------------------------------------------

import torch


def get_effective_rank(scale, temp=1):
    D = (scale * scale) ** (1 / temp)
    _sum = D.sum(dim=1, keepdim=True)
    pD = D / _sum
    try:
        entropy = -torch.sum(pD * torch.log(pD), dim=1)
        erank = torch.exp(entropy)
    except Exception as e:
        print(e)
        pass
    return erank


def get_ordered_scale_multiple(scale):
    ordered_scale, _ = torch.sort(scale, descending=True)
    ordered_scale_multiple = ordered_scale / ordered_scale[:, 2:3]
    return ordered_scale_multiple, ordered_scale
