import torch
import numpy as np


def decode(sequence, characters):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j + 1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


def decode_target(sequence, characters):
    return ''.join([characters[x] for x in sequence]).replace('-', '')


def calc_acc(target, output, characters):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true, characters) == decode(pred, characters)
                  for true, pred in zip(target, output_argmax)])
    return torch.tensor(a.mean())
