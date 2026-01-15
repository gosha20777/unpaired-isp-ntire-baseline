import torch
import pyiqa
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dists_loss = pyiqa.create_metric('dists', device=device, as_loss=True)


def compute_dists_loss(generated, target_dslr):

    generated = torch.clamp(generated, 0, 1)
    target_dslr = torch.clamp(target_dslr, 0, 1)

    loss_dists = dists_loss(target_dslr, generated)

    return loss_dists
