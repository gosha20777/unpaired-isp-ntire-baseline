import torch
import pyiqa
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_plus_loss = pyiqa.create_metric('lpips+', device=device, as_loss=True)


def compute_lpips_plus_loss(generated, target_dslr):

    generated = torch.clamp(generated, 0, 1)
    target_dslr = torch.clamp(target_dslr, 0, 1)

    loss_lpips_plus = lpips_plus_loss(target_dslr, generated)

    return loss_lpips_plus
