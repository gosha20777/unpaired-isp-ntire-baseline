import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import preprocess_img_disc

def compute_ms_ssim_loss(generated, target, data_range=1.0, size_average=True, channels=1):
    generated, target = preprocess_img_disc(target, generated, channels)
    ms_ssim_value = ms_ssim(generated, target, data_range=data_range, size_average=size_average)
    return 1 - ms_ssim_value


