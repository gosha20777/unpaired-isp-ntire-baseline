import torch
import numpy as np
import random
from scipy.stats import norm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir="runs")


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gauss_kernel(kernlen=21, nsig=3, channels=3):
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def blur(x, kernel):
    kernel = kernel.to(x.device)
    padding = kernel.size(-1) // 2
    return F.conv2d(x, kernel, stride=1, padding=padding, groups=x.size(1))

kernel_blur = gauss_kernel(kernlen=21, nsig=3, channels=3).to(device)


def preprocess_img_disc(real_images, fake_images, channels):
    if channels == 1:
        new_real = 0.299 * real_images[:, 0, :, :] + 0.587 * real_images[:, 1, :, :] + 0.114 * real_images[:, 2, :, :]
        new_fake = 0.299 * fake_images[:, 0, :, :] + 0.587 * fake_images[:, 1, :, :] + 0.114 * fake_images[:, 2, :, :]

        new_real = new_real.unsqueeze(1)
        new_fake = new_fake.unsqueeze(1)
    else:
        new_real = blur(real_images, kernel_blur)
        new_fake = blur(fake_images, kernel_blur)
    return new_real, new_fake


def show_image(img_tensor, title="Image"):
    img = img_tensor.detach().cpu()
    img = torch.clamp(img, 0, 1)

    img_np = (img.numpy() * 255).astype("uint8")

    if img.shape[0] == 3:
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")

    image_pil = Image.fromarray(img_np)
    image_pil.save(f"{title}.jpg")

def save_images(batch_idx, img_list, img_names, batch = False):
    if batch_idx % 100 == 0:
        for _ in range(len(img_list)):
            show_image(img_list[_][0], title=f"images/{batch_idx if batch == True else ''}_{img_names[_]}")