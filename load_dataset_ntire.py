import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from typing import Literal, Tuple, List, Union
import os
import random
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from config import config

class NtireDataset(Dataset):
    def __init__(self, img_dir1, img_dir2, raw_transform, raw_to_rgb_transform, dslr_transform, shuffled):
        self.img_path1 = sorted([os.path.join(img_dir1, f) for f in os.listdir(img_dir1)])
        self.img_path2 = sorted([os.path.join(img_dir2, f) for f in os.listdir(img_dir2)])
        self.raw_transform = raw_transform
        self.raw_to_rgb_transform = raw_to_rgb_transform
        self.dslr_transform = dslr_transform
        print("Target images are unpaired!")

    def __len__(self):
        return len(self.img_path1)

    def __getitem__(self, idx):
        raw_np = np.load(self.img_path1[idx])
        dslr_np = Image.open(self.img_path2[idx])

        raw_ = self.raw_transform(raw_np)
        custom_rgb_ = self.raw_to_rgb_transform(raw_np)
        dslr_ = self.dslr_transform(dslr_np)
        return raw_, custom_rgb_, dslr_


class RawTransform(nn.Module):
    """
    A custom PyTorch transform to process a 2D raw Bayer image.
    This module performs a 4-channel "packing" of the raw Bayer data,
    then normalizes it using specified black and white levels. It's a
    PyTorch-native reimplementation of the provided NumPy-based logic.
    """
    def __init__(self, white_level=1023.0):
        super().__init__()

        self.white_level = white_level

    def forward(self, raw_image: torch.Tensor) -> torch.Tensor:
        # Assume raw_image is a 3D tensor (H, W, 4)
        return raw_image.permute(2, 0, 1).float() / self.white_level


class RawToRgbTransform(nn.Module):
    """
    A PyTorch module/transform to convert a 16-bit raw Bayer image to a processed
    8-bit sRGB-like image tensor using a traditional ISP pipeline.
    
    The pipeline includes:
    1. Scaling input to [0, 1] (Linearization).
    2. Demosaicing (Bilinear interpolation).
    3. Applying White Balance (Gray World algorithm).
    4. Contrast stretching (Min-Max normalization).
    5. Gamma correction.
    6. Resizing to output dimension.
    """
    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = 512,
        white_level: float = 1023.0,
        gamma: float = 2.2,
    ):
        super().__init__()
        if gamma <= 0:
            raise ValueError("Gamma must be a positive number.")

        self.white_level = white_level
        self.gamma_exponent = 1.0 / gamma
        
        # Image resizer (Antialiased)
        self.resize = v2.Resize(
            output_size, 
            interpolation=v2.InterpolationMode.BILINEAR, 
            antialias=True
        )

    def forward(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the full raw-to-RGB processing pipeline.
        Args:
            raw_tensor (torch.Tensor): A raw image tensor of shape (H/2, W/2, 4).
        Returns:
            torch.Tensor: A processed RGB tensor of shape (3, H_out, W_out).
        """
        # --- 1. Scaling & Permutation ---
        # (H/2, W/2, 4) -> (4, H/2, W/2)
        x = raw_tensor.permute(2, 0, 1).float()
        
        # Scale to [0, 1]
        x = x / self.white_level

        # --- 2. Simple Demosaic (Stacking) ---
        # Вместо интерполяции просто собираем каналы.
        # Мы считаем, что R, G, B пиксели в блоке 2x2 находятся в одной точке.
        # Для бейзлайна это допустимое упрощение.
        
        r = x[0, ...]
        g = (x[1, ...] + x[2, ...]) * 0.5 # Average Green
        b = x[3, ...]
        
        # Получаем тензор (3, H/2, W/2)
        rgb = torch.stack([r, g, b], dim=0)

        # --- 3. White Balance (Gray World) ---
        # Векторизированный подход (быстрее и безопаснее для autograd)
        
        # Считаем среднее по пространственным осям (H, W), оставляя каналы (Dim 0)
        # view(3, -1).mean(1) превращает (3, H, W) в (3) - среднее для каждого канала
        means = rgb.view(3, -1).mean(dim=1)
        
        # Защита от деления на ноль
        means = torch.maximum(means, torch.tensor(1e-6, device=rgb.device))
        
        # means[0]=R, means[1]=G, means[2]=B
        # Считаем гейны относительно зеленого канала
        gain_r = means[1] / means[0]
        gain_b = means[1] / means[2]
        
        # Собираем вектор гейнов: [gain_r, 1.0, gain_b]
        gains = torch.stack([gain_r, torch.tensor(1.0, device=rgb.device), gain_b])
        
        # Применяем гейны через broadcasting: (3, H, W) * (3, 1, 1)
        # view(3, 1, 1) нужен, чтобы умножить каждый канал на свой коэффициент
        rgb_wb = rgb * gains.view(3, 1, 1)
        
        # Clip
        rgb_wb = torch.clamp(rgb_wb, 0.0, 1.0)

        # --- 4. Contrast Stretching ---
        min_val = rgb_wb.min()
        max_val = rgb_wb.max()
        range_val = max_val - min_val

        # Используем where для защиты от деления на 0 без if/else
        scale = torch.where(
            range_val > 1e-6, 
            1.0 / range_val, 
            torch.tensor(1.0, device=rgb.device)
        )
        
        rgb_stretched = (rgb_wb - min_val) * scale

        # --- 5. Gamma Correction ---
        rgb_gamma = torch.pow(rgb_stretched, self.gamma_exponent)
        
        # --- 6. Resize ---
        # v2.Resize сделает финальную интерполяцию
        rgb_out = self.resize(rgb_gamma)

        return rgb_out


raw_transform = v2.Compose([
            torch.from_numpy,
            RawTransform(),
        ])

raw_to_rgb_transform = v2.Compose([
            torch.from_numpy,
            RawToRgbTransform(),
        ])

dslr_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ])

dataset = NtireDataset(config.path_input, config.path_output, raw_transform, raw_to_rgb_transform, dslr_transform, shuffled=config.shuffled)

def get_dataloader():
    random_indices = random.sample(range(len(dataset)), config.chunks_size)
    sampler = SubsetRandomSampler(random_indices)
    return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True, num_workers=16)