import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from scipy.ndimage import convolve
import random
import torch.nn.functional as F
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from config import config

class ZRRDataset(Dataset):
    def __init__(self, img_dir1, img_dir2, raw_transform, raw_to_rgb_transform, dslr_transform, shuffled):
        self.img_path1 = sorted([os.path.join(img_dir1, f) for f in os.listdir(img_dir1)])
        self.img_path2 = sorted([os.path.join(img_dir2, f) for f in os.listdir(img_dir2)])
        self.raw_transform = raw_transform
        self.raw_to_rgb_transform = raw_to_rgb_transform
        self.dslr_transform = dslr_transform
        if shuffled:
            print("Target images are shuffled!")
            self.img_path2 = random.sample(self.img_path2, len(self.img_path2))
        else:
            print("Target images are not shuffled!")


    def __len__(self):
        return len(self.img_path1)

    def __getitem__(self, idx):
        raw_ = Image.open(self.img_path1[idx])
        custom_rgb_ = Image.open(self.img_path1[idx])
        dslr_ = Image.open(self.img_path2[idx])

        raw_ = self.raw_transform(raw_)
        custom_rgb_ = self.raw_to_rgb_transform(custom_rgb_)
        dslr_ = self.dslr_transform(dslr_)

        return raw_, custom_rgb_, dslr_



raw_transform = transforms.Compose([
    transforms.Lambda(lambda raw: (
        lambda: (
            (raw_array := np.array(raw)),
            (RAW_combined := np.dstack((
                raw_array[1::2, 1::2],
                raw_array[0::2, 1::2],
                raw_array[0::2, 0::2],
                raw_array[1::2, 0::2]
            ))),
            (min_vals := np.array([62, 60, 58, 61], dtype=np.float32)),
            (max_val := 1023),
            (normalized_channels := (RAW_combined - min_vals) / (max_val - min_vals)),

            torch.tensor(normalized_channels, dtype=torch.float32).permute(2, 0, 1)
        )[-1]
    )())
])


from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

gamma = 2.2
wb_factors = [1.6, 1.2, 1.4]


def raw_to_rgb_transform(raw):
    raw_array = np.array(raw, dtype=np.float32)

    if len(raw_array.shape) > 2:
        raw_array = raw_array[:, :, 0]

    raw_array = raw_array / 65535.0

    bayer_data = (raw_array * 65535).astype(np.uint16)

    try:
        rgb_image = demosaicing_CFA_Bayer_Menon2007(bayer_data, pattern='RGGB')
    except Exception as e:
        print(f"Demosaicing error: {e}")
        print(f"Bayer data shape: {bayer_data.shape}, min: {bayer_data.min()}, max: {bayer_data.max()}")
        raise

    rgb_image = rgb_image.astype(np.float32) / 65535.0

    rgb_image = np.stack([
        rgb_image[:, :, 0] * wb_factors[0],
        rgb_image[:, :, 1] * wb_factors[1],
        rgb_image[:, :, 2] * wb_factors[2],
    ], axis=-1)

    rgb_image = np.clip(rgb_image, 0, 1)

    range_val = rgb_image.max() - rgb_image.min()
    if range_val > 0:
        rgb_image = (rgb_image - rgb_image.min()) / range_val

    rgb_image = np.power(rgb_image, 1.0 / gamma)

    rgb_image_tensor = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    rgb_image_resized = F.interpolate(
        rgb_image_tensor,
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )

    return rgb_image_resized.squeeze(0)

def raw_to_rgb_transform2(raw):
    raw_array = np.array(raw, dtype=np.float32)
    raw_array = raw_array / 255.0 if raw_array.max() <= 255 else raw_array / 1023.0

    bayer_data = (raw_array * 65535).astype(np.uint16)

    rgb_image = demosaicing_CFA_Bayer_Menon2007(bayer_data, pattern='RGGB')

    rgb_image = rgb_image.astype(np.float32) / 65535.0

    rgb_image = np.stack([
        rgb_image[:, :, 0] * wb_factors[0],
        rgb_image[:, :, 1] * wb_factors[1],
        rgb_image[:, :, 2] * wb_factors[2],
    ], axis=-1)

    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    rgb_image = np.power(rgb_image, 1.0 / gamma)

    rgb_image_tensor = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    rgb_image_resized = F.interpolate(
        rgb_image_tensor,
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )

    return rgb_image_resized.squeeze(0)


dslr_transform = transforms.Compose([
    transforms.Lambda(lambda I: (
        torch.tensor(
            np.array(I),
            dtype=torch.float32
        ).permute(2, 0, 1) / 255
    ))
])

dataset = ZRRDataset(config.path_input, config.path_output, raw_transform, raw_to_rgb_transform, dslr_transform, shuffled=config.shuffled)

def get_dataloader():
    random_indices = random.sample(range(len(dataset)), config.chunks_size)
    sampler = SubsetRandomSampler(random_indices)
    return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True, num_workers=4)