import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models import BaselineModel, DPEDDiscriminator_logit
from utils import preprocess_img_disc

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

INPUT_DIR = "../raw_images/test/mediatek_raw"
OUTPUT_DIR = "visual_result"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class RawImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        return self.preprocess_raw_image(image_path), self.image_files[idx]

    @staticmethod
    def preprocess_raw_image(image_path):
        raw_image = Image.open(image_path)
        raw_array = np.array(raw_image)

        RAW_combined = np.dstack((
            raw_array[1::2, 1::2],
            raw_array[0::2, 1::2],
            raw_array[0::2, 0::2],
            raw_array[1::2, 0::2]
        ))
        min_vals = np.array([62, 60, 58, 61], dtype=np.float32)
        max_val = 1023
        normalized_channels = (RAW_combined - min_vals) / (max_val - min_vals)

        tensor = torch.tensor(normalized_channels, dtype=torch.float32).permute(2, 0, 1)
        return tensor


def save_generated_image(output_tensor, save_path):
    output_tensor = output_tensor.permute(1, 2, 0).detach().cpu()
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_image = (output_tensor.numpy() * 255).astype(np.uint8)
    Image.fromarray(output_image).save(save_path)


batch_size = 8
dataset = RawImageDataset(INPUT_DIR)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for id in range(27220, 27221, 100):
    MODEL_LOAD_PATH = f"output/baseline_model_{id}.pth"
    model = BaselineModel().to(device)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))

    model.eval()
    for batch_idx, (input_tensors, filenames) in enumerate(dataloader):
        input_tensors = input_tensors.to(device)

        with torch.no_grad():
            generated_outputs = model(input_tensors)
            x, y = preprocess_img_disc(generated_outputs, generated_outputs, 1)

        print(f"Model {id}, Batch {batch_idx}")
        for i, filename in enumerate(filenames):
            output_path = os.path.join(OUTPUT_DIR, f"{filename}")
            save_generated_image(generated_outputs[i], output_path)
