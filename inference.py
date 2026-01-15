import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import csv
import argparse
from torchvision.transforms.functional import to_tensor
from math import log10
import sys
from models import BaselineModel

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(max_val / torch.sqrt(mse))

class ImagePairDataset(Dataset):
    def __init__(self, input_dir, reference_dir):
        self.input_dir = input_dir
        self.reference_dir = reference_dir
        self.image_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        input_file = self.image_files[idx]
        input_path = os.path.join(self.input_dir, input_file)
        reference_path = os.path.join(self.reference_dir, input_file)

        input_tensor = self.preprocess_raw_image(input_path)
        reference_tensor = self.preprocess_reference_image(reference_path)

        return input_tensor, reference_tensor, input_file

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

    @staticmethod
    def preprocess_reference_image(image_path):
        image = Image.open(image_path).convert("RGB")
        return to_tensor(image)

def save_generated_image(tensor, save_path):
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(save_path)


def main(model_id):
    set_seed(0)

    INPUT_DIR = "../raw_images/val/mediatek_raw"
    REFERENCE_DIR = "../raw_images/val/fujifilm"
    OUTPUT_CSV = f"csv/{model_id}.csv"
    MODEL_LOAD_PATH = f"output/baseline_model_{model_id}.pth"
    OUTPUT_DIR = "created"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BaselineModel().to(device)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    model.eval()

    batch_size = 8
    dataset = ImagePairDataset(INPUT_DIR, REFERENCE_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = []

    with torch.no_grad():
        for batch_idx, (input_tensors, reference_tensors, filenames) in enumerate(dataloader):
            input_tensors = input_tensors.to(device)
            reference_tensors = reference_tensors.to(device)

            generated_outputs = model(input_tensors)
            generated_outputs = torch.clamp(generated_outputs, 0, 1)

            for i in range(len(filenames)):
                psnr_value = calculate_psnr(generated_outputs[i], reference_tensors[i])
                results.append([filenames[i], model_id, psnr_value])

            print(f"Batch {batch_idx} processed")

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Model_ID', 'PSNR'])
        writer.writerows(results)

    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=int, help="Model ID to load")
    args = parser.parse_args()
    main(args.model_id)
