import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import argparse
import sys
from tqdm import tqdm

# Ensure 'models.py' is accessible
try:
    from models import BaselineModel
except ImportError:
    print("Error: Could not import 'BaselineModel'. Make sure 'models.py' is in the current directory.")
    sys.exit(1)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class InferenceDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        # Assuming inputs are .npy files based on previous context
        self.image_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        input_path = os.path.join(self.input_dir, filename)
        
        # Preprocess directly here
        raw_array = np.load(input_path)
        max_val = 1023.0
        normalized_channels = raw_array / max_val
        input_tensor = torch.tensor(normalized_channels, dtype=torch.float32).permute(2, 0, 1)
        
        return input_tensor, filename

def save_generated_image(tensor, save_path):
    """
    Converts a (C, H, W) tensor to a PIL image and saves it.
    """
    # Move to CPU, permute to (H, W, C), and convert to numpy
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()
    
    # Scale to 0-255 and clip
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    
    # Save
    Image.fromarray(tensor).save(save_path)

def main(model_id, input_dir, output_dir, models_folder):
    set_seed(0)

    # Construct the model filename based on the ID
    model_filename = f"baseline_model_{model_id}.pth"
    model_path = os.path.join(models_folder, model_filename)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"--- Generation Configuration ---")
    print(f"Device:          {device}")
    print(f"Model ID:        {model_id}")
    print(f"Loading from:    {model_path}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory:{output_dir}")
    print(f"--------------------------------")

    # 1. Validation
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # 2. Load Model
    print("Loading model...")
    model = BaselineModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
    model.eval()

    # 3. Setup Data
    dataset = InferenceDataset(input_dir)
    # Batch size 1 is safer for high-res generation, increase if GPU allows
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 4. Prepare Output Directory
    os.makedirs(output_dir, exist_ok=True)

    # 5. Generation Loop
    print(f"Generating images for {len(dataset)} files...")
    
    with torch.no_grad():
        for input_tensors, filenames in tqdm(dataloader, desc="Saving JPEGs"):
            input_tensors = input_tensors.to(device)

            # Inference
            generated_outputs = model(input_tensors)
            generated_outputs = torch.clamp(generated_outputs, 0, 1)

            # Save batch results
            for i in range(len(filenames)):
                output_filename = filenames[i].replace(".npy", ".jpg")
                save_path = os.path.join(output_dir, output_filename)
                save_generated_image(generated_outputs[i], save_path)

    print(f"\nSuccess! Images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JPEGs using a specific model ID.")
    
    # Positional argument for the ID
    
    # Optional arguments
    parser.add_argument('--input_dir', type=str, default="data/test1/raws", 
                        help="Directory containing input .npy files")
    parser.add_argument('--output_dir', type=str, default="results/generated", 
                        help="Directory to save the output JPEGs")
    parser.add_argument('--models_folder', type=str, default='output', 
                        help="Folder containing the .pth model files")
    parser.add_argument('--model_id', default=1700, type=int, help="The ID of the model to use (e.g., 50)")

    args = parser.parse_args()
    
    main(args.model_id, args.input_dir, args.output_dir, args.models_folder)