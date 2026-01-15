import os
import tensorflow as tf
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import pandas as pd

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def loss_psnr(y_true, y_pred):
    loss_mse = tf.math.reduce_mean(tf.pow(y_true - y_pred, 2))
    loss_psnr = 20 * log10(1.0 / tf.sqrt(loss_mse))
    return loss_psnr


def loss_ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_pred, y_true, max_val=1.0))


def ms_ssim_loss(y_true, y_pred):
    return tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)


lpips_alexnet = LearnedPerceptualImagePatchSimilarity(net_type='alex')


def calculate_lpips(img1_path, img2_path):

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    img1_tensor = TF.to_tensor(img1).unsqueeze(0) * 2 - 1
    img2_tensor = TF.to_tensor(img2).unsqueeze(0) * 2 - 1

    with torch.no_grad():
        lpips_score = lpips_alexnet(img1_tensor, img2_tensor).item()
    return lpips_score


def process_images(reference_dir, comparison_dirs):

    results = []
    reference_images = sorted([f for f in os.listdir(reference_dir) if f.endswith('.jpg')])

    for comp_dir in comparison_dirs:
        comp_name = os.path.basename(comp_dir)
        print(f"\nProcessing folder: {comp_name}")

        for ref_img in tqdm(reference_images, desc=f"Comparing images in {comp_name}"):
            ref_path = os.path.join(reference_dir, ref_img)
            comp_path = os.path.join(comp_dir, ref_img.replace('.jpg', '.png'))

            if os.path.exists(comp_path):
                image1 = tf.io.read_file(ref_path)
                image1 = tf.image.decode_image(image1, channels=3)
                image1 = tf.cast(image1, tf.float32) / 255.0

                image2 = tf.io.read_file(comp_path)
                image2 = tf.image.decode_image(image2, channels=3)
                image2 = tf.cast(image2, tf.float32) / 255.0

                psnr_score = loss_psnr(image1, image2).numpy()
                ssim_score = loss_ssim(image1, image2).numpy()
                ms_ssim_score = ms_ssim_loss(image1, image2).numpy()

                lpips_score = calculate_lpips(ref_path, comp_path)

                results.append([ref_img, psnr_score, ssim_score, ms_ssim_score, lpips_score])
            else:
                print(f"Missing corresponding image for {ref_img} in {comp_dir}")

    return results


if __name__ == '__main__':

    reference_dir = '../../ZRR2/test/canon'
    comparison_dir1 = ['visual_result']

    metrics_results1 = process_images(reference_dir, comparison_dir1)

    df1 = pd.DataFrame(metrics_results1, columns=['Name', 'PSNR', 'SSIM', 'MS-SSIM', 'LPIPS-ALEX'])

    excel_file1 = 'baseline_supervised.csv'
    df1.to_csv(excel_file1, index=False)

    print("\nMetrics have been successfully saved.")
