from utils import preprocess_img_disc
import torch.nn.functional as F
import torch

def compute_generator_loss_mse(generated, custom_rgb):
    custom_rgb_resized = F.interpolate(custom_rgb, size=(256, 256), mode='bilinear', align_corners=False)
    target_blur, enhanced_blur = preprocess_img_disc(custom_rgb_resized, generated, 3)
    #color_loss = F.mse_loss(target_blur, enhanced_blur)
    color_loss = torch.sum((target_blur - enhanced_blur) ** 2) / (2 * generated.shape[0])

    return color_loss