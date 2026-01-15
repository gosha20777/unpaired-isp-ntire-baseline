import torch
import torch.nn.functional as F
from utils import preprocess_img_disc, writer
import pyiqa
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_model = pyiqa.create_metric('lpips+', device='cuda')
lpips_net = lpips_model.net
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to("cuda").eval()

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def extract_vit_features(image_tensor):
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode="bilinear", align_corners=False)
    else:
        image_tensor = transform_vit(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to("cuda")

    vit_output = vit_model(image_tensor)
    return vit_output.last_hidden_state[:, 1:, :]

def preprocess_fake_for_lpips(fake_images):
    grayscale = fake_images.mean(dim=1, keepdim=True)
    grayscale_rgb = grayscale.expand(-1, 3, -1, -1)
    return grayscale_rgb


def zero_centered_gradient_penalty(samples, logits):
    gradient, = torch.autograd.grad(outputs=logits.sum(), inputs=samples, create_graph=True)

    if gradient.dim() == 4:
        return gradient.square().sum([1, 2, 3])
    elif gradient.dim() == 3:
        return gradient.square().sum([1, 2])
    else:
        return gradient.square().sum()


def compute_discriminator_loss(disc_dict, real_images, fake_images, channels, batch_idx, chunk_idx, gamma=10):
    real_, fake_ = preprocess_img_disc(real_images, fake_images, channels)
    if disc_dict["info"] == "LPIPS":
        lpips_layer = disc_dict["layer_id"]
        if channels == 1:
            fake_ = preprocess_fake_for_lpips(fake_)
            real_ = preprocess_fake_for_lpips(real_)
            real_ = lpips_net.net(real_)[lpips_layer].detach().clone().requires_grad_(True)
            fake_ = lpips_net.net(fake_)[lpips_layer].detach().clone().requires_grad_(True)
    elif disc_dict["info"] == "ViT":
        if channels == 3:
            real_ = extract_vit_features(real_).detach().clone().requires_grad_(True)
            fake_ = extract_vit_features(fake_).detach().clone().requires_grad_(True)

    real_logits = disc_dict["model"](real_)
    fake_logits = disc_dict["model"](fake_)

    r1_penalty = zero_centered_gradient_penalty(real_, real_logits) * 100
    r2_penalty = zero_centered_gradient_penalty(fake_, fake_logits) * 100

    relativistic_logits = real_logits - fake_logits
    d_loss = F.softplus(-relativistic_logits)

    d_loss = d_loss + (gamma / 2) * (r1_penalty + r2_penalty)

    return d_loss.mean()


def compute_generator_loss(disc_dict, real_images, fake_images, channels):
    real_images, fake_images = preprocess_img_disc(real_images, fake_images, channels)

    if disc_dict["info"] == "LPIPS":
        lpips_layer = disc_dict["layer_id"]

        if channels == 1:
            fake_images = preprocess_fake_for_lpips(fake_images)
            real_images = preprocess_fake_for_lpips(real_images)
            fake_images = lpips_net.net(fake_images)[lpips_layer]
            real_images = lpips_net.net(real_images)[lpips_layer]
    elif disc_dict["info"] == "ViT":
        if channels == 3:
            real_images = extract_vit_features(real_images)
            fake_images = extract_vit_features(fake_images)

    real_logits = disc_dict["model"](real_images)
    fake_logits = disc_dict["model"](fake_images)

    relativistic_logits = fake_logits - real_logits
    g_loss = F.softplus(-relativistic_logits).mean()


    return g_loss





