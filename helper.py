import torch
import os
import numpy as np

from losses.vgg_loss import compute_content_loss
from losses.tv_loss import compute_tv_loss
from losses.gan_loss import compute_generator_loss, compute_discriminator_loss
from losses.color_mse import compute_generator_loss_mse
from losses.dists import compute_dists_loss
from losses.lpips_plus import compute_lpips_plus_loss
from utils import writer
from config import config
import models
from models import BaselineModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_baseline(model, batch_idx, chunk_idx, generated):

    if batch_idx % 10 == 0:
        save_path = f"output/baseline_model_{chunk_idx * 100 + batch_idx}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def load_baseline_model():
    model = BaselineModel().to(device)

    if os.path.exists(config.pretrained_path):
        model.load_state_dict(torch.load(config.pretrained_path, map_location=device))
        print(f"Pretrained model loaded from {config.pretrained_path}")
    else:
        print(f"Pretrained model not found. Starting training from scratch.")

    return model


def normalize_loss(model, optimizer, loss):

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    total_norm = torch.sqrt(torch.sum(torch.tensor([p.grad.norm(2) ** 2 for p in model.parameters() if p.grad is not None])))
    ret = 1.0 / (total_norm.to(device) + 1e-8)
    optimizer.zero_grad()

    return ret


def instantiate_discriminator():
    if not hasattr(config, 'discriminators') or not config.discriminators:
        config.discriminators = {}
        return []

    discriminators_list = []
    for key in config.discriminators:
        disc = config.discriminators[key]
        model_instance = getattr(models, disc["name"], None)().to(device)

        if model_instance is None:
            print(f"{disc['name']} not found in models.py")
            continue

        optimizer = torch.optim.Adam(
            model_instance.parameters(),
            lr=disc["learning_rate"],
            betas=disc["betas"]
        )

        disc_info = {
            "name": disc["name"],
            "model": model_instance,
            "optimizer": optimizer,
            "info" : disc["info"],
            "layer_id": disc["layer_id"],
            "w_loss": disc["w_loss"],
            "channels": disc["channels"]
        }

        discriminators_list.append(disc_info)
    return discriminators_list

def train_baseline(color_loss_type, texture_loss_type, vgg, discriminators_list, optimizer, generated, custom_rgb,
                   target_dslr, content_loss_type, batch_idx, model, chunk_idx):
    if content_loss_type == 1:
        content_loss = compute_content_loss(vgg, generated, target_dslr)
    elif content_loss_type == 2:
        content_loss = compute_content_loss(vgg, generated, custom_rgb)

    tv_loss = compute_tv_loss(generated)

    gan_losses = []
    gan_w_losses = []
    for disc in discriminators_list:
        loss = compute_generator_loss(disc, target_dslr, generated, disc['channels'])
        gan_losses.append(loss)
        gan_w_losses.append(normalize_loss(model, optimizer, loss))


    w_content = normalize_loss(model, optimizer, content_loss) * config.w_content
    w_tv = normalize_loss(model, optimizer, tv_loss) * config.w_tv



    total_loss = (w_content * content_loss
                  + w_tv * tv_loss
                  )
    for i, disc in enumerate(discriminators_list):
        total_loss += disc["w_loss"] * gan_w_losses[i] * gan_losses[i]

    optimizer.zero_grad()

    writer.add_scalar('Generator_Losses/Content_Loss', w_content * content_loss.item(), chunk_idx * 100 + batch_idx)
    writer.add_scalar('Generator_Losses/TV_Loss', w_tv * tv_loss.item(), chunk_idx * 100 + batch_idx)
    writer.add_scalar('Generator_Losses/Total_Loss', total_loss.item(), chunk_idx * 100 + batch_idx)
    for i, disc in enumerate(discriminators_list):
        writer.add_scalar(f'Generator_Losses/GAN_loss_{i}', disc["w_loss"] * gan_w_losses[i]  * gan_losses[i].item(),
                          chunk_idx * 100 + batch_idx)


    loss_components = {
        "Content Loss": (w_content * content_loss),
        "TV Loss": (w_tv * tv_loss),
    }

    for i, disc in enumerate(discriminators_list):
        loss_components[f"GAN Loss {i}"] = disc["w_loss"] * gan_w_losses[i] * gan_losses[i]

    gradient_norms = {}

    for loss_name, loss_value in loss_components.items():
        optimizer.zero_grad()
        loss_value.backward(retain_graph=True)

        total_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                total_grad_norm += grad_norm ** 2

        gradient_norms[loss_name] = total_grad_norm ** 0.5

        writer.add_scalar(f'Gradients_GEN/{loss_name}_Norm', gradient_norms[loss_name], chunk_idx * 100 + batch_idx)

    optimizer.zero_grad()
    total_loss.backward()
    writer.add_scalar('Gradients_GEN/Total_Loss_Norm', sum(gradient_norms.values()), chunk_idx * 100 + batch_idx)

    optimizer.step()



def train_disc(disc_dict, generated, target_dslr, batch_idx, channels, chunk_idx):

    if (channels == 3 and (chunk_idx * 100 + batch_idx) % config.gen_per_disc_ratio_color == 0 )\
            or (channels == 1 and (chunk_idx * 100 + batch_idx) % config.gen_per_disc_ratio_texture == 0):
        disc_dict['optimizer'].zero_grad()
        d_loss = compute_discriminator_loss(disc_dict, target_dslr, generated, channels, batch_idx, chunk_idx)

        print(f"Batch {batch_idx} {disc_dict['name']}_{disc_dict['layer_id']} - D Loss: {d_loss:.4f}")
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc_dict['model'].parameters(), max_norm=config.disc_grad_clip)

        total_grad_norm = 0
        for name, param in disc_dict['model'].named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                total_grad_norm += grad_norm ** 2
                writer.add_scalar(f"Gradients_DISC_{disc_dict['name']}_{disc_dict['layer_id']}//{name}", grad_norm, chunk_idx * 100 + batch_idx)

        total_grad_norm = total_grad_norm ** 0.5
        writer.add_scalar(f"Gradients_DISC_{disc_dict['name']}_{disc_dict['layer_id']}//Total_Norm", total_grad_norm, chunk_idx * 100 + batch_idx)

        disc_dict['optimizer'].step()


        writer.add_scalar(f"Discriminator{disc_dict['name']}_{disc_dict['layer_id']}/Loss", d_loss.item(), chunk_idx * 100 + batch_idx)
