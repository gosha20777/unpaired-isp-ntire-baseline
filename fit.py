import torch
from models import BaselineModel
import models
from load_dataset_ntire import get_dataloader
from losses.vgg_loss import VGGFromMAT
from helper import save_baseline, train_baseline, train_disc
from utils import *
from config import config
from helper import instantiate_discriminator, load_baseline_model
from utils import save_images
set_seed(0)

model = load_baseline_model()

optimizer = torch.optim.Adam(list(model.parameters()), lr=config.gen_lr, betas=config.gen_betas)
discriminators_list = instantiate_discriminator()

vgg = VGGFromMAT(config.vgg_path).to(device)

for chunk_idx in range(config.start_point, config.chunks_count):
    dataloader = get_dataloader()

    for batch_idx, (input_raw, custom_rgb, target_dslr) in enumerate(dataloader):
        input_raw = input_raw.to(device)
        custom_rgb = custom_rgb.to(device)
        target_dslr = target_dslr.to(device)

        generated = model(input_raw)

        save_images(batch_idx, [generated, custom_rgb, target_dslr], ["gen", "custom", "target"])


        train_baseline(config.color_loss_type, config.texture_loss_type, vgg, discriminators_list, optimizer, generated, custom_rgb, target_dslr,
                               config.content_loss_type, batch_idx, model, chunk_idx)

        for i, disc in enumerate(discriminators_list):
            train_disc(discriminators_list[i], generated, target_dslr, batch_idx, discriminators_list[i]['channels'], chunk_idx)


        save_baseline(model, batch_idx, chunk_idx, generated)

