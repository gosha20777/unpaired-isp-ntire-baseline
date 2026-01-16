from easydict import EasyDict

config = EasyDict()

config.path_input = "data/train/raws"
config.path_output = "data/train/jpegs"
config.vgg_path = "checkpoint/vgg_pretrained/imagenet-vgg-verydeep-19.mat"
config.pretrained_path = ""
config.color_loss_type = "mse"
config.content_loss_type = 2
config.texture_loss_type = "relativistic"
config.shuffled = True
config.lambda_gp = 10
config.w_tv = 1.0
config.w_texture_factor = 1.0
config.w_color_factor = 1.0
config.w_content = 1.0
config.w_color = 1.0
config.w_texture = 1.0
config.w_dists = 1.0
config.w_lpips_plus = 1.0
config.chunks_count = 5000
config.batch_size = 16
config.num_batches = 100
config.chunks_size = 3200
config.gen_lr = 5e-4
config.gen_betas = [0.5, 0.9]
config.start_point = 1
config.gen_per_disc_ratio_color = 10
config.gen_per_disc_ratio_texture = 10
config.disc_grad_clip = 1
config.grayscale_channels = 1
config.rgb_channels = 3

config.discriminators = {

    "0": {
        "name": "Discriminator_LPIPS_64",
        "info": "LPIPS",
        "layer_id": 0,
        "learning_rate": 1e-5,
        "betas": (0.5, 0.9),
        "w_loss" : 1.0 * config.w_texture_factor,
        "channels" : 1
    },
    "3": {
        "name": "Discriminator_LPIPS_256",
        "info": "LPIPS",
        "layer_id": 3,
        "learning_rate": 1e-5,
        "betas": (0.5, 0.9),
        "w_loss" : 1.0 * config.w_texture_factor,
        "channels" : 1
    },
    "5": {
        "name": "ViTDiscriminator",
        "info": "ViT",
        "layer_id": -1,
        "learning_rate": 1e-5,
        "betas": (0.5, 0.9),
        "w_loss": 1 * config.w_color_factor,
        "channels": 3
    }

}