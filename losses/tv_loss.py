import torch

def l2_loss(tensor):
    return torch.sum(tensor ** 2) / 2

def compute_tv_loss(image):
    device = image.device

    tv_y_size = torch.tensor((image.shape[2] - 1) * image.shape[3] * image.shape[1], dtype=torch.float32, device=device)
    tv_x_size = torch.tensor(image.shape[2] * (image.shape[3] - 1) * image.shape[1], dtype=torch.float32, device=device)

    y_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_tv = l2_loss(y_diff)

    x_diff = image[:, :, :, 1:] - image[:, :, :, :-1]
    x_tv = l2_loss(x_diff)

    loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / image.shape[0]

    return loss_tv