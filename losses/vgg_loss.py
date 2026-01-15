import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import scipy.io
import kornia.color as kornia_color
from utils import save_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_MEAN = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)


def preprocess(image):
    return (image * 255) - IMAGE_MEAN.to(image.device)


class VGGFromMAT(nn.Module):
    def __init__(self, path_to_vgg_net):
        super(VGGFromMAT, self).__init__()

        self.layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        data = scipy.io.loadmat(path_to_vgg_net)
        self.weights = data['layers'][0]

        self.net = self._build_network()

        for param in self.parameters():
            param.requires_grad = False

    def _build_network(self):
        layers = nn.ModuleDict()
        current_layer = None

        for i, name in enumerate(self.layers):
            layer_type = name[:4]
            if layer_type == 'conv':
                kernels, bias = self.weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (3, 2, 0, 1))
                bias = bias.reshape(-1)

                conv = nn.Conv2d(
                    in_channels=kernels.shape[1],
                    out_channels=kernels.shape[0],
                    kernel_size=kernels.shape[2],
                    stride=1,
                    padding=kernels.shape[2] // 2,
                    bias=True
                )
                conv.weight.data = torch.tensor(kernels, dtype=torch.float32)
                conv.bias.data = torch.tensor(bias, dtype=torch.float32)

                layers[name] = conv

            elif layer_type == 'relu':
                layers[name] = nn.ReLU(inplace=True)

            elif layer_type == 'pool':
                layers[name] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        return layers

    def forward(self, x):
        outputs = {}
        for name, layer in self.net.items():
            x = layer(x)
            outputs[name] = x
        return outputs


def to_L_channel(image_tensor):
    image_LAB = kornia_color.rgb_to_lab(image_tensor)
    L_channel = image_LAB[:, :1, :, :]

    L_channel = L_channel / 100.0

    return L_channel.repeat(1, 3, 1, 1)


def compute_content_loss(vgg, generated, target):
    save_images(10, [to_L_channel(generated), to_L_channel(target)], ["gen2", "target2"])

    generated_preprocessed = preprocess(to_L_channel(generated))
    target_preprocessed = preprocess(to_L_channel(target))

    generated_features = vgg(generated_preprocessed)
    target_features = vgg(target_preprocessed)

    layer_name = 'relu5_4'
    generated_map = generated_features[layer_name]
    target_map = target_features[layer_name]

    content_size = generated_map.numel() / generated.shape[0]
    content_loss = 2 * F.mse_loss(generated_map, target_map, reduction='sum') / content_size

    return content_loss
