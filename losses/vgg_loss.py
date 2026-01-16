import torch
import torch.nn.functional as F
import torch.nn as nn
import kornia.color as kornia_color
from utils import save_images
import torchvision.transforms.v2 as transforms
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocess_transform = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VGGFromMAT(nn.Module):
    """
    A wrapper for the torchvision VGG19 model that extracts feature maps
    from specified intermediate layers.
    """
    def __init__(self, path_to_vgg_net,required_layers=['relu5_4']):
        super().__init__()
        # Load the pre-trained VGG19 model's feature extraction part
        vgg_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # VGG19 layer names mapping to the layer indices in torchvision.models
        # This allows us to select layers by their original names.
        self.layer_map = {
            'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
            'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
            'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13, 'conv3_3': 14,
            'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
            'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22, 'conv4_3': 23,
            'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
            'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31, 'conv5_3': 32,
            'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35
        }

        self.required_layers = required_layers

        # Create a new sequential model containing all layers up to the last one required.
        # This is an optimization to avoid computing unused layers.
        last_layer_idx = 0
        for layer_name in required_layers:
            if self.layer_map[layer_name] > last_layer_idx:
                last_layer_idx = self.layer_map[layer_name]

        self.vgg_slice = vgg_features[:last_layer_idx + 1].to(device).eval()

        # Freeze the model parameters
        for param in self.vgg_slice.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extracts the feature maps from the required layers."""
        outputs = {}
        for name, layer in self.vgg_slice.named_children():
            x = layer(x)
            # Check if the current layer's original name is in our required list
            for layer_name, layer_idx in self.layer_map.items():
                if str(layer_idx) == name and layer_name in self.required_layers:
                    outputs[layer_name] = x
        return outputs


def to_L_channel(image_tensor):
    image_LAB = kornia_color.rgb_to_lab(image_tensor)
    L_channel = image_LAB[:, :1, :, :]

    L_channel = L_channel / 100.0

    return L_channel.repeat(1, 3, 1, 1)


def compute_content_loss(vgg, generated, target):
    save_images(10, [to_L_channel(generated), to_L_channel(target)], ["gen2", "target2"])

    generated_preprocessed = preprocess_transform(to_L_channel(generated))
    target_preprocessed = preprocess_transform(to_L_channel(target))

    generated_features = vgg(generated_preprocessed)
    target_features = vgg(target_preprocessed)

    layer_name = 'relu5_4'
    generated_map = generated_features[layer_name]
    target_map = target_features[layer_name]

    content_size = generated_map.numel() / generated.shape[0]
    content_loss = 2 * F.mse_loss(generated_map, target_map, reduction='sum') / content_size

    return content_loss