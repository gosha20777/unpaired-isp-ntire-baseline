import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)

    def forward(self, input_image):
        c1 = torch.tanh(self.conv1(input_image))

        c2 = F.relu(self.conv2(c1))

        c3 = F.relu(self.conv3(c2))

        enhanced = F.pixel_shuffle(c3, upscale_factor=2)

        return enhanced


class DPEDDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(DPEDDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=2)

        self.feature_map_size = None

        self.fc1 = None
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)

        if self.feature_map_size is None:
            self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(self.feature_map_size, 1024).to(x.device)

        x = torch.flatten(x, start_dim=1)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.softmax(self.fc2(x), dim=1)

        return x


class BaseDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(BaseDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=1)

        self.feature_map_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)

        if self.feature_map_size is None:
            self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(self.feature_map_size, 1024).to(x.device)

        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)

        return x

class DPEDDiscriminator_logit(nn.Module):
    def __init__(self, input_channels):
        super(DPEDDiscriminator_logit, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=1)

        self.feature_map_size = 1
        self.fc1 = nn.Linear(self.feature_map_size, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)

        if self.feature_map_size != x.size(1) * x.size(2) * x.size(3):
            self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(self.feature_map_size, 1024).to(x.device)

        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)

        return x


class DPEDDiscriminator_logit_color(nn.Module):
    def __init__(self, input_channels=3):
        super(DPEDDiscriminator_logit_color, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=2)

        self.feature_map_size = 1
        self.fc1 = nn.Linear(self.feature_map_size, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)

        if self.feature_map_size != x.size(1) * x.size(2) * x.size(3):
            self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(self.feature_map_size, 1024).to(x.device)

        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)

        return x



class Discriminator_LPIPS(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator_LPIPS, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=1)

        self.feature_map_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)

        if self.feature_map_size is None:
            self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(self.feature_map_size, 1024).to(x.device)

        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)

        return x

class Discriminator_LPIPS_64(Discriminator_LPIPS):
    def __init__(self):
        super().__init__(input_channels=64)

class Discriminator_LPIPS_192(Discriminator_LPIPS):
    def __init__(self):
        super().__init__(input_channels=192)

class Discriminator_LPIPS_384(Discriminator_LPIPS):
    def __init__(self):
        super().__init__(input_channels=384)

class Discriminator_LPIPS_256(Discriminator_LPIPS):
    def __init__(self):
        super().__init__(input_channels=256)


class ViTDiscriminator(nn.Module):
    def __init__(self, input_dim=768, num_patches=196, hidden_dim=512, num_layers=3):
        super(ViTDiscriminator, self).__init__()

        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ])

    def forward(self, vit_features):

        x = vit_features

        for layer in self.layers:
            x = layer(x)

        x = torch.mean(x, dim=1)

        return x