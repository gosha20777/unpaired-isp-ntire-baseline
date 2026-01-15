
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientISP(nn.Module):
    def __init__(self):
        super(EfficientISP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)

    def forward(self, input_image):
        c1 = torch.tanh(self.conv1(input_image))

        c2 = F.relu(self.conv2(c1))

        c3 = F.relu(self.conv3(c2))

        enhanced = F.pixel_shuffle(c3, upscale_factor=2)

        return enhanced

class RobustISP(nn.Module):
    def __init__(self):
        super(RobustISP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=3, stride=1, padding=1)

    def forward(self, input_image):
        c1 = torch.tanh(self.conv1(input_image))

        c2 = F.relu(self.conv2(c1))

        c3 = F.relu(self.conv3(c2))

        enhanced = F.pixel_shuffle(c3, upscale_factor=2)

        return enhanced

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class TextureModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextureModule, self).__init__()

        self.high_freq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.low_freq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        high_freq = self.high_freq(x)
        low_freq = self.low_freq(x)
        return torch.cat([high_freq, low_freq], dim=1)


class ToneMappingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ToneMappingModule, self).__init__()

        self.level1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.level2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.level3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):

        x1 = self.level1(x)
        x2 = self.level2(x1)
        x3 = self.level3(x2)

        x3_up = self.upsample(x3)
        x2_combined = x1 + x3_up
        illumination = self.upsample(x2_combined)

        if illumination.size() != x.size():
            illumination = F.interpolate(illumination, size=(x.size(2), x.size(3)),
                                         mode='bilinear', align_corners=False)

        reflectance = x - illumination

        return reflectance


class RMFABlock(nn.Module):
    def __init__(self, channels):
        super(RMFABlock, self).__init__()

        self.texture_module = TextureModule(channels, channels)
        self.tone_mapping = ToneMappingModule(channels, channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        texture_out = self.texture_module(x)

        tone_out = self.tone_mapping(x)

        concat_out = torch.cat([texture_out, tone_out], dim=1)

        fused = self.fusion(concat_out)

        out = self.channel_attention(fused)
        out = self.spatial_attention(out)

        return out + x


class RMFANet(nn.Module):
    def __init__(self, num_blocks=20, width=16):
        super(RMFANet, self).__init__()

        self.input_module = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.rmfa_blocks = nn.ModuleList([
            RMFABlock(width) for _ in range(num_blocks)
        ])

        self.output_module = nn.Sequential(
            nn.Conv2d(width, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.input_module(x)

        for block in self.rmfa_blocks:
            out = block(out)

        out = self.output_module(out)

        return out


def create_tiny_rmfa_net():
    return RMFANet(num_blocks=2, width=16)


def create_medium_rmfa_net():
    return RMFANet(num_blocks=8, width=16)


def create_large_rmfa_net():
    return RMFANet(num_blocks=20, width=16)


def create_large_rmfa_net_w32():
    return RMFANet(num_blocks=20, width=32)


def create_large_rmfa_net_w64():
    return RMFANet(num_blocks=20, width=64)


def preprocess_raw_with_three_channel_split(raw_data, black_level=63, normalization_factor=1023):
    has_batch_dim = len(raw_data.shape) == 3

    if not isinstance(raw_data, torch.Tensor):
        raw_data = torch.tensor(raw_data, dtype=torch.float32)
    else:
        raw_data = raw_data.clone().detach().to(torch.float32)

    if has_batch_dim:
        batch_size, height, width = raw_data.shape

        processed_batch = []
        for i in range(batch_size):
            sample = raw_data[i]

            sample = torch.clamp(sample - black_level, min=0) / normalization_factor

            R = torch.ones_like(sample)
            G = torch.ones_like(sample)
            B = torch.ones_like(sample)

            R[0::2, 0::2] = sample[0::2, 0::2]

            G[0::2, 1::2] = sample[0::2, 1::2]
            G[1::2, 0::2] = sample[1::2, 0::2]

            B[1::2, 1::2] = sample[1::2, 1::2]

            processed_batch.append(torch.stack([R, G, B], dim=0))

        return torch.stack(processed_batch, dim=0)
    else:
        raw_data = torch.clamp(raw_data - black_level, min=0) / normalization_factor

        R = torch.ones_like(raw_data)
        G = torch.ones_like(raw_data)
        B = torch.ones_like(raw_data)

        R[0::2, 0::2] = raw_data[0::2, 0::2]

        G[0::2, 1::2] = raw_data[0::2, 1::2]
        G[1::2, 0::2] = raw_data[1::2, 0::2]

        B[1::2, 1::2] = raw_data[1::2, 1::2]

        return torch.stack([R, G, B], dim=0)