import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def latent_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_channels, out_channels),
    )


def delatent_block(in_channels, out_channels, original_shape):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.Unflatten(1, original_shape)
    )


def max_pool():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def conv_1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super(UNetEncoder, self).__init__()

        self.conv_1 = conv_block(in_channels, 32)
        self.pool_1 = max_pool()
        self.conv_2 = conv_block(32, 64)
        self.pool_2 = max_pool()
        self.conv_3 = conv_block(64, 128)
        self.latent = latent_block(128 * 56 * 56, latent_dim)

    def forward(self, x):
        enc_x1 = self.conv_1(x)
        pool_x = self.pool_1(enc_x1)

        enc_x2 = self.conv_2(pool_x)
        pool_x = self.pool_2(enc_x2)

        bottleneck = self.conv_3(pool_x)

        latent = self.latent(bottleneck)

        return latent, [enc_x1, enc_x2]


class UNetDecoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=512):
        super(UNetDecoder, self).__init__()

        self.up_conv_2 = up_conv(128, 64)
        self.conv_2 = conv_block(128, 64)
        self.up_conv_1 = up_conv(64, 32)
        self.conv_1 = conv_block(64, 32)
        self.delatent = delatent_block(latent_dim, 128 * 56 * 56, (128, 56, 56))
        self.final = conv_1x1(32, out_channels)

    def forward(self, latent, skips):
        enc_x1, enc_x2 = skips

        bottleneck = self.delatent(latent)

        dec_x = self.up_conv_2(bottleneck)
        dec_x = torch.cat((dec_x, enc_x2), dim=1)
        dec_x = self.conv_2(dec_x)

        dec_x = self.up_conv_1(dec_x)
        dec_x = torch.cat((dec_x, enc_x1), dim=1)
        dec_x = self.conv_1(dec_x)

        dec_x = self.final(dec_x)

        return dec_x


class UNetAutoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        super(UNetAutoencoder, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = UNetDecoder(out_channels=out_channels, latent_dim=latent_dim)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        dec = self.decoder(bottleneck, skips)
        return dec
