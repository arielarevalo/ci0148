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


def max_pool():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def conv_1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super(UNetEncoder, self).__init__()

        self.conv_1 = conv_block(in_channels, 64)
        self.pool_1 = max_pool()
        self.conv_2 = conv_block(64, 128)
        self.pool_2 = max_pool()
        self.conv_3 = conv_block(128, 256)
        self.pool_3 = max_pool()
        self.conv_4 = conv_block(256, 512)
        self.pool_4 = max_pool()
        self.conv_5 = conv_block(512, 1024)

        self.latent = latent_block(1024 * 14 * 14, latent_dim)

    def forward(self, x):
        enc_x1 = self.conv_1(x)
        pool_x = self.pool_1(enc_x1)

        enc_x2 = self.conv_2(pool_x)
        pool_x = self.pool_2(enc_x2)

        enc_x3 = self.conv_3(pool_x)
        pool_x = self.pool_3(enc_x3)

        enc_x4 = self.conv_4(pool_x)
        pool_x = self.pool_4(enc_x4)

        bottleneck = self.conv_5(pool_x)

        return bottleneck, [enc_x1, enc_x2, enc_x3, enc_x4]

    def extract_latent(self, x):
        bottleneck, _ = self.forward(x)
        latent_vectors = self.latent(bottleneck)
        return latent_vectors


class UNetDecoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UNetDecoder, self).__init__()

        self.up_conv_4 = up_conv(1024, 512)
        self.conv_4 = conv_block(1024, 512)
        self.up_conv_3 = up_conv(512, 256)
        self.conv_3 = conv_block(512, 256)
        self.up_conv_2 = up_conv(256, 128)
        self.conv_2 = conv_block(256, 128)
        self.up_conv_1 = up_conv(128, 64)
        self.conv_1 = conv_block(128, 64)

        self.final = conv_1x1(64, out_channels)

    def forward(self, x, skips):
        enc_x1, enc_x2, enc_x3, enc_x4 = skips

        dec_x = self.up_conv_4(x)
        dec_x = torch.cat((dec_x, enc_x4), dim=1)
        dec_x = self.conv_4(dec_x)

        dec_x = self.up_conv_3(dec_x)
        dec_x = torch.cat((dec_x, enc_x3), dim=1)
        dec_x = self.conv_3(dec_x)

        dec_x = self.up_conv_2(dec_x)
        dec_x = torch.cat((dec_x, enc_x2), dim=1)
        dec_x = self.conv_2(dec_x)

        dec_x = self.up_conv_1(dec_x)
        dec_x = torch.cat((dec_x, enc_x1), dim=1)
        dec_x = self.conv_1(dec_x)

        dec_x = self.final(dec_x)

        return dec_x


class UNetAutoencoder(nn.Module):
    def __init__(self, encoder=UNetEncoder(in_channels=3, latent_dim=512), decoder=UNetDecoder(out_channels=3)):
        super(UNetAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        dec = self.decoder(bottleneck, skips)
        return dec
