import torch
import torch.nn as nn

class UNetAutoencoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, latent_dim=256):
    self.encoder = UNetEncoder(in_channels, latent_dim)
    self.decoder = UNetDecoder(out_channels)
  
  def __init__(self, encoder, decoder):
    super(UNetAutoencoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )
  
  def bottleneck_block(in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True)
    )
  
  def latent_block(in_channels, out_channels):
    return nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(in_channels, out_channels)
    )
  
  def max_pool():
    return nn.MaxPool2d(kernel_size=2, stride=2)
  
  def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

  def forward(self, x):
    # Encoder
    enc1 = self.encoder1(x)
    enc1_pool = self.pool1(enc1)
    
    enc2 = self.encoder2(enc1_pool)
    enc2_pool = self.pool2(enc2)
    
    enc3 = self.encoder3(enc2_pool)
    enc3_pool = self.pool3(enc3)
    
    enc4 = self.encoder4(enc3_pool)
    enc4_pool = self.pool4(enc4)

    # Bottleneck
    bottleneck = self.bottleneck(enc4_pool)

    # Latent vector
    latent_vector = self.latent(bottleneck)

    # Decoder
    dec4 = self.upconv4(bottleneck)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.decoder4(dec4)

    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.decoder3(dec3)

    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.decoder2(dec2)

    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.decoder1(dec1)

    # Final layer
    output = self.final_layer(dec1)

    return output, latent_vector

class UNetEncoder(nn.Module):
  def __init__(self, in_channels=3, latent_dim=256):
    super(UNetEncoder, self).__init__()
    # Encoder
    self.encoder1 = UNetAutoencoder.conv_block(in_channels, 64)
    self.pool1 = UNetAutoencoder.max_pool()
    self.encoder2 = UNetAutoencoder.conv_block(64, 128)
    self.pool2 = UNetAutoencoder.max_pool()
    self.encoder3 = UNetAutoencoder.conv_block(128, 256)
    self.pool3 = UNetAutoencoder.max_pool()

    # Bottleneck
    self.bottleneck = UNetAutoencoder.bottleneck_block(256, 512)
    
    # Latent vector layer
    self.latent = UNetAutoencoder.latent_block(256, latent_dim)
      
  def forward(self, x):
    enc1 = self.encoder1(x)
    enc1_pool = self.pool1(enc1)
    
    enc2 = self.encoder2(enc1_pool)
    enc2_pool = self.pool2(enc2)
    
    enc3 = self.encoder3(enc2_pool)
    enc3_pool = self.pool3(enc3)
    
    bottleneck = self.bottleneck(enc3_pool)
    
    return bottleneck, [enc1, enc2, enc3]

  def extract_latent(self, x):
    bottleneck, _ = self.forward(x)
    latent_vector = self.latent(bottleneck)
    return latent_vector
  
class UNetDecoder(nn.Module):
  def __init__(self, out_channels=3):
    super(UNetDecoder, self).__init__()
    self.upconv3 = self.up_conv(256, 256)
    self.dec3 = self.upconv_block(256, 128)
    self.upconv2 = self.up_conv(128, 128)
    self.dec2 = self.upconv_block(128, 64)
    self.upconv1 = self.up_conv(64, 64)
    self.dec1 = self.upconv_block(64, out_channels)

    self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

  def forward(self, bottleneck, enc_feats):
    enc1, enc2, enc3 = enc_feats

    dec3 = self.upconv3(bottleneck)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.decoder3(dec3)

    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.decoder2(dec2)

    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.decoder1(dec1)

    output = self.final_layer(dec1)
    return output