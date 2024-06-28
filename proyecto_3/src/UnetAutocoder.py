import torch
import torch.nn as nn

class UNetAutoencoder(nn.Module):
  def __init__(self, in_channels=3, latent_dim=512):
    super(UNetAutoencoder, self).__init__()
    # Encoder
    self.enc1 = self.conv_block(in_channels, 64)
    self.pool1 = self.max_pool()
    self.enc2 = self.conv_block(64, 128)
    self.pool2 = self.max_pool()
    self.enc3 = self.conv_block(128, 256)
    self.pool3 = self.max_pool()
    
    # Bottleneck
    self.bottleneck = self.bottleneck_block(256, 512)
    
    # Latent
    self.latent = self.latent_block(512, latent_dim)

    # Decoder
    self.upconv3 = self.up_conv(512, 512)
    self.dec3 = self.upconv_block(512, 256)
    self.upconv2 = self.up_conv(512, 512)
    self.dec2 = self.upconv_block(256, 128)
    self.upconv1 = self.up_conv(512, 512)
    self.dec1 = self.upconv_block(128, 64)
    
    # Final Convolution
    self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)

  def conv_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )
  
  def bottleneck_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True)
    )
  
  def latent_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(in_channels, out_channels)
    )
  
  def max_pool(self):
    return nn.MaxPool2d(kernel_size=2, stride=2)
  
  def up_conv(self, in_channels, out_channels):
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

# Example usage
# model = UNetAutoencoder(in_channels=3, latent_dim=512)  # Assuming 512 latent dimensions
# input_image = torch.randn(1, 3, 256, 256)  # Batch size 1, RGB image of size 256x256
# output_image, latent_vector = model(input_image)
# print(output_image.shape)  # Should be (1, 3, 256, 256)
# print(latent_vector.shape)  # Should be (1, 512)
