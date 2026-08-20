import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        # normalize=False for the first layer (standard for GANs)
        self.enc1 = self.conv_block(1, 64, normalize=False)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)
        self.enc7 = self.conv_block(512, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 512, normalize=False)
        
        # Decoder
        # Dropout in the first few layers to add randomness
        self.dec1 = self.up_block(512, 512, dropout=True)
        self.dec2 = self.up_block(1024, 512, dropout=True)
        self.dec3 = self.up_block(1024, 512, dropout=True)
        self.dec4 = self.up_block(1024, 512)
        self.dec5 = self.up_block(1024, 256)
        self.dec6 = self.up_block(512, 128)
        self.dec7 = self.up_block(256, 64)
        
        # Final Output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    # Helper for Convolutional layers
    def conv_block(self, in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize: 
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    # Helper for Upsampling
    def up_block(self, in_c, out_c, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True) # ReLU is standard for decoder
        ]
        if dropout: 
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsampling
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Upsampling with Skip Connections
        
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e7], 1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], 1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], 1)
        
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], 1)
        
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], 1)
        
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], 1)
        
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], 1)
        
        return self.final(d7)