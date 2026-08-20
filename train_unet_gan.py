import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import os

# My modules
from models.unet import UNet
from models.discriminator import Discriminator
from data.dataset import ColorizationDataset

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
LR = 2e-4
BATCH_SIZE = 16
EPOCHS = 20
LAMBDA_L1 = 100.0 # Weight for L1 loss
IMG_PATH = "data/landscapes/*.jpg"

def train():
    # Setup Data
    files = glob.glob(IMG_PATH)
    if not files:
        print(f"Error: No images found in {IMG_PATH}")
        return
    
    loader = DataLoader(ColorizationDataset(files), batch_size=BATCH_SIZE, shuffle=True)

    # Setup Models
    generator = UNet().to(device)
    discriminator = Discriminator().to(device)
    
    # Load weights from Basic UNet (Warm start)
    if os.path.exists("checkpoints/unet_best.pth"):
        print("Loading (U-Net) weights...")
        generator.load_state_dict(torch.load("checkpoints/unet_best.pth", map_location=device))

    # Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    # Losses
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    # Create output folder
    os.makedirs("checkpoints", exist_ok=True)

    print(f"GAN Training ({EPOCHS} epochs)...")

    for epoch in range(EPOCHS):
        for idx, (L_input, ab_real) in enumerate(loader):
            L_input = L_input.to(device)
            ab_real = ab_real.to(device)
            
            # Train Discriminator
            fake_colors = generator(L_input)
            
            # Real loss
            pred_real = discriminator(L_input, ab_real)
            loss_real = criterion_bce(pred_real, torch.ones_like(pred_real))
            
            # Fake loss (detach to avoid updating G)
            pred_fake = discriminator(L_input, fake_colors.detach())
            loss_fake = criterion_bce(pred_fake, torch.zeros_like(pred_fake))
            
            loss_d = (loss_real + loss_fake) * 0.5
            
            opt_disc.zero_grad()
            loss_d.backward()
            opt_disc.step()

            # Train Generator
            # Rerun discriminator to get gradients for G
            pred_fake_g = discriminator(L_input, fake_colors)
            
            loss_gan = criterion_bce(pred_fake_g, torch.ones_like(pred_fake_g))
            loss_l1 = criterion_l1(fake_colors, ab_real) * LAMBDA_L1
            
            loss_g = loss_gan + loss_l1
            
            opt_gen.zero_grad()
            loss_g.backward()
            opt_gen.step()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")
        
        if (epoch+1) % 5 == 0:
            torch.save(generator.state_dict(), "checkpoints/gan_generator_best.pth")
            print("Checkpoint saved.")

if __name__ == "__main__":
    train()