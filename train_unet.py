import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import os

# My modules
from models.unet import UNet
from data.dataset import ColorizationDataset

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
LR = 2e-4
BATCH_SIZE = 32
EPOCHS = 20
IMG_PATH = "data/landscapes/*.jpg"

def train():
    # Setup Data
    files = glob.glob(IMG_PATH)
    if not files:
        print(f"Error: No images found in {IMG_PATH}")
        return
    
    print(f"Found {len(files)} images. Starting training...")
    
    # I use a larger batch size here (32) compared to GAN (16) because 
    # pure U-Net uses less memory (no discriminator).
    loader = DataLoader(ColorizationDataset(files), batch_size=BATCH_SIZE, shuffle=True)

    # Setup Model
    unet = UNet().to(device)
    
    # Optimizer & Loss
    # L1 Loss is better for colorization than MSE because MSE encourages "muddy" gray averages
    criterion = nn.L1Loss()
    optimizer = optim.Adam(unet.parameters(), lr=LR, betas=(0.5, 0.999))
    
    # Create folder for saving weights
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    print(f"--> Starting Phase 1: Pure U-Net Training ({EPOCHS} epochs)...")

    for epoch in range(EPOCHS):
        unet.train()
        running_loss = 0.0
        
        for i, (input_l, target_ab) in enumerate(loader):
            input_l = input_l.to(device)
            target_ab = target_ab.to(device)
            
            # Forward pass
            outputs = unet(input_l)
            loss = criterion(outputs, target_ab)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average L1 Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        if (epoch+1) % 5 == 0:
            torch.save(unet.state_dict(), "checkpoints/unet_best.pth")
            print("Checkpoint saved.")

if __name__ == "__main__":
    train()