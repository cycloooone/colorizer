import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import warnings

# UNET class
from models.unet import UNet

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ignore the skimage warning about negative colors
warnings.filterwarnings("ignore")

def calculate_psnr(img1, img2):
    """
    I am using PSNR (Peak Signal-to-Noise Ratio) to measure quality.
    - It measures how much "noise" (difference) is in the predicted image compared to the original.
    - Measured in Decibels (dB). Higher is better.
    - Usually, >30dB is great, <20dB is bad.
    - Formula for PSNR: 20 * log10(MAX / sqrt(MSE))
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def process_img(path):
    # Load image and resize to 256
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)
    img_array = np.array(img)
    
    # Convert to LAB space
    img_lab = rgb2lab(img_array).astype("float32")
    
    # Normalize L channel to be between -1 and 1
    L_channel = img_lab[:, :, 0] / 50. - 1.
    
    # Turn into tensor for the model
    L_tensor = torch.from_numpy(L_channel).unsqueeze(0).unsqueeze(0).to(device)
    
    return L_tensor, L_channel, img_array

def to_rgb(L_original, ab_pred):
    # Take the predicted ab channels and move to cpu
    ab = ab_pred.detach().cpu().numpy()[0].transpose(1, 2, 0)
    
    # Resize back to normal numbers
    L = (L_original + 1.) * 50.
    ab = ab * 110.
    
    # Combine and convert back to RGB
    lab_image = np.dstack((L, ab))
    return lab2rgb(lab_image)

def main():
    # Make sure output folder exists otherwise create
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    # Create the two models
    net_basic = UNet().to(device) # Basic UNET
    net_gan = UNet().to(device)   # GAN version
    
    print("Loading weights...")
    try:
        # I need to use map_location in case I run this on my laptop (cpu)
        map_loc = device 
        net_basic.load_state_dict(torch.load("checkpoints/unet_best.pth", map_location=map_loc))
        net_gan.load_state_dict(torch.load("checkpoints/gan_generator_best.pth", map_location=map_loc))
    except Exception as e:
        print("Error loading weights!", e)
        return

    net_basic.eval()
    net_gan.eval()

    # Get all jpg/png images (Filter to avoid crashing on hidden files)
    all_files = glob.glob("data/test_samples/*")
    test_images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(test_images)} images \n")

    for path in test_images:
        name = os.path.basename(path)
        
        # Prepare data
        L_input, L_orig, ground_truth = process_img(path)
        
        # Predict
        with torch.no_grad():
            pred_basic = net_basic(L_input)
            pred_gan = net_gan(L_input)
            
        # Convert to image
        img_basic = to_rgb(L_orig, pred_basic)
        img_gan = to_rgb(L_orig, pred_gan)
        
        # Calculate Scores
        # Ground truth needs to be 0-1 float for math
        gt_float = ground_truth / 255.0
        
        score_basic = calculate_psnr(gt_float, img_basic)
        score_gan = calculate_psnr(gt_float, img_gan)
        
        print(f"{name}:   Scores -> Unet Basic: {score_basic:.2f} | Unet GAN: {score_gan:.2f}")

        # Saving the plot
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        
        ax[0].imshow(L_orig, cmap='gray')
        ax[0].set_title("Input")
        ax[0].axis("off")
        
        ax[1].imshow(ground_truth)
        ax[1].set_title("Original")
        ax[1].axis("off")
        
        ax[2].imshow(img_basic)
        ax[2].set_title(f"L1 Loss Only\nPSNR: {score_basic:.1f}")
        ax[2].axis("off")
        
        ax[3].imshow(img_gan)
        ax[3].set_title(f"GAN\nPSNR: {score_gan:.1f}")
        ax[3].axis("off")
        
        plt.savefig(f"outputs/result_{name}")
        plt.close()
    print('\n ------- Please look at results on "outputs" folder -------\n')
    print('If you wish test with your own images')
    print('Upload -----> "data/test_samples" folder.\n')

if __name__ == "__main__":
    main()