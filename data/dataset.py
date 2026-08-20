import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms

class ColorizationDataset(Dataset):
    def __init__(self, file_paths):
        self.paths = file_paths
        # I use Bicubic interpolation because it keeps lines sharper than Bilinear
        self.transforms = transforms.Resize((256, 256), Image.BICUBIC)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            # Load image and ensure it's RGB (even if it's black/white originally)
            img = Image.open(self.paths[idx]).convert("RGB")
            
            # Resize to 256x256
            img = self.transforms(img)
            img = np.array(img)
            
            # Convert to LAB color space
            # RGB is 0-255, but LAB is:
            # L: 0 to 100
            # a: -128 to 128
            # b: -128 to 128
            img_lab = rgb2lab(img).astype("float32")
            
            # Convert to Tensor
            img_lab = transforms.ToTensor()(img_lab)
            
            # Normalize to range [-1, 1] for the neural network
            # L channel: (0 to 100) -> (-1 to 1)
            L = img_lab[[0], ...] / 50. - 1.
            
            # ab channels: (-128 to 128) -> (-1 to 1)
            # I divide by 110 because real photos rarely hit the max/min of 128
            ab = img_lab[[1, 2], ...] / 110.
            
            return L, ab
            
        except Exception as e:
            # If a file is broken, just return zeros so training doesn't crash
            print(f"Warning: Skipping broken image {self.paths[idx]}")
            return torch.zeros(1, 256, 256), torch.zeros(2, 256, 256)