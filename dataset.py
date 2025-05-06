# === dataset.py ===
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import random

class InpaintingFromFilesDataset(Dataset):
    def __init__(self, clean_dir, image_size=128, train=True):
        self.clean_dir = clean_dir
        self.paths = sorted(os.listdir(clean_dir))  # always sorted for consistency
        self.train = train
        self.image_size = image_size
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5 if train else 0),
            T.RandomRotation(15, fill=0),
            T.ToTensor()
        ])

    def random_square_mask(self, img_size=128, min_patch=96, max_patch=96, num_holes=1):
        mask = torch.ones((1, img_size, img_size))
        for _ in range(num_holes):
            patch_size = random.randint(min_patch, max_patch)
            x = random.randint(0, img_size - patch_size)
            y = random.randint(0, img_size - patch_size)
            mask[:, y:y+patch_size, x:x+patch_size] = 0
        return mask


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        filename = self.paths[idx]
        clean_path = os.path.join(self.clean_dir, filename)

        # Load and transform clean image
        clean = self.transform(Image.open(clean_path).convert("L"))

        if self.train:
            # Generate a random mask
            mask = self.random_square_mask(self.image_size)
            # Apply the mask
            masked = clean * mask
        else:
            # No random mask during evaluation
            masked = clean
            mask = (masked > 0.1).float()

        return {
            "masked": masked,
            "mask": mask,
            "target": clean
        }
