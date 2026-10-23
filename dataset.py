"""Dataset loader for ShanghaiTech-style data (images + .mat head annotations).

Expected layout (ShanghaiTech Part A / B):
    data/part_A/train_data/images/IMG_1.jpg
    data/part_A/train_data/ground-truth/GT_IMG_1.mat
"""
import os
import glob

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_density_map(shape_hw, points, sigma: float = 15.0):
    """Place a 1 at every head location and blur with a Gaussian.
    The map sums to the number of people (ground-truth count)."""
    h, w = shape_hw
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        xi, yi = min(int(x), w - 1), min(int(y), h - 1)
        if xi >= 0 and yi >= 0:
            density[yi, xi] += 1.0
    return gaussian_filter(density, sigma=sigma, mode="constant")


class CrowdDataset(Dataset):
    def __init__(self, root: str, train: bool = True, crop: int = 256):
        self.img_paths = sorted(glob.glob(os.path.join(root, "images", "*.jpg")))
        self.gt_dir = os.path.join(root, "ground-truth")
        self.train, self.crop = train, crop
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        mat = sio.loadmat(os.path.join(self.gt_dir, f"GT_{name}.mat"))
        points = mat["image_info"][0, 0][0, 0][0]  # (N, 2) array of (x, y)
        density = make_density_map((img.height, img.width), points)

        if self.train:
            # random crop + horizontal flip augmentation
            w, h = img.size
            c = min(self.crop, w, h)
            x0, y0 = np.random.randint(0, w - c + 1), np.random.randint(0, h - c + 1)
            img = img.crop((x0, y0, x0 + c, y0 + c))
            density = density[y0:y0 + c, x0:x0 + c]
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                density = density[:, ::-1].copy()

        # Network output is 1/8 resolution: downsample density and keep the sum constant
        h8, w8 = density.shape[0] // 8, density.shape[1] // 8
        small = np.asarray(
            Image.fromarray(density).resize((w8, h8), Image.BICUBIC), dtype=np.float32
        ) * 64.0
        import torch
        return self.norm(img), torch.from_numpy(small).unsqueeze(0)
