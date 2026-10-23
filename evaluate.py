"""Evaluate a trained model with MAE and MSE (RMSE) on held-out data.
    python evaluate.py --test_root data/part_A/test_data --weights weights/model.pth
"""
import argparse
import math

import torch
from torch.utils.data import DataLoader

from dataset import CrowdDataset
from utils import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_root", required=True)
    p.add_argument("--weights", default="weights/model.pth")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    loader = DataLoader(CrowdDataset(args.test_root, train=False), batch_size=1)

    abs_sum, sq_sum = 0.0, 0.0
    with torch.no_grad():
        for img, target in loader:
            pred = model(img.to(device)).sum().item()
            gt = target.sum().item()
            abs_sum += abs(pred - gt)
            sq_sum += (pred - gt) ** 2

    n = len(loader)
    print(f"MAE: {abs_sum / n:.2f}")
    print(f"MSE (RMSE): {math.sqrt(sq_sum / n):.2f}")


if __name__ == "__main__":
    main()
