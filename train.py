"""Train CSRNet.  Example:
    python train.py --train_root data/part_A/train_data --val_root data/part_A/test_data --epochs 100
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CrowdDataset
from model import CSRNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", required=True)
    p.add_argument("--val_root", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--out", default="weights/model.pth")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(CrowdDataset(args.train_root, train=True), batch_size=1, shuffle=True)
    val_loader = DataLoader(CrowdDataset(args.val_root, train=False), batch_size=1)

    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for img, target in train_loader:
            img, target = img.to(device), target.to(device)
            loss = criterion(model(img), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        abs_err = 0.0
        with torch.no_grad():
            for img, target in val_loader:
                pred = model(img.to(device)).sum().item()
                abs_err += abs(pred - target.sum().item())
        mae = abs_err / len(val_loader)
        print(f"Epoch {epoch:3d} | train loss {running / len(train_loader):.6f} | val MAE {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), args.out)
            print(f"  saved best model (MAE {best_mae:.2f})")


if __name__ == "__main__":
    main()
