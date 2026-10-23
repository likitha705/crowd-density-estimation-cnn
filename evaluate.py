"""Evaluate a trained model on held-out data and plot the regression result.
 
    python evaluate.py --test_root data/part_A/test_data --weights weights/model.pth
 
Outputs: MAE, MSE (RMSE), R^2 and results/regression_plot.png
(Predicted count vs Actual count with a fitted regression line).
"""
import argparse
import math
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
 
from dataset import CrowdDataset
from utils import load_model
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_root", required=True)
    p.add_argument("--weights", default="weights/model.pth")
    p.add_argument("--out", default="results/regression_plot.png")
    args = p.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    loader = DataLoader(CrowdDataset(args.test_root, train=False), batch_size=1)
 
    preds, gts = [], []
    with torch.no_grad():
        for img, target in loader:
            preds.append(model(img.to(device)).sum().item())
            gts.append(target.sum().item())
    preds, gts = np.array(preds), np.array(gts)
 
    mae = np.mean(np.abs(preds - gts))
    mse = math.sqrt(np.mean((preds - gts) ** 2))
    ss_res = np.sum((gts - preds) ** 2)
    ss_tot = np.sum((gts - gts.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"MAE: {mae:.2f}")
    print(f"MSE (RMSE): {mse:.2f}")
    print(f"R^2: {r2:.3f}")
 
    # Regression plot: predicted vs actual count
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    slope, intercept = np.polyfit(gts, preds, 1)
    lim = max(gts.max(), preds.max()) * 1.05
    xs = np.linspace(0, lim, 100)
 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gts, preds, alpha=0.7, label="Test images")
    ax.plot(xs, xs, "g--", label="Ideal (y = x)")
    ax.plot(xs, slope * xs + intercept, "r-", label=f"Regression line (y = {slope:.2f}x + {intercept:.1f})")
    ax.set_xlabel("Actual crowd count")
    ax.set_ylabel("Predicted crowd count")
    ax.set_title(f"Predicted vs Actual  |  MAE {mae:.1f}, MSE {mse:.1f}, R² {r2:.3f}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
 
 
if __name__ == "__main__":
    main()
