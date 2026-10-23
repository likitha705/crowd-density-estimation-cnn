import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import CSRNet

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_model(weights_path: str, device: str = "cpu") -> CSRNet:
    model = CSRNet(pretrained_frontend=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict(model: CSRNet, image: Image.Image, device: str = "cpu"):
    """Return (estimated_count, density_map as 2D numpy array)."""
    x = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    density = model(x).squeeze().cpu().numpy()
    return float(density.sum()), density


def safety_level(count: float, threshold: int):
    """Simple 3-level crowd safety rule used by the dashboard."""
    if count >= threshold:
        return "DANGER", "Overcrowded. Open exits and restrict entry."
    if count >= 0.75 * threshold:
        return "WARNING", "Crowd is close to the limit. Monitor closely."
    return "SAFE", "Crowd level is within safe limits."
