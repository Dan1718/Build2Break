# models/frame_detector.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import timm  # PyTorch Image Models library

class FrameDataset(Dataset):
    """
    Dataset wrapper for video frames.
    Converts frames to PyTorch tensors and normalizes them.
    """
    def __init__(self, frames):
        self.frames = frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),  # Xception input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return self.transform(frame)


class FrameDetector:
    """
    Frame-based detector using PyTorch Xception.
    Automatically picks the latest .pth checkpoint for inference.
    """
    def __init__(self, weights_dir="models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create Xception model
        self.model = timm.create_model('xception', pretrained=False, num_classes=1)
        self.model.to(self.device)
        self.model.eval()

        # Locate all .pth files
        pth_files = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError(f"No .pth weights found in {weights_dir}")

        # Sort and pick the latest version (assumes names like v0, v1, v2.1)
        def version_key(fname):
            # Extract version numbers, e.g., v2.1 -> [2,1]
            ver = fname.replace('.pth','').split('v')[-1]
            return [int(x) if x.isdigit() else float(x) for x in ver.split('.')]

        latest_file = sorted(pth_files, key=version_key)[-1]
        path = os.path.join(weights_dir, latest_file)

        # Load only the latest checkpoint
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def detect(self, frames):
        """
        Runs inference on a list of frames.
        Returns the averaged fake probability score across all frames.
        """
        if len(frames) == 0:
            return 0.0

        dataset = FrameDataset(frames)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        scores = []
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            probs = torch.sigmoid(output).view(-1)
            scores.extend(probs.cpu().numpy())

        return float(np.mean(scores))
