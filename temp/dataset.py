from pathlib import Path

from torchvision import transforms
import torch
from torch.utils.data import Dataset 
from PIL import Image 

class SRDataset(Dataset):
    def __init__(
            self,
            data_dir,
            split,
            scale: int = 2,
            max_samples: int | None = None,
            ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.scale = scale
        self.transform = transforms.ToTensor()



        lr_dir = self.data_dir / f"x{self.scale}"/ self.split
        hr_dir = self.data_dir / "x1" / self.split
        extensions = {".png", ".jpg", ".jpeg"}
        lr_images = sorted(
                p for p in lr_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in extensions 
                )

        self.pairs = []
        for lr_path in lr_images:
            relative_path = lr_path.relative_to(lr_dir)
            hr_path = hr_dir / relative_path 
            if not hr_path.is_file():
                raise FileNotFoundError(
                        f"Missing HR image for {lr_path}: {hr_path}"
                        )
            self.pairs.append((lr_path, hr_path))
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path = self.pairs[idx][0]
        hr_path = self.pairs[idx][1]
        with Image.open(lr_path) as img:
            lr_image = img.convert("RGB")
        with Image.open(hr_path) as img:
            hr_image = img.convert("RGB")
        lr_tensor = self.transform(lr_image)
        hr_tensor = self.transform(hr_image)
        return (lr_tensor, hr_tensor)
