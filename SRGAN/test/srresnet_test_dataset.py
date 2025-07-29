import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SRResNetTestDataset(Dataset):
    def __init__(self,img_folder,hr_size,scale=4):
        self.scale = scale
        self.img_folder = img_folder
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform_hr = transforms.Compose(
            [
                transforms.Resize((self.hr_size, self.hr_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform_lr = transforms.Compose(
            [
                transforms.Resize(
                    self.lr_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_folder, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")

        hr_image = self.transform_hr(image)
        lr_image = self.transform_lr(hr_image)

        return {
            "hr": hr_image,
            "lr": lr_image,
        }
        