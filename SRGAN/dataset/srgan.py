from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random


class SRGANDataset(Dataset):
    def __init__(self, hf_dataset, hr_size=128, scale=4, train_mode=True):
        self.hf_dataset = hf_dataset
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.train = train_mode

        if self.train:
            self.hr_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(hr_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # [0,1]
                ]
            )
        else:
            self.hr_transform = transforms.Compose(
                [
                    transforms.CenterCrop(min(hf_dataset[0]["image"].size)),
                    transforms.Resize((hr_size, hr_size)),
                    transforms.ToTensor(),
                ]
            )

        # LR is produced from PIL HR → Resize → ToTensor
        self.lr_down = transforms.Compose(
            [
                transforms.Resize(
                    self.lr_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        pil = self.hf_dataset[idx]["image"].convert("RGB")
        hr = self.hr_transform(pil)  # tensor 3×H×W  in [0,1]

        # build LR from the PIL (NOT from tensor) to avoid aliasing
        lr = self.lr_down(pil)

        # for square crop consistency during training
        if self.train and (lr.shape[-2:] != (self.lr_size, self.lr_size)):
            lr = transforms.functional.center_crop(lr, self.lr_size)

        return {"lr": lr, "hr": hr}
