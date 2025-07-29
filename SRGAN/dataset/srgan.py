from torchvision import transforms
from torch.utils.data import Dataset
import torch


class SRGANDataset(Dataset):
    def __init__(
        self, hf_dataset, hr_size=96, scale=4, train_mode=False, add_noise=False
    ):
        self.hf_dataset = hf_dataset
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.scale = scale
        self.train_mode = train_mode
        self.add_noise = add_noise

        if train_mode:
            self.transform_hr = transforms.Compose(
                [
                    transforms.Resize((hr_size + 24, hr_size + 24)),
                    transforms.RandomCrop(hr_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  
                ]
            )
        else:
            self.transform_hr = transforms.Compose(
                [
                    transforms.Resize((hr_size, hr_size)),
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
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example["image"].convert("RGB")

        hr_image = self.transform_hr(image)

        if self.add_noise and self.train_mode:
            noise = torch.randn_like(hr_image) * 0.01
            hr_image = torch.clamp(hr_image + noise, 0.0, 1.0)

        lr_image = self.transform_lr(hr_image)

        return {"lr": lr_image, "hr": hr_image}
