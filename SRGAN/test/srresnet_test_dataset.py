from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


class SRResNetTestDataset(Dataset):
    def __init__(self, img_folder, hr_size, scale=4):
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.img_folder = img_folder
        self.img_files = [
            f
            for f in os.listdir(img_folder)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")  

        square = transforms.CenterCrop( min(Image.open(img_path).size) )  

        self.transform_hr = transforms.Compose([
            square,
            transforms.Resize((self.hr_size, self.hr_size)),
            transforms.ToTensor(),
        ])

        self.transform_lr = transforms.Compose([
            square,
            transforms.Resize(self.lr_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        
        hr_image = self.transform_hr(image)  
        lr_image = self.transform_lr(image)  

        return {"lr": lr_image, "hr": hr_image}
