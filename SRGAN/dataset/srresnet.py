from torchvision import transforms
from torch.utils.data import Dataset

class SRResNetDataset(Dataset):
    def __init__(self,hf_dataset,hr_size=96,scale=4):
        self.hf_dataset = hf_dataset
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.scale = scale
        self.transform_hr = transforms.Compose([
            transforms.RandomCrop(hr_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ])
        
        self.transform_lr = transforms.Compose([
            transforms.Resize(self.lr_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.hf_dataset) 

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example['image']
        
        hr_image = self.transform_hr(image)
        lr_image = self.transform_lr(transforms.ToPILImage()(hr_image))
        
        return {
            'lr': lr_image,
            'hr': hr_image
        }
    
    