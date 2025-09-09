import os
import random
import tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from skimage.metrics import structural_similarity as ssim

from datasets import load_dataset
from SRGAN.model.srresnet import SRResNet
from SRGAN.dataset.srresnet import SRResNetDataset
from SRGAN.loss.srresnet_loss import srresnet_loss

# For Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs("SRGAN/checkpoints", exist_ok=True)
os.makedirs("SRGAN/final", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4,betas=(0.9, 0.999))
criterion = srresnet_loss()

train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train[:35%]")
val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation[:10%]")

train_ds = SRResNetDataset(train_dataset, scale=4, train_mode=True)
val_ds = SRResNetDataset(val_dataset, scale=4, train_mode=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

wandb.init(
    project="SRGAN",
    name=f"SRResNet_lr{1e-4}_bs{16}",
    config={
        "learning_rate": 1e-4,
        "batch_size": 16,
        "num_epochs": 20,
        "scale_factor": 4,
        "loss_fn": "MSE Loss",
        "optimizer": "Adam",
        "model_name": "SRResNet",
        "dataset": "ILSVRC/imagenet-1k",
        "train_size": "25%",
        "val_size": "10%",
    },
)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def calculate_psnr(sr, hr):
    mse = nn.functional.mse_loss(sr, hr)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr.item()


def calculate_ssim(sr, hr):
    sr_im = sr.permute(1, 2, 0).cpu().numpy()
    hr_im = hr.permute(1, 2, 0).cpu().numpy()
    ssim_value = ssim(sr_im, hr_im, data_range=1.0, channel_axis=-1)
    return ssim_value


def validate(model, val_loader, criterion):
    model.eval()
    total_mse_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)

            sr_images = model(lr_images)
            batch_loss = criterion(sr_images, hr_images)
            total_mse_loss += batch_loss.item()

            for i in range(sr_images.size(0)):
                sr_image = sr_images[i]
                hr_image = hr_images[i]

                psnr = calculate_psnr(sr_image, hr_image)
                ssim_value = calculate_ssim(sr_image, hr_image)
                count += 1
                total_psnr += psnr
                total_ssim += ssim_value

    avg_val_loss = total_mse_loss / len(val_loader)
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_val_loss, avg_psnr, avg_ssim


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    model.train()
    best_val_loss = float("inf")
    early_stopper = EarlyStopping(patience=5, min_delta=0.0001)

    for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        train_loss = 0

        progress_bar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            unit="batch",
            leave=False,
        )

        for i, batch in enumerate(progress_bar):
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)

            optimizer.zero_grad()
            sr_images = model(lr_images)
            batch_loss = criterion(sr_images, hr_images)
            batch_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += batch_loss.item()

            progress_bar.set_postfix(
                train_loss=f"{batch_loss.item():.4f}",
                avg_train_loss=f"{train_loss/(i+1):.4f}",
            )

        avg_train_loss = train_loss / len(train_loader)

        if (epoch + 1) % 2 == 0:
            mse, psnr_val, ssim_val = validate(model, val_loader, criterion)

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "val_loss(MSE)": mse,
                    "val_psnr": psnr_val,
                    "val_ssim": ssim_val,
                    "train_loss": avg_train_loss,
                }
            )

            progress_bar.set_postfix(
                val_loss=f"{mse:.4f}",
                val_psnr=f"{ssim_val:.4f}",
                val_ssim=f"{ssim_val:.4f}",
            )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss(MSE): {avg_train_loss:.4f}, Val Loss: {mse:.4f}"
            )

            torch.save(
                model.state_dict(), f"SRGAN/checkpoints/srresnet_epoch_{epoch+1}.pth"
            )

            if mse < best_val_loss:
                best_val_loss = mse
                torch.save(model.state_dict(), "SRGAN/checkpoints/best_srresnet.pth")

            if early_stopper.step(mse):
                print(f"Early stopping triggered at epoch {epoch+1}.")
                torch.save(
                    model.state_dict(), "SRGAN/checkpoints/early_stopped_srresnet.pth"
                )
                break

        else:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")


if __name__ == "__main__":
    num_epochs = 20
    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, criterion, num_epochs)
    wandb.finish()
    print("Training and testing complete.")
