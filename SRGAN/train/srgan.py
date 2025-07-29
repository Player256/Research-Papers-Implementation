import os
import tqdm
import torch
import wandb
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from skimage.metrics import structural_similarity as ssim

from SRGAN.model.srgan import Discriminator
from SRGAN.model.srresnet import SRResNet
from SRGAN.dataset.srgan import SRGANDataset
from SRGAN.loss.srgan_loss import SRGANLoss


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = SRResNet().to(device)
generator.load_state_dict(torch.load("SRGAN/checkpoints/best_srresnet.pth"))

discriminator = Discriminator().to(device)

criterion = SRGANLoss().to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=3e-4)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)


train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train[:1%]")
val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation[:1300]")

train_ds = SRGANDataset(train_dataset, scale=4, train_mode=True)
val_ds = SRGANDataset(val_dataset, scale=4, train_mode=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

wandb.init(
    project="SRGAN",
    name="SRGAN_Training",
    config={
        "epochs": 100,
        "batch_size": 16,
        "train_split": "0.5%",
        "val_split": "0.1%",
        "loss": "VGG content + adversarial",
        "generator": "SRResNet",
    },
)


def calculate_psnr(sr, hr):
    mse = torch.nn.functional.mse_loss(sr, hr)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(sr, hr):
    sr_im = sr.permute(1, 2, 0).cpu().numpy()
    hr_im = hr.permute(1, 2, 0).cpu().numpy()
    return ssim(sr_im, hr_im, data_range=1.0, channel_axis=-1)


def validate(generator, val_loader):
    generator.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = generator(lr)
            for i in range(sr.size(0)):
                psnr = calculate_psnr(sr[i], hr[i])
                ssim_val = calculate_ssim(sr[i], hr[i])
                total_psnr += psnr
                total_ssim += ssim_val
                count += 1
    return total_psnr / count, total_ssim / count


def train(generator,discriminator,train_loader,num_epochs):
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        g_running_loss = 0.0
        d_running_loss = 0.0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            with torch.no_grad():
                fake_sr = generator(lr)

            real_pred = discriminator(hr)
            fake_pred = discriminator(fake_sr.detach())

            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)

            d_loss_real = criterion.adversarial_loss(real_pred, real_labels)
            d_loss_fake = criterion.adversarial_loss(fake_pred, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            d_optimizer.step()

            # fake_sr = generator(lr)
            fake_pred = discriminator(fake_sr)  
            g_loss, content_loss, adv_loss = criterion(
                fake_sr, hr, fake_pred, is_generator=True
            )

            g_optimizer.zero_grad()
            g_loss.backward()
            clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_optimizer.step()

            g_running_loss += g_loss.item()
            d_running_loss += d_loss.item()

            pbar.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

        avg_g_loss = g_running_loss / len(train_loader)
        avg_d_loss = d_running_loss / len(train_loader)

        wandb.log(
            {
                "epoch": epoch + 1,
                "g_loss": avg_g_loss,
                "d_loss": avg_d_loss,
            }
        )

        if (epoch + 1) % 5 == 0:
            psnr_val, ssim_val = validate(generator, val_loader)
            wandb.log({"val_psnr": psnr_val, "val_ssim": ssim_val})
            print(f"[Eval] Epoch {epoch+1} — PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

        torch.save(
            generator.state_dict(), f"SRGAN/checkpoints/GAN/generator_epoch{epoch+1}.pth"
        )
        torch.save(
            discriminator.state_dict(),
            f"SRGAN/checkpoints/GAN/discriminator_epoch{epoch+1}.pth",
        )

    wandb.finish()
    print("Training complete.")
