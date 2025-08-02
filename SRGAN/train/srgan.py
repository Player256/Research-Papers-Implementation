import os
import tqdm
import torch
import wandb
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from skimage.metrics import structural_similarity as ssim

from SRGAN.model.srgan import Discriminator
from SRGAN.model.srresnet import SRResNet
from SRGAN.dataset.srgan import SRGANDataset
from SRGAN.loss.srgan_loss import SRGANLoss


class Config:
    seed = 42
    epochs = 20
    batch_size = 16
    hr_size = 96
    scale = 4
    g_lr = 1e-4
    d_lr = 1e-7
    adam_betas = (0.9, 0.999)
    gradient_clip_g = 1.0
    gradient_clip_d = 5.0
    warmup_steps = 1000
    early_stopping_patience = 10
    early_stopping_min_delta = 0.0001
    train_split_percent = "train[:25%]"
    val_split_validation_percent = "validation[:12500]"
    checkpoints_dir_gan = "SRGAN/checkpoints/GAN"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.counter = 0
        self.best_weights = None
        self.restore_best_weights = restore_best_weights

    def __call__(self, current_metric, model):
        if current_metric < self.best_metric - self.min_delta:
            self.best_metric = current_metric
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict()
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights:
            model.load_state_dict(self.best_weights)
            print("Restored model to best weights based on validation metric.")


def validate(generator_model, val_data_loader, device):
    generator_model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for batch in val_data_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = generator_model(lr)
            for i in range(sr.size(0)):
                psnr = calculate_psnr(sr[i], hr[i])
                ssim_val = calculate_ssim(sr[i], hr[i])
                total_psnr += psnr
                total_ssim += ssim_val
                count += 1
    return total_psnr / count, total_ssim / count


def train_srgan():
    config = Config()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = SRResNet().to(device)

    if os.path.exists("SRGAN/checkpoints/best_srresnet.pth"):
        generator.load_state_dict(torch.load("SRGAN/checkpoints/best_srresnet.pth"))
        print("Loaded pre-trained SRResNet weights for Generator.")
    else:
        print(
            "Pre-trained SRResNet weights not found. Generator will train from scratch."
        )

    discriminator = Discriminator().to(device)

    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.g_lr, betas=config.adam_betas
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.d_lr, betas=config.adam_betas
    )
    loss_fn = SRGANLoss().to(device)

    print(
        f"Loading training data: ILSVRC/imagenet-1k, split='{config.train_split_percent}'"
    )
    train_dataset_hf = load_dataset(
        "ILSVRC/imagenet-1k", split=config.train_split_percent, streaming=False
    )
    print(
        f"Loading validation data: ILSVRC/imagenet-1k, split='{config.val_split_validation_percent}'"
    )
    val_dataset_hf = load_dataset(
        "ILSVRC/imagenet-1k", split=config.val_split_validation_percent, streaming=False
    )

    train_ds = SRGANDataset(
        train_dataset_hf, hr_size=config.hr_size, scale=config.scale, train_mode=True
    )
    val_ds = SRGANDataset(
        val_dataset_hf, hr_size=config.hr_size, scale=config.scale, train_mode=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    wandb.init(
        project="SRGAN",
        name="SRGAN_Training_CustomDataset",
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "train_split": config.train_split_percent,
            "val_split": config.val_split_validation_percent,
            "g_lr": config.g_lr,
            "d_lr": config.d_lr,
            "gradient_clip_g": config.gradient_clip_g,
            "gradient_clip_d": config.gradient_clip_d,
            "loss": "VGG content + adversarial",
            "generator": "SRResNet",
            "discriminator": "Discriminator",
            "early_stopping_patience": config.early_stopping_patience,
        },
    )

    early_stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        restore_best_weights=True,
    )

    global_step = 0
    os.makedirs(config.checkpoints_dir_gan, exist_ok=True)
    best_val_psnr = -1.0

    print("Starting SRGAN training...")
    for epoch in range(config.epochs):
        generator.train()
        discriminator.train()
        g_running_loss = 0.0
        d_running_loss = 0.0

        pbar = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch"
        )

        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            with torch.no_grad():
                fake_sr = generator(lr)

            real_pred = discriminator(hr)
            real_labels = torch.ones_like(real_pred)
            d_loss_real = loss_fn.adversarial_loss(real_pred, real_labels)

            fake_pred_d = discriminator(fake_sr.detach())
            fake_labels = torch.zeros_like(fake_pred_d)
            d_loss_fake = loss_fn.adversarial_loss(fake_pred_d, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            clip_grad_norm_(discriminator.parameters(), config.gradient_clip_d)
            d_optimizer.step()

            fake_sr_g = generator(lr)
            fake_pred_g = discriminator(fake_sr_g)
            g_total_loss, g_content_loss, g_adv_loss = loss_fn(
                fake_sr_g, hr, fake_pred_g, is_generator=True
            )

            g_optimizer.zero_grad()
            g_total_loss.backward()
            clip_grad_norm_(generator.parameters(), config.gradient_clip_g)
            g_optimizer.step()

            global_step += lr.size(0)
            g_running_loss += g_total_loss.item()
            d_running_loss += d_loss.item()

            pbar.set_postfix(
                g_loss=f"{g_total_loss.item():.4f}",
                d_loss=f"{d_loss.item():.4f}",
                g_content_loss=f"{g_content_loss.item():.4f}",
                g_adv_loss=f"{g_adv_loss.item():.4f}",
                step=global_step,
            )
            wandb.log(
                {
                    "batch/g_loss": g_total_loss.item(),
                    "batch/d_loss": d_loss.item(),
                    "batch/g_content_loss": g_content_loss.item(),
                    "batch/g_adv_loss": g_adv_loss.item(),
                    "batch/global_step": global_step,
                }
            )

        avg_g_loss = g_running_loss / len(train_loader)
        avg_d_loss = d_running_loss / len(train_loader)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/avg_g_loss": avg_g_loss,
                "train/avg_d_loss": avg_d_loss,
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            val_psnr, val_ssim = validate(generator, val_loader, device)
            wandb.log({"val/psnr": val_psnr, "val/ssim": val_ssim, "epoch": epoch + 1})
            print(
                f"[Eval] Epoch {epoch+1} — PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}"
            )

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save(
                    generator.state_dict(),
                    os.path.join(config.checkpoints_dir_gan, "best_generator.pth"),
                )
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(config.checkpoints_dir_gan, "best_discriminator.pth"),
                )
                print(f"Saved best models with PSNR: {best_val_psnr:.2f}")

            if early_stopper(-val_psnr, generator):
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        torch.save(
            generator.state_dict(),
            os.path.join(config.checkpoints_dir_gan, f"generator_epoch{epoch+1}.pth"),
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(
                config.checkpoints_dir_gan, f"discriminator_epoch{epoch+1}.pth"
            ),
        )

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train_srgan()
