import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import tqdm
import numpy as np

from datasets import load_dataset
from ..model.srresnet import SRResNet
from ..dataset.srresnet import SRResNetDataset
from ..loss.srresnet_loss import srresnet_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = srresnet_loss()

train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train[:8%]")
val_dataset = load_dataset("ILSVRC/imagenet-1k", split="train[8%:10%]")
test_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation[:1%]")

train_ds = SRResNetDataset(train_dataset, scale=4)
val_ds = SRResNetDataset(val_dataset, scale=4)
test_ds = SRResNetDataset(test_dataset, scale=4)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

wandb.init(
    project="SRGAN",
    name="SRResNet_Training",
    config={
        "learning_rate": 1e-4,
        "batch_size": 16,
        "num_epochs": 100,
        "scale_factor": 4,
        "loss_fn": "MSE Loss",
        "optimizer": "Adam",
        "model_name": "SRResNet",
        "dataset": "ILSVRC/imagenet-1k",
        "train_size": "8%",
        "val_size": "2%",
        "test_size": "1%",
    },
)


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)

            sr_images = model(lr_images)
            batch_loss = criterion(sr_images, hr_images)
            val_loss += batch_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        progress_bar = tqdm.tqdm(test_loader, desc="Testing", unit="batch")
        for batch in progress_bar:
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)

            sr_images = model(lr_images)
            batch_loss = criterion(sr_images, hr_images)
            test_loss += batch_loss.item()

            progress_bar.set_postfix(test_loss=f"{batch_loss.item():.4f}")

    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    model.train()
    best_val_loss = float("inf")

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
            optimizer.step()
            train_loss += batch_loss.item()

            progress_bar.set_postfix(
                train_loss=f"{batch_loss.item():.4f}",
                avg_train_loss=f"{train_loss/(i+1):.4f}",
            )

        avg_train_loss = train_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion)

        wandb.log(
            {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss}
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_srresnet.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"srresnet_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "srresnet_final.pth")


if __name__ == "__main__":
    num_epochs = 100
    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, criterion, num_epochs)

    print("Loading best model for testing...")
    model.load_state_dict(torch.load("best_srresnet.pth"))
    test_loss = test(model, test_loader, criterion)

    wandb.log({"test_loss": test_loss})
    print(f"Test Loss: {test_loss:.4f}")

    wandb.finish()
    print("Training and testing complete.")
