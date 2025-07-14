import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset

from ..model.srresnet import SRResNet
from ..dataset.srresnet import SRResNetDataset
from ..loss.srresnet_loss import srresnet_loss

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = srresnet_loss()

dataset = load_dataset("ILSVRC/imagenet-1k",split="train[:1%]")

train_dataset = SRResNetDataset(dataset)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

wandb.init(
    project="SRGAN",
    name="SRResNet_Training",
    config={
        "learning_rate": 1e-4,
        "batch_size": 16,
        "num_epochs": 10,
        "scale_factor": 4,
        "loss_fn": "MSE Loss",
        "optimizer": "Adam",
        "model_name": "SRResNet",
        "dataset": "ILSVRC/imagenet-1k(train[:1%])"
    }
)

def train(model,train_loader,optimizer,criterion,num_epochs):
    model.train()
    for epoch in range(num_epochs):
        loss = 0
        for i,batch in enumerate(train_loader):
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)
            optimizer.zero_grad()
            sr_images = model(lr_images)
            batch_loss = criterion(sr_images, hr_images)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        avg_loss = loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "srresnet.pth")


if __name__ == "__main__":
    num_epochs = 10
    train(model, train_loader, optimizer, criterion, num_epochs)
    wandb.finish()
    print("Training complete. Model saved as 'srresnet.pth'.")
