import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset

from model.srresnet import SRResNet
from dataset.srresnet import SRResNetDataset
from loss.srresnet_loss import srresnet_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = srresnet_loss()

dataset = load_dataset("")


