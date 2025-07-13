import torch
import torch.nn as nn

def srresnet_loss():
    loss = nn.MSELoss()
    return loss
