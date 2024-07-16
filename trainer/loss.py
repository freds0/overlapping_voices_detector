import torch.nn.functional as F
import torch

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def mse_loss(output, target):
    loss = F.mse_loss(output, target)
    print(loss)
    return loss 
