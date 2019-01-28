import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def one_hot(label, num_classes):
    """One-hot encoding: transform label(N, 1, D, H, W) to one-hot form(N, num_classes, D, H, W).
    """
    N, C, D, H, W = label.size()
    label = label.long().to(torch.device('cpu'))
    return torch.zeros((N, num_classes, D, H, W)).scatter_(1, label, 1).to(torch.device('cuda'))

class BCE_loss(nn.BCELoss):
    def __init__(self):
        super(BCE_loss, self).__init__()
        
    def forward(self, output, target):
        """
        output is a torch variable of size B x C x D x H x W representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the output
        """
        self.num_classes = output.size(1)
        target = one_hot(target, self.num_classes)
        return super(BCE_loss, self).forward(output, target)

class Dice_loss(nn.Module):
    """Dice loss
    The implementation of dice loss in V-Net
    """
    def __init__(self):
        super(Dice_loss, self).__init__()
      
    def forward(self, output, target):
        """
        output is a torch variable of size B x C x D x H x W representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the output
        """
        self.num_classes = output.size(1)
        target = one_hot(target, self.num_classes)
        reduce_axes = (2, 3, 4)
        dice_numerator = 2.0 * (output * target).sum(reduce_axes)
        dice_denominator = (output ** 2).sum(reduce_axes) + (target ** 2).sum(reduce_axes)
        smooth = 1e-5 
        dice_loss = dice_numerator / (dice_denominator + smooth)
        return 1 - torch.mean(dice_loss)