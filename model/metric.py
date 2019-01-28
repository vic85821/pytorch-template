import torch
import torch.nn as nn

class Dice_score(nn.Module):
    def __init__(self):
        super(Dice_score, self).__init__()
        self.__name__ = 'Dice'
        
    def one_hot(self, label):
        """One-hot encoding: transform label(N, 1, D, H, W) to one-hot form(N, num_classes, D, H, W).
        """
        N, C, D, H, W = label.size()
        label = label.long().to(torch.device('cpu'))
        return torch.zeros((N, self.num_classes, D, H, W)).scatter_(1, label, 1).to(torch.device('cuda'))
        
    def forward(self, output, target):
        """
        output is a torch variable of size B x C x D x H x W representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the output
        """
        self.num_classes = output.size(1)
        target = self.one_hot(target)
        output = self.one_hot(torch.argmax(output, dim=1, keepdim=True))
        reduce_axes = (2, 3, 4)
        intersection = 2.0 * (output * target).sum(reduce_axes)
        union = (output).sum(reduce_axes) + (target).sum(reduce_axes)
        smooth = 1e-5
        dice_score = intersection / (union + smooth)
        return torch.mean(dice_score, dim=0).to(torch.device('cpu')).detach().numpy()