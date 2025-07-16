import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        bceloss = F.binary_cross_entropy_with_logits(inputs, targets)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) * self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bceloss = alpha_t * bceloss

        loss = focal_weight * bceloss
        return loss
