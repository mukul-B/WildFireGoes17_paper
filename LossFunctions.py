"""
Loss functions

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""

import torch
from torch import nn
import torch.nn.functional as F

SMOOTH = 1e-6
ACTIVATION_RELU = "relu"
ACTIVATION_SIGMOID = "sigmoid"
ACTIVATION_BOTH = "both"

# Global RMSE
class GMSE(nn.Module):
    def __init__(self, beta):
        self.last_activation = ACTIVATION_RELU
        super(GMSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        rmse = torch.sqrt(torch.mean((targets - pred) ** 2))
        return rmse


# Global plus local MSE
class GLMSE(nn.Module):
    def __init__(self, beta):
        self.last_activation = ACTIVATION_RELU
        self.beta = beta
        super(GLMSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        global_rmse = torch.sqrt(torch.mean((targets - pred) ** 2))

        a = (targets - pred) ** 2
        a = a * targets
        local_rmse = torch.sqrt(torch.sum(a) / torch.sum(targets))
        total_loss = self.beta * local_rmse + (1 - self.beta) * global_rmse
        return local_rmse, global_rmse, total_loss


# jaccard_loss : segmentation bases
class jaccard_loss(nn.Module):
    def __init__(self, beta):
        super(jaccard_loss, self).__init__()
        self.last_activation = ACTIVATION_SIGMOID

    def forward(self, pred, targets):
        # pred = pred.view(-1)
        # targets = targets.view(-1)
        targets[targets > 0] = 1
        # pred[pred >= 0.5] = 1
        # pred[pred < 0.5] = 0
        intersection = torch.sum(pred * targets, (1, 2, 3))
        sum_pred = torch.sum(pred, (1, 2, 3))
        sum_targets = torch.sum(targets, (1, 2, 3))
        loss = - torch.mean(
            (intersection + SMOOTH) / (torch.sum(sum_targets) + torch.sum(sum_pred) - intersection + SMOOTH))
        return loss


# two branch approach
class two_branch_loss(nn.Module):
    def __init__(self, beta):
        super(two_branch_loss, self).__init__()
        self.last_activation = ACTIVATION_BOTH
        self.beta = beta

    def forward(self, pred, target):
        pred_sup, pred_seg = pred[0], pred[1]

        # rmse
        pred_sup = pred_sup.view(-1)
        target_sup = target.view(-1)
        rmse_loss = torch.sqrt(torch.mean((target_sup - pred_sup) ** 2))


        # jaccard
        binary_target = target.cuda()
        binary_target[binary_target > 0] = 1

        # binary_target_sup = binary_target.view(-1)
        # # rmse_loss = torch.sqrt(torch.sum(((target_sup - pred_sup) ** 2 )* binary_target_sup) / torch.sum(binary_target_sup))
        # rmse_loss = torch.sqrt(torch.mean(((target_sup - pred_sup) ** 2 )* binary_target_sup) )
        intersection = torch.sum(pred_seg * binary_target, (1, 2, 3))
        sum_pred = torch.sum(pred_seg, (1, 2, 3))
        sum_targets = torch.sum(binary_target, (1, 2, 3))
        jaccard_loss = - torch.mean(
            (intersection + SMOOTH) / (torch.sum(sum_targets) + torch.sum(sum_pred) - intersection + SMOOTH))

        total_loss = self.beta * rmse_loss + (1 - self.beta) * jaccard_loss
        return rmse_loss, jaccard_loss, total_loss
        # return rmse_loss, rmse_loss, rmse_loss
    
# Global plus local MSE
class LRMSE(nn.Module):
    def __init__(self, beta):
        self.last_activation = ACTIVATION_RELU
        self.beta = beta
        super(LRMSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        a = (targets - pred) ** 2
        a = a * targets
        local_rmse = torch.sqrt(torch.sum(a) / torch.sum(targets))
     
        return local_rmse
