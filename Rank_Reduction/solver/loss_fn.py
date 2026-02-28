import torch
import torch.nn as nn
import numpy as np


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0, l1_grad = 1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.l1_grad = l1_grad

    def forward(self, y_pred, y_true):
        sq_error = 0.5*((y_true - y_pred)**2).mean(dim=1) # B, C
        abs_error = self.l1_grad*torch.abs(y_true - y_pred).mean(dim=1)  # B, C

        B, C = sq_error.shape
        n_ = B*C

        switch_point_offset = 0.5*self.delta**2 - self.l1_grad*self.delta

        mask_ = abs_error < self.delta
        loss = sq_error[mask_].sum() + (abs_error[~mask_] + switch_point_offset).sum()
        return loss/n_
    

class MedianSqLoss(nn.Module):
    def __init__(self):
        super(MedianSqLoss, self).__init__()

    def forward(self, y_pred, y_true):
        sq_error = ((y_true - y_pred)**2).mean(dim=1) # B, C

        median_batch_sq = torch.median(sq_error, dim=0)[0]
        #print(median_batch_sq)

        return torch.mean(median_batch_sq)
    

class lowerQSqLoss(nn.Module):
    def __init__(self, quantile=0.75):
        super(lowerQSqLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        sq_error = ((y_true - y_pred)**2).mean(dim=1) # B, C
        B, _= sq_error.shape
        idx_bound = int(np.round(B*self.quantile))

        sorted_sq_err, _ = torch.sort(sq_error, dim=0)


        return torch.mean(sorted_sq_err[:idx_bound])
    

class lowerQSqLossCosineAnneal(nn.Module):
    def __init__(self, ending_quantile = 0.1, total_steps=100):
        super(lowerQSqLossCosineAnneal, self).__init__()
        self.ending_quantile = ending_quantile
        self.total_steps=total_steps
        self.current_step = 0
        self.quantile = 1.0

    def forward(self, y_pred, y_true):
        sq_error = ((y_true - y_pred)**2).mean(dim=1) # B, C
        B, _= sq_error.shape
        idx_bound = int(np.round(B*self.quantile))

        sorted_sq_err, _ = torch.sort(sq_error, dim=0)


        return torch.mean(sorted_sq_err[:idx_bound])
    
    def step(self):
        self.current_step+=1
        cos_anneal = 0.5 * (1 + np.cos(np.pi * self.current_step / self.total_steps))
        self.quantile = self.ending_quantile + (1.0-self.ending_quantile)*cos_anneal


class MSElossWithNuclearNorm(nn.Module):
    def __init__(self, lambda_):
        super(MSElossWithNuclearNorm, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true, model):
        loss_mse = self.mse(y_pred, y_true)
        loss_rank = torch.norm(model.linear.weight, p='nuc')

        return loss_mse + loss_rank*self.lambda_
        


class MSElossWithL1(nn.Module):
    def __init__(self, lambda_):
        super(MSElossWithL1, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true, model):
        loss_mse = self.mse(y_pred, y_true)
        loss_rank = torch.norm(model.linear.weight, p=1)

        return loss_mse + loss_rank*self.lambda_