import torch
import torch.nn as nn
from torch.optim import Optimizer

class ACEShield(Optimizer):
    def __init__(self, base_optimizer, model, sigma_env=0.01, kappa=1.0):
        self.base_optimizer = base_optimizer
        self.model = model
        self.sigma_env = sigma_env
        self.kappa = kappa
        self.param_groups = base_optimizer.param_groups
        self.violation_count = 0

    def compute_fisher_trace(self):
        fisher_trace = 0.0
        total_params = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_sq = p.grad.data ** 2
                    fisher_diag_inv = 1.0 / (grad_sq + 1e-8)
                    fisher_trace += torch.sum(fisher_diag_inv).item()
                    total_params += p.numel()
        
        if total_params > 0:
            fisher_trace = fisher_trace / total_params
        
        return fisher_trace

    def step(self, closure=None):
        uncertainty = self.compute_fisher_trace()
        threshold = self.kappa * (self.sigma_env ** 2)

        if uncertainty > threshold:
            self.violation_count += 1
            projection_factor = threshold / (uncertainty + 1e-8)
            
            print(f"[ACE SHIELD] Constraint violated! Uncertainty: {uncertainty:.6f} > {threshold:.6f}")
            print(f"  Projecting gradients with factor: {projection_factor:.4f}")
            
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(projection_factor)
        
        return self.base_optimizer.step(closure)

    def zero_grad(self):
        self.base_optimizer.zero_grad()