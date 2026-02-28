import torch
import numpy as np

class WeightDecayOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, l1_decay=0.0, l2_decay=0.0, **kwargs):
        """
        Custom optimizer that applies L1 and L2 regularization (weight decay).
        
        Args:
            base_optimizer (Optimizer): An instance of a PyTorch optimizer (e.g., SGD, Adam).
            l1_decay (float): Coefficient for L1 regularization (default: 0.0).
            l2_decay (float): Coefficient for L2 regularization (default: 0.0).
        """
        if l1_decay < 0.0 or l2_decay < 0.0:
            raise ValueError("L1 and L2 decay coefficients must be non-negative")
        kwargs.pop("weight_decay", None)

        defaults = dict(l1_decay=l1_decay, l2_decay = l2_decay, **kwargs)
        super(WeightDecayOptimizer, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with L1 and L2 weight decay.
        """
        
        # Apply weight decay manually
        for group in self.base_optimizer.param_groups:
            lr_ = group["lr"]
            l1_decay_ = group['l1_decay']
            l2_decay_ = group['l2_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                if l1_decay_> 0:
                    p.add_(lr_*l1_decay_*torch.sign(p))
                
                # Apply L2 regularization (w^2)
                if l2_decay_ > 0:
                    p.add_(lr_*l2_decay_*p)

        # Perform the base optimizer's step
        self.base_optimizer.step(closure)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups