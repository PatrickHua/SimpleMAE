import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    def __init__(self, ratio=0.75, n_patch=(224/16)**2, random='uniform'):
        super().__init__()

        self.n_mask = int(ratio*n_patch)      
        assert random == 'uniform', "Haven't implement it yet!"
    
    def forward(self, x):
        T, N, D = x.shape
        rand_idx = torch.rand(T, N, device=x.device).argsort(dim=0)
        masked_idx, unmasked_idx = rand_idx.split(self.n_mask)
        # sort_idx = torch.argsort(rand_idx)
        # shuffled_x = x[rand_idx]

        # return x.t()[torch.argsort(new_idx)].t()

class MAE(nn.Module):
    def __init__(self, encoder, decoder, mask) -> None:
        super().__init__()
        self.create_mask = MaskGenerator(**mask)
        
        
    
    def forward(self, x):
        pass


















