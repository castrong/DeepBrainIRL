# Core imports
from copy import deepcopy

# Installed imports
import numpy as np 
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes, add_latent_to_brain=False, brain_dim=None, act=nn.Tanh(), output_act=nn.Identity()):
        super().__init__()
        self.act = act
        self.output_act = output_act
        self.add_latent_to_brain = add_latent_to_brain

        self.layers = []
        self.acts = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            if i < len(layer_sizes) - 2:
                self.acts.append(act)
            else:
                self.acts.append(output_act)

        self.layers = nn.ModuleList(self.layers)
        self.acts = nn.ModuleList(self.acts)

        if add_latent_to_brain:
            self.latent_to_brain_layer = nn.Linear(np.sum(layer_sizes[1:-1]), brain_dim)
            self.latent_to_brain_act = nn.Identity()

    def forward(self, x):
        zs = [x]
        for (layer, act) in zip(self.layers, self.acts):
            x = act(layer(x))
            zs.append(x)
        
        if self.add_latent_to_brain:
            all_latents = torch.hstack(zs[1:-1])
            brain_output = self.latent_to_brain_act(self.latent_to_brain_layer(all_latents))
            return torch.hstack((x, brain_output))
        else:
            return x
        

class BrainLoss(nn.Module):
    def __init__(self, pred_dim, brain_dim, brain_loss_weight):
        super(BrainLoss, self).__init__()
        self.pred_dim = pred_dim
        self.brain_dim = brain_dim
        self.brain_loss_weight = brain_loss_weight

    def forward(self, output, target):
        pred_loss = nn.MSELoss()(output[:, 0:self.pred_dim], target[:, 0:self.pred_dim])
        brain_loss = nn.MSELoss()(output[:, self.pred_dim:], target[:, self.pred_dim:])
        return pred_loss + brain_loss * self.brain_loss_weight
class PredLoss(nn.Module):
    def __init__(self, pred_dim):
        super(PredLoss, self).__init__()
        self.pred_dim = pred_dim

    def forward(self, output, target):
        return nn.MSELoss()(output[:, 0:self.pred_dim], target[:, 0:self.pred_dim])

def normal_model_from_brain_model(brain_model):
    """
        Take a model with the  
    """
    new_model = deepcopy(brain_model)
    new_model.add_latent_to_brain = False
    return new_model
