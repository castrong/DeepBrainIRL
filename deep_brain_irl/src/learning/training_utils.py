# Core imports
import os

# Installed imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR


# Setup some fairly general supervised learning code
def supervised_learning(X, Y, X_val, Y_val, model, n_epochs, batch_size, train_loss, eval_losses, eval_loss_labels, lr=0.001, save_checkpoints=False, checkpoint_period=1, checkpoint_path="", save_plots=False, filename=""):
    # Create path to save checkpoints
    if save_checkpoints:
        os.makedirs(checkpoint_path, exist_ok=True)
    
    # Create your optimizer
    optimizer=optim.Adam(model.parameters(), lr=lr)

    # Setup the dataloader 
    loader = DataLoader(list(zip(X, Y)), shuffle=True, batch_size=batch_size)

    # Keep track of training and validation losses
    train_losses = np.zeros((n_epochs+1, len(eval_losses)))
    val_losses = np.zeros((n_epochs+1, len(eval_losses)))

    # Fill in the loss from before training
    train_losses[0, :] = [loss(model(X), Y).detach().numpy() for loss in eval_losses]
    val_losses[0, :] = [loss(model(X_val), Y_val).detach().numpy() for loss in eval_losses]

    # Train the model 
    for epoch in range(n_epochs):
        for X_batch, Y_batch in loader:
            
            # Predict, then calculate the loss
            Y_pred = model(X_batch)
            loss = train_loss(Y_pred, Y_batch)

            # Optimize using the gradient from that loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses[epoch+1, :] = [loss(model(X), Y).detach().numpy() for loss in eval_losses]
        val_losses[epoch+1, :] = [loss(model(X_val), Y_val).detach().numpy() for loss in eval_losses]

        print(f'Finished epoch {epoch}, latest losses {train_losses[epoch+1, :]}, validation losses {val_losses[epoch+1, :]}') 

        if save_checkpoints and epoch % checkpoint_period == 0:
            torch.save({'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()},
                        checkpoint_path + f"checkpoint_{epoch+1}.pt") # epoch+1 so that the first checkpoint is checkpoint_1.pt. it then matches with argmin in the train_losses and val_losses. conceptually it's after epoch+1 epochs.

    # Create a subfigure with len(eval_losses) subplots horizontally
    fig, axs = plt.subplots(1, len(eval_losses), figsize=(15, 5))

    for i, label in enumerate(eval_loss_labels):
        ax = axs[i]
        ax.plot(range(n_epochs+1), train_losses[:, i], label=f'Training {label}')
        ax.plot(range(n_epochs+1), val_losses[:, i], label=f'Validation {label}')
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_title(f"Training and Validation Loss vs. Epoch for {label}")

    if save_plots:
        plt.savefig(filename)

    return train_losses, val_losses
    

def find_normalization(X):
    """
        Return the mean and std to normalize a matrix by 
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return X_mean, X_std

def normalize_matrix(X, means, stdevs):
    """
        Normalize a matrix by subtracting the mean and dividing by the standard deviation
    """
    return (X - means) / stdevs

def unnormalize_matrix(X, means, stdevs):
    """
        Unnormalize a matrix by multiplying by the standard deviation and adding the mean
    """
    return X * stdevs + means