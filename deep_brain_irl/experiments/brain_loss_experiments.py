# Core imports
import os

# Installed imports 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Local imports 
from deep_brain_irl.src.utils.data_loading import *
from deep_brain_irl.src.utils.experiment_utils import *
from deep_brain_irl.src.utils.visualization import *


print("Using interactive backend: ", mpl.is_interactive())
# Set it off for this so not getting popups all the time
if mpl.is_interactive():
    plt.ioff()

mpl.use('Agg')



def run_brain_weight_experiment(brain_weights, n_trials=10, output_dir="./Output/brain_weight_experiment/"):
    k_closest_vehicles = 3
    k_closest_pedestrians = 0
    brain_regions = ["IPS", "PPA", "RSC"] 
    perform_pca = True
    n_features_brain_pca = 5
    include_brain_data = True
    use_filtered_brain_data = True

    delays = [0]
    n_steps_ahead = [15, 30, 45, 60, 75, 90]
    brain_lookahead = [0, 30, 60, 90]
    n_epochs = 10
    batch_size = 64
    hidden_layer_sizes = [32, 32]
    learning_rate = 0.0002
    animate = False
    visualize_snapshots = False


    # Load the IDs
    train_ids = np.load(f"./Data/Demonstrations/train_ids_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npy")
    validation_ids = np.load(f"./Data/Demonstrations/validation_ids_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npy")
    test_ids = np.load(f"./Data/Demonstrations/test_ids_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npy")
    all_ids = np.concatenate((train_ids, validation_ids, test_ids))

    # Load the parsers
    parser_filenames = [f"./Data/LogAndLabelData/{id}_positions.xml" for id in all_ids]
    parsers = load_log_data(parser_filenames)

    # Put the parsers in a dictionary mapping from run_id to parser
    parser_dict = {}
    for i in range(len(all_ids)):
        print("loading parser: ", all_ids[i])
        parser_dict[all_ids[i]] = parsers[i]

    # Setup some lists to track the data from the expeirment
    min_train_losses = np.zeros((n_trials, len(brain_weights)))
    min_validation_losses = np.zeros((n_trials, len(brain_weights)))

    train_mean_errors = np.zeros((n_trials, len(brain_weights)))
    validation_mean_errors = np.zeros((n_trials, len(brain_weights)))

    for i in range(n_trials):
        print("Trial: ", i)
        for j, brain_weight in enumerate(brain_weights):
            print("Trial ", i, " Brain Weight: ", brain_weight, " percent done: ", round(100 * (i * len(brain_weights) + j) / (n_trials * len(brain_weights)), 2), "%")
            # TODO: split off the data loading from the training so that can have this be much more efficient. 
            train_losses, validation_losses, loss_labels, train_errors, validation_errors = train_evaluate_configuration(k_closest_vehicles, k_closest_pedestrians, 
                                                                                                        brain_regions, perform_pca, n_features_brain_pca, include_brain_data, use_filtered_brain_data, brain_weight, 
                                                                                                        delays, n_steps_ahead, brain_lookahead, 
                                                                                                        n_epochs, batch_size, hidden_layer_sizes, learning_rate, 
                                                                                                        parser_dict=parser_dict, animate=animate, visualize_snapshots=visualize_snapshots)

            min_train_losses[i, j] = np.min(train_losses[:, 0]) # just the state prediction loss, not the combined loss
            min_validation_losses[i, j] = np.min(validation_losses[:, 0]) # just the state prediction loss, not the combined loss

            train_mean_errors[i, j] = np.mean(train_errors) # closed loop rollout
            validation_mean_errors[i, j] = np.mean(validation_errors) # closed loop rollout

    # Plots of the minimum losses vs. brain weight with error bars
    scatter_means_with_error_bars(min_train_losses, 0.9, brain_weights, "Minimum Prediction Training Loss vs. Regularization Weight", xlabel="Weight", ylabel="Minimum Training Loss", savefig=True, filename=os.path.join(output_dir, "brain_weight_exp_min_train_losses.png"))
    scatter_means_with_error_bars(min_validation_losses, 0.9, brain_weights, "Minimum Prediction Validation Loss vs. Regularization Weight", xlabel="Weight", ylabel="Minimum Validation Loss", savefig=True, filename=os.path.join(output_dir, "brain_weight_exp_min_validation_losses.png"))
    
    # Plots of closed loop performance. 
    scatter_means_with_error_bars(train_mean_errors, 0.9, brain_weights, "Closed Loop Training Error vs. Regularization Weight", xlabel="Weight", ylabel="Closed Loop Training Error", savefig=True, filename=os.path.join(output_dir, "brain_weight_exp_train_closedloop_errors.png"))
    scatter_means_with_error_bars(validation_mean_errors, 0.9, brain_weights, "Closed Loop Validation Error vs. Regularization Weight", xlabel="Weight", ylabel="Closed Loop Validation Error", savefig=True, filename=os.path.join(output_dir, "brain_weight_exp_validation_closedloop_errors.png"))


    # Save min_train_losses, min_validation_losses, train_mean_errors, validation_mean_errors to file
    np.save(os.path.join(output_dir, "min_train_losses.npy"), min_train_losses)
    np.save(os.path.join(output_dir, "min_validation_losses.npy"), min_validation_losses)
    np.save(os.path.join(output_dir, "train_mean_errors.npy"), train_mean_errors)
    np.save(os.path.join(output_dir, "validation_mean_errors.npy"), validation_mean_errors)

    # Return the data we've collected 
    return min_train_losses, min_validation_losses, train_mean_errors, validation_mean_errors

# Create log-spaced brain weights from 1e-3 to 1e3
brain_weights = np.logspace(-3, 3, 2)
n_trials = 2
min_train_losses, min_validation_losses, train_mean_errors, validation_mean_errors = run_brain_weight_experiment(brain_weights, n_trials=n_trials, output_dir="./Output/brain_weight_experiment/")