# Core imports 
import os

# Installed imports 
import numpy as np

# Local imports 
from deep_brain_irl.src.learning.training_utils import *
from deep_brain_irl.src.learning.networks import *
from deep_brain_irl.src.utils.data_loading import *
from deep_brain_irl.src.utils.data_processing import *
from deep_brain_irl.src.utils.experiment_utils import * 
from deep_brain_irl.src.utils.visualization import *

def train_evaluate_configuration(k_closest_vehicles, k_closest_pedestrians,
                                ROIs_to_include, perform_pca, n_features_brain_pca, include_brain_data, use_filtered_brain_data, brain_loss_weight,
                                delays, n_steps_ahead, brain_lookahead, 
                                n_epochs, batch_size, hidden_layer_sizes, learning_rate, 
                                parser_dict=None, animate=False, visualize_snapshots=False, 
                                subsample_animation=7, n_train_to_visualize = 10, n_validation_to_visualize = 10, subsample_snapshots=15):
    """ 
        A wrapper to do all the data loading, training, 
        and then simulating for a particular configuration. 
        This is meant to make running a bunch of experiments with different configurations 
        a little bit easier. 

        Should return some of our metrics of interest, and output a bunch of visualizations 
        and some data (e.g. training losses) to file in case we need them later to do more analysis.
    """

    # Create an output directory based on the parameters 
    # split into subdirectories based on the type of thing. so highest level vehicles, pedestrians, etc.
    # create a foldername with all of the parameters for now
    level_0 = "./Output/"
    level_1 = f"vehicles={k_closest_vehicles}_pedestrians={k_closest_pedestrians}"
    level_2 = f"ROIs={ROIs_to_include}_pca={perform_pca}_n_features_brain_pca={n_features_brain_pca}_include_brain_data={include_brain_data}_use_filtered_brain_data={use_filtered_brain_data}_brain_loss_weight={brain_loss_weight}"
    level_3 = f"delays={delays}_n_steps_ahead={n_steps_ahead}_brain_lookahead={brain_lookahead}"
    level_4 = f"n_epochs={n_epochs}_batch_size={batch_size}_hidden_layer_sizes={hidden_layer_sizes}_learning_rate={learning_rate}"
    output_dir = os.path.join(level_0, level_1, level_2, level_3, level_4)

    os.makedirs(output_dir, exist_ok=True)

    # Load IDs of the demonstrations to use for training, validation, and testing
    train_ids = np.load(f"./Data/Demonstrations/train_ids_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npy")
    validation_ids = np.load(f"./Data/Demonstrations/validation_ids_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npy")
    test_ids = np.load(f"./Data/Demonstrations/test_ids_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npy")
    all_ids = np.concatenate((train_ids, validation_ids, test_ids))

    # Load train and validation demonstrations and the centerlines (map information)
    train_demonstrations = load_demonstrations_file(f"./Data/Demonstrations/train_demonstrations_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npz")
    validation_demonstrations = load_demonstrations_file(f"./Data/Demonstrations/validation_demonstrations_k_closest_vehicles={k_closest_vehicles}_k_closest_pedestrians={k_closest_pedestrians}.npz")
    centerlines = load_centerlines("./Data/LogAndLabelData/map_centerlines.npz")

    # Load the brain data
    if use_filtered_brain_data:
        brain_filenames = [f"./Data/BrainData/{id}_combined data_filtered_removed_['fixation_related', 'low_level_vision', 'high_level_vision', 'motor', 'motor_plans'].npy" for id in all_ids]
    else:
        brain_filenames = [f"./Data/BrainData/{id}_combined data.npy" for id in all_ids]

    brain_data = load_brain_data(brain_filenames)

    # Put the brain data in a dictionary mapping from brain_id to brain data
    brain_data_dict = {}
    for i in range(len(all_ids)):
        brain_data_dict[all_ids[i]] = brain_data[i]

    # Restrict to the desired ROIs
    for id in brain_data_dict.keys():
        brain_data_dict[id] = restrict_to_ROIs(brain_data_dict[id], ROIs_to_include)

    brain_data_dim = brain_data_dict[all_ids[0]].shape[1]

    # Load the parsers 
    if parser_dict is None:
        parser_filenames = [f"./Data/LogAndLabelData/{id}_positions.xml" for id in all_ids]
        parsers = load_log_data(parser_filenames)

        # Put the parsers in a dictionary mapping from run_id to parser
        parser_dict = {}
        for i in range(len(all_ids)):
            parser_dict[all_ids[i]] = parsers[i]

    # Perform PCA on the brain data 
    if perform_pca:
        # Just use the training brain data for PCA
        train_brain_data = [brain_data_dict[id] for id in train_ids]

        # Stack the training brain data to perform PCA
        stacked_train_brain_data = np.vstack(train_brain_data)

        # Now, create a dictionary representing all of the 
        # brain dat 
        pca_brain = pca(stacked_train_brain_data, n_features_brain_pca, plot_variance=True, output_dir=output_dir)
        pca_brain_dict = {}
        for id in brain_data_dict.keys():
            pca_brain_dict[id] = pca_brain.transform(brain_data_dict[id])

        brain_data_dim = n_features_brain_pca
        brain_data_dict = pca_brain_dict

    print("brain data dim: ", brain_data_dim)
    print("use pca: ", perform_pca)
    print("pca dim: ", n_features_brain_pca)

    # Pull out just the states, then convert to centerline frame 
    train_demonstration_states_world_frame = [fix_centerline_direction(demonstration[0], k_closest_vehicles, k_closest_pedestrians) for demonstration in train_demonstrations]
    validation_demonstration_states_world_frame = [fix_centerline_direction(demonstration[0], k_closest_vehicles, k_closest_pedestrians) for demonstration in validation_demonstrations]

    train_demonstration_states_line_frame = [line_frame(demonstration, k_closest_vehicles, k_closest_pedestrians) for demonstration in train_demonstration_states_world_frame]
    validation_demonstration_states_line_frame = [line_frame(demonstration, k_closest_vehicles, k_closest_pedestrians) for demonstration in validation_demonstration_states_world_frame]

    # Interpolate the brain data to match the length of the demonstrations
    interpolated_brain_data_dict = {}
    for id in brain_data_dict.keys():
        interpolated_brain_data_dict[id] = interpolate_brain_data(brain_data_dict[id], parser_dict[id], parser_dict[id].nFrames)


    # Collect your training data
    # first, pull out training pairs from each demonstration
    training_pairs = [line_frame_to_training_matrices(demonstration_line_frame, k_closest_vehicles, k_closest_pedestrians, n_steps_ahead, delays, include_brain_data=include_brain_data, interpolated_brain_data_dict=interpolated_brain_data_dict, brain_lookahead=brain_lookahead, all_ids=all_ids) for demonstration_line_frame in train_demonstration_states_line_frame]
    validation_pairs = [line_frame_to_training_matrices(demonstration_line_frame, k_closest_vehicles, k_closest_pedestrians, n_steps_ahead, delays, include_brain_data=include_brain_data, interpolated_brain_data_dict=interpolated_brain_data_dict, brain_lookahead=brain_lookahead, all_ids=all_ids) for demonstration_line_frame in validation_demonstration_states_line_frame]
        
    # Then, stack these X, Y pairs to get the final training data
    X_train = np.vstack([pair[0] for pair in training_pairs])
    Y_train = np.vstack([pair[1] for pair in training_pairs])   
    X_validation = np.vstack([pair[0] for pair in validation_pairs])
    Y_validation = np.vstack([pair[1] for pair in validation_pairs])

    # Normalize the data, both train and validation
    X_train_mean, X_train_std = find_normalization(X_train)
    Y_train_mean, Y_train_std = find_normalization(Y_train)

    X_train = normalize_matrix(X_train, X_train_mean, X_train_std)
    Y_train = normalize_matrix(Y_train, Y_train_mean, Y_train_std)

    X_validation = normalize_matrix(X_validation, X_train_mean, X_train_std)
    Y_validation = normalize_matrix(Y_validation, Y_train_mean, Y_train_std)

    # Setup training
    # we start with converting into tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_validation = torch.tensor(X_validation, dtype=torch.float32)
    Y_validation = torch.tensor(Y_validation, dtype=torch.float32)

    # Get the dimensions of the predicted states we're mapping to and
    # the brain data written down so that our layer sizes can be right in the model
    state_pred_dim = 2*len(n_steps_ahead) # copies of delta x and delta y output
    brain_dim = brain_data_dim * len(brain_lookahead) if include_brain_data else 0 # copies of brain data

    # Choose your loss(es) for training and evaluation
    brain_loss = BrainLoss(state_pred_dim, brain_dim, brain_loss_weight)
    pred_loss = PredLoss(state_pred_dim)
    if include_brain_data:
        train_loss = brain_loss 
    else:
        train_loss = pred_loss

    eval_losses = [pred_loss, brain_loss]
    labels=["State Prediction Loss", "Combined Loss"]

    model = MLP([X_train.shape[1], *hidden_layer_sizes, state_pred_dim], add_latent_to_brain=include_brain_data, brain_dim=brain_dim)
    train_losses, validation_losses = supervised_learning(X_train, Y_train, X_validation, Y_validation,
                                                          model, n_epochs, batch_size, train_loss, eval_losses, labels, 
                                                          lr=learning_rate, 
                                                          save_checkpoints=True, checkpoint_period=1, checkpoint_path=os.path.join(output_dir, "checkpoints/"), 
                                                          save_plots=True, filename=os.path.join(output_dir, "training_losses.png"))
    np.save(os.path.join(output_dir, "train_losses.npy"), train_losses)
    np.save(os.path.join(output_dir, "validation_losses.npy"), validation_losses)
    np.save(os.path.join(output_dir, "loss_labels.npy"), labels)

    # Load the checkpoint corresponding to the lowest validation loss
    checkpoint_path = os.path.join(output_dir, "checkpoints/")
    best_index = np.argmin(validation_losses[:, 0]) # best validation prediction loss
    print("loading checkpoint ", best_index)
    checkpoint = torch.load(checkpoint_path + f"checkpoint_{best_index}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # After training, turn off the brain data outputs
    model.add_latent_to_brain = False

    # Update X_train and Y_train to just have the first state_pred_dim values
    # to work with the rest of animations and stuff
    # TODO: incorporate this into the model. 
    Y_train_mean, Y_train_std = Y_train_mean[0:state_pred_dim], Y_train_std[0:state_pred_dim]


    # Now that we've learned our model, we can visualize its performance
    export_as_gif = True
    percent_to_animate=1.0
    render_speedup = 8.0

    train_anims = []
    validation_anims = []
    for i in range(n_train_to_visualize):
        demonstration_states = demonstration_states = train_demonstration_states_line_frame[i]

        if animate:
            print("train animation ", i)
            # Animate demonstrations
            rollout_prediction, all_predictions = rollout_prediction_line_frame(demonstration_states, model, X_train_mean, X_train_std, Y_train_mean, Y_train_std, n_steps_ahead, delays, k_closest_vehicles, k_closest_pedestrians)
            anim = animate_trajectory(demonstration_states, rollout_prediction, all_predictions, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians, percent_to_animate=percent_to_animate, render_speedup=render_speedup, subsample=subsample_animation, export_as_gif=export_as_gif, filename=os.path.join(output_dir, f"animations/animation_train_{i}"))
            train_anims.append(anim)

        if visualize_snapshots:
            # Also visualize points through the trajectory
            visualize_points_through_trajectory(demonstration_states, model, X_train_mean, X_train_std, Y_train_mean, Y_train_std, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians, subsample_snapshots, output_dir=os.path.join(output_dir, f"snapshots/train_{i}/"), start_filename="")

    for i in range(n_validation_to_visualize):
        demonstration_states = validation_demonstration_states_line_frame[i]

        if animate:
            print("validation animation ", i)
            # Animate demonstrations
            rollout_prediction, all_predictions = rollout_prediction_line_frame(demonstration_states, model, X_train_mean, X_train_std, Y_train_mean, Y_train_std, n_steps_ahead, delays, k_closest_vehicles, k_closest_pedestrians)
            anim = animate_trajectory(demonstration_states, rollout_prediction, all_predictions, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians, percent_to_animate=percent_to_animate, render_speedup=render_speedup, subsample=subsample_animation, export_as_gif=export_as_gif, filename=os.path.join(output_dir, f"animations/animation_validation_{i}"))
            validation_anims.append(anim)

        if visualize_snapshots:
            # Also visualize points through the trajectory
            visualize_points_through_trajectory(demonstration_states, model, X_train_mean, X_train_std, Y_train_mean, Y_train_std, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians, subsample_snapshots, output_dir=os.path.join(output_dir, f"snapshots/validation_{i}/"), start_filename="")

    # Closed loop rollout all the training and validation demonstrations 
    # and then calculate a mean squared error on the distance from the true (x, y)
    train_errors = []
    for demonstration_states in train_demonstration_states_line_frame:
        rollout_prediction, all_predictions = rollout_prediction_line_frame(demonstration_states, model, X_train_mean, X_train_std, Y_train_mean, Y_train_std, n_steps_ahead, delays, k_closest_vehicles, k_closest_pedestrians)
        error = rollout_prediction[:, 0:2] - demonstration_states[:, 0:2]
        mean_squared_error = np.mean(np.square(np.linalg.norm(error, axis=1)))
        train_errors.append(mean_squared_error)

    validation_errors = []
    for demonstration_states in validation_demonstration_states_line_frame:
        rollout_prediction, all_predictions = rollout_prediction_line_frame(demonstration_states, model, X_train_mean, X_train_std, Y_train_mean, Y_train_std, n_steps_ahead, delays, k_closest_vehicles, k_closest_pedestrians)
        error = rollout_prediction[:, 0:2] - demonstration_states[:, 0:2]
        mean_squared_error = np.mean(np.square(np.linalg.norm(error, axis=1)))
        validation_errors.append(mean_squared_error)
    
    # Save the train and validation errors
    np.save(os.path.join(output_dir, "train_closedloop_MSE.npy"), train_errors)
    np.save(os.path.join(output_dir, "validation_closedloop_MSE.npy"), validation_errors)

    # Visualize the train and validation errors on a histogram 
    bins = 20
    bins = np.histogram(np.hstack((train_errors, validation_errors)), bins=bins)[1] # hacky thing to get the bin width to be the same for both
    plt.hist(train_errors, bins=bins, alpha=0.5, label='train', density=True)
    plt.hist(validation_errors, bins=bins, alpha=0.5, label='validation', density=True)
    plt.legend(loc='upper right')
    plt.xlabel("Closed Loop MSE")
    plt.ylabel("Density")
    plt.title("Closed Loop MSE Distribution for Train and Validation")
    plt.savefig(os.path.join(output_dir, "closedloop_MSE_histogram.png"))

    # Also visualize the mean train and validation errors with error bars
    plot_means_with_error_bars([train_errors, validation_errors], 0.9, savefig=True, filename=os.path.join(output_dir, "closedloop_MSE_means.png"))

    return train_losses, validation_losses, labels, train_errors, validation_errors