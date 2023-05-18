"""
    Helper functions for visualizing and examing the learned models
"""
# Core imports
from copy import deepcopy
import os

# Installed imports 
import functools
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib as mpl
import scipy as sp
import torch
from tqdm import tqdm


# Local imports 
from deep_brain_irl.src.learning.training_utils import normalize_matrix, unnormalize_matrix
from deep_brain_irl.src.utils.data_processing import line_frame_to_network_input, vehicle_index_to_col, pedestrian_index_to_col, centerline_col 

def rollout_prediction_line_frame(state_matrix_line_frame, model, X_mean, X_stdev, Y_mean, Y_std, n_steps_ahead, delays, k_closest_vehicles, k_closest_pedestrians):
    """ 
    
    """
    # Keep track of the full states we reach in our rollout
    rollout_states = np.zeros(state_matrix_line_frame.shape)
    cur_rollout_state = deepcopy(state_matrix_line_frame[0, :])

    # Also track all the predictions for all the points 
    all_predictions = np.zeros((state_matrix_line_frame.shape[0], len(n_steps_ahead) * 2))

    max_delay = np.max(delays)

    # Go ahead and simulate forward, each time getting a prediction, then finding your next state 
    # by interpolating along the prediction but just for dt instead of n_steps_ahead * dt. 
    # It's sort of a different way of picturing doing the rollouts.  
    for i in range(state_matrix_line_frame.shape[0]):
        # Update the current rollout state
        cur_rollout_state[4:] = deepcopy(state_matrix_line_frame[i, 4:])
        rollout_states[i, :] = cur_rollout_state

        # Pull out the relevant states 
        relevant_rollout_states = rollout_states[np.maximum(i-max_delay, 0):i+1, :]

        # Get the input features for the model
        # TODO: super inefficient, this will produce an input for each of the states in the recent list
        network_input = line_frame_to_network_input(relevant_rollout_states, k_closest_vehicles, k_closest_pedestrians, delays=delays) 
        cur_input = network_input[-1, :] # 

        # normalize your input
        normalized_input = torch.tensor(normalize_matrix(cur_input, X_mean, X_stdev), dtype=torch.float32)

        # Run through the model
        network_output_normalized = model(normalized_input).detach().numpy()

        # Unnormalize the output
        network_output = unnormalize_matrix(network_output_normalized, Y_mean,  Y_std)

        # Get your delta x and y from just the first prediction 
        delta_x_1, delta_y_1 = network_output[0:2]
        first_pred_steps = n_steps_ahead[0] # assumes n_steps_ahead is ordered increasing

        # Fill in the all_predictions matrix
        cur_ego = rollout_states[i, 0:4]
        for j in range(len(n_steps_ahead)):
            all_predictions[i, j*2:(j+1)*2] = cur_ego[0:2] + network_output[j*2:(j+1)*2]

        # Find the new ego state
        ego_state_n_steps_ahead = np.zeros(4)
        ego_state_n_steps_ahead[0:2] = cur_ego[0:2] + network_output[0:2]  # update x, y 
        ego_state_n_steps_ahead[2] = np.arctan2(delta_y_1, delta_x_1) # update heading. assumes going forwards (no reversing)
        ego_state_n_steps_ahead[3] = np.linalg.norm([delta_x_1*15/first_pred_steps, delta_y_1*15/first_pred_steps]) # update speed
        
        new_ego_state = cur_ego + 1/first_pred_steps * (ego_state_n_steps_ahead - cur_ego) # only step 1/n_steps_ahead of the way there
        cur_rollout_state[0:4] = new_ego_state

    return rollout_states, all_predictions

def plot_line_segment(line, color='black', index=0, plot_index=False):
    x1, y1 = line[0]
    x2, y2 = line[1]
    plt.plot([x1, x2], [y1, y2], color, linewidth=1)
    if plot_index: 
        # write the number i next to each line
        plt.text((x1+x2)/2, (y1+y2)/2, str(index), fontsize=10, color='red')

def plot_road_segment(line, color='black'):
    road_width = 1200
    x1, y1 = line[0]
    x2, y2 = line[1]
    # The centerline 
    plot_line_segment(line, color='y-')
    # A line road_width/2 to the left of the first line
    plot_line_segment([[x1-road_width/2, y1], [x2-road_width/2, y2]], color)
    # A line road_width/2 to the right of the first line
    plot_line_segment([[x1+road_width/2, y1], [x2+road_width/2, y2]], color)

def visualize_map(centerlines):
    for i, line in enumerate(centerlines):
        plot_line_segment(line, i+1)


def visualize_prediction_line_frame(state_matrix_line_frame, model, X_mean, X_stdev, Y_mean, Y_std, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians):
    """ 
        Visualize a trajectory predicted by the model being applied repeatedly
    """
    # First find the groundtruth xs and ys in the line frame 
    xs_groundtruth = state_matrix_line_frame[:, 0]
    ys_groundtruth = state_matrix_line_frame[:, 1]

    ego_states = rollout_prediction_line_frame(state_matrix_line_frame, model, X_mean, X_stdev, Y_mean, Y_std, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians)
    
    plt.figure()
    plt.plot(xs_groundtruth, ys_groundtruth, label='Ground truth trajectory')
    plt.plot(ego_states[:, 0], ego_states[:, 1], label='Predicted trajectory', color='r', linestyle='dashed')
    plt.axis('equal')
    plt.legend()

    return ego_states


def plot_car_at_position(position, heading, color, label=""):
    ax = plt.gca()
    width = 400 
    height = 200
    box = patches.Rectangle((position[0] - width/2, position[1] - height/2), width, height, color=color, alpha=1.0)
    rotation = mpl.transforms.Affine2D().rotate_around(position[0], position[1], heading) + ax.transData
    box.set_transform(rotation)
    ax.add_patch(box)

    # plot a line representing the heading
    heading_length = 300
    heading_x = position[0] + heading_length*np.cos(heading)
    heading_y = position[1] + heading_length*np.sin(heading)
    plt.plot([position[0], heading_x], [position[1], heading_y], color='black', label=label)


def plot_pedestrian_at_position(position, color, label=""):
    ax = plt.gca() 
    radius = 200
    circle = patches.Circle((position[0], position[1]), radius=radius)
    ax.add_patch(circle)

def visualize_state(state, k_closest_vehicles, k_closest_pedestrians, cur_index=0, fov_x=10000, fov_y=10000):
    # Plot the car
    plot_car_at_position(state[0:2], state[2], 'black')

    # Plot the other vehicles 
    for i in range(k_closest_vehicles):
        vehicle_index = vehicle_index_to_col(i)
        vehicle_pos = state[vehicle_index:vehicle_index+2]
        vehicle_heading = state[vehicle_index+2]
        plot_car_at_position(vehicle_pos, vehicle_heading, 'b', "Other Vehicles")


    # Plot the pedestrians
    for i in range(k_closest_pedestrians):
        pedestrian_index = pedestrian_index_to_col(i, k_closest_vehicles)
        pedestrian_pos = state[pedestrian_index:pedestrian_index+2]
        plot_pedestrian_at_position(pedestrian_pos, 'b', 'Pedestrians')


    # Plot the road 
    road_width = 2400
    line_col = centerline_col(k_closest_vehicles, k_closest_pedestrians)
    plot_road_segment([state[line_col:line_col+2], state[line_col+2:line_col+4]])

    # Set xlim and ylim 
    ax = plt.gca()
    ax.axis('equal')
    ax.set_xlim([state[0] - fov_x/2, state[0] + fov_x/2])
    ax.set_ylim([state[1] - fov_y/2, state[1] + fov_y/2])
    ax.invert_yaxis()

    # Add in text giving the time 
    t = cur_index * 1/15
    ax.text(state[0] - 0.9*fov_x/2, state[1] + 0.9*fov_y/2, "time = " + str(round(t, 1)))

def plot_next_positions(states, k_closest_vehicles, k_closest_pedestrians, cur_index, n_steps_ahead):

    for step_index, steps in enumerate(n_steps_ahead):
        next_ego_pos = states[cur_index + steps, 0:2]
        markersize = (len(n_steps_ahead) - step_index - 1)*0.5 + 1
        plt.plot(next_ego_pos[0], next_ego_pos[1], 'ko', markersize=markersize)


        for i in range(k_closest_vehicles):
            vehicle_index = vehicle_index_to_col(i)
            vehicle_pos = states[cur_index + steps, vehicle_index:vehicle_index+2]
            plt.plot(vehicle_pos[0], vehicle_pos[1], 'bo', markersize=markersize)

        for i in range(k_closest_pedestrians):
            pedestrian_index = pedestrian_index_to_col(i, k_closest_vehicles)
            pedestrian_pos = states[cur_index + steps, pedestrian_index:pedestrian_index+2]
            plt.plot(pedestrian_pos[0], pedestrian_pos[1], 'bo', markersize=markersize)


def visualize_points_through_trajectory(state_matrix_demonstration, model, X_mean, X_stdev, Y_mean, Y_std, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians, subsample, output_dir="", start_filename=""):
    # Setup the directory for saving
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for index in range(0, state_matrix_demonstration.shape[0] - np.max(n_steps_ahead), subsample):
        plt.cla()
        # Find the predictions from this point 
        state = state_matrix_demonstration[[index], :]
        network_input = line_frame_to_network_input(state, k_closest_vehicles, k_closest_pedestrians)
        normalized_input = torch.tensor(normalize_matrix(network_input, X_mean, X_stdev), dtype=torch.float32)
        network_output_normalized = model(normalized_input).detach().numpy()
        network_output = unnormalize_matrix(network_output_normalized, Y_mean, Y_std) 
        
        # Create the plot 
        visualize_state(state[0, :], k_closest_vehicles, k_closest_pedestrians, cur_index=index)
        # Plot the next states for agents in the demonstration
        plot_next_positions(state_matrix_demonstration, k_closest_vehicles, k_closest_pedestrians, index, n_steps_ahead)

        # Plot the predicted next states for the ego vehicle
        for i in range(len(n_steps_ahead)):
            markersize = (len(n_steps_ahead) - i - 1)*0.5 + 1

            delta_x, delta_y = network_output[0, 2*i:2*(i+1)]
            plt.plot(state[0, 0] + delta_x, state[0, 1] + delta_y, 'ro', markersize=markersize)

        # Save to file 
        plt.savefig(output_dir + start_filename + "_frame_" + str(index) + ".png")


def animation_frame(frame_num, state_matrix_demonstration, state_matrix_predicted, all_predictions, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians, subsample):
    # Clear the current axis
    plt.cla()
    cur_index = frame_num*subsample
    # Plot the ground truth state (vehicle, other vehicles, etc.)
    cur_state_demonstration = state_matrix_demonstration[cur_index, :]
    visualize_state(cur_state_demonstration, k_closest_vehicles, k_closest_pedestrians, cur_index=cur_index)
    # Plot the next states for agents in the demonstration
    plot_next_positions(state_matrix_demonstration, k_closest_vehicles, k_closest_pedestrians, cur_index, n_steps_ahead)

    # Plot the predicted ego vehicle
    cur_state_prediction = state_matrix_predicted[cur_index, :]
    plot_car_at_position(cur_state_prediction[0:2], cur_state_prediction[2], 'r', 'Predicted trajectory')

    # Plot the predicted next states for the ego vehicle 
    for i in range(len(n_steps_ahead)):
        markersize = (len(n_steps_ahead) - i - 1)*0.5 + 1

        prediction = all_predictions[cur_index, 2*i:2*(i+1)]
        plt.plot(prediction[0], prediction[1], 'ro', markersize=markersize)


def animate_trajectory(state_matrix_demonstration, state_matrix_predicted, all_predictions, n_steps_ahead, k_closest_vehicles, k_closest_pedestrians, percent_to_animate=1.0, render_speedup=1.0, subsample=10, export_as_gif=False, export_as_mp4=False, filename="./Output/animation"):
    # Create path to filename if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig, axs = plt.subplots()

    original_fps = 15
    fps = render_speedup / subsample * original_fps 
    frames = int(percent_to_animate * (state_matrix_demonstration.shape[0] - np.max(n_steps_ahead)) / subsample)

    if frames <= 0:
        print("not enough time to animate (can't get ground truth next states for long enough)")
        return None

    anim_interval = 1000/fps
    anim = FuncAnimation(fig, functools.partial(animation_frame, 
                                                state_matrix_demonstration=state_matrix_demonstration, 
                                                state_matrix_predicted=state_matrix_predicted, 
                                                all_predictions=all_predictions,
                                                n_steps_ahead=n_steps_ahead,
                                                k_closest_vehicles=k_closest_vehicles, 
                                                k_closest_pedestrians=k_closest_pedestrians,
                                                subsample=subsample),  
                                                frames=frames, interval=anim_interval, repeat=True)
    
    if export_as_gif:
        anim.save(filename+".gif", writer='imagemagick', fps=1/(anim_interval / 1000))
        return None
    elif export_as_mp4:
        anim.save(filename+".mp4", fps=1/(anim_interval / 1000))
    else:
        return HTML(anim.to_jshtml()) #, HTML(anim.to_html5_video())


def error_barsize(vec, confidence):
    conf_interval_low, conf_interval_high = sp.stats.t.interval(confidence, len(vec) - 1, loc=0, scale=sp.stats.sem(vec))
    return (conf_interval_high - conf_interval_low) / 2

def scatter_means_with_error_bars(matrix, confidence, xs, title, xlabel, ylabel, savefig, filename):
    plt.figure()
    means = [np.mean(matrix[:, i]) for i in range(matrix.shape[1])]
    error_barsizes = [error_barsize(vec, confidence) for vec in matrix]
    plt.scatter(xs, means)
    plt.errorbar(xs, means, yerr=error_barsizes, fmt='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    if savefig:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)

def plot_means_with_error_bars(vecs, confidence, labels=None, title="", xlabel="", ylabel="", savefig=False, filename=""):
    plt.figure()
    xs = np.arange(len(vecs))
    means = [np.mean(vec) for vec in vecs]

    error_barsizes = [error_barsize(vec, confidence) for vec in vecs]
    plt.bar(xs, means, width=np.diff(xs)/4, yerr=error_barsizes)
    if labels is not None:
        plt.xticks(xs, labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


    if savefig:
        # If the path doesnt exist create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)


def plot_means_with_error_bars_matrix(data_matrix, confidence, labels=None, xs=None, title="", xlabel="", ylabel="", savefig=False, filename=""):
    vecs = [data_matrix[:, i] for i in range(data_matrix.shape[1])]
    plot_means_with_error_bars(vecs, confidence, labels=labels, xs=xs, title=title, xlabel=xlabel, ylabel=ylabel, savefig=savefig, filename=filename)
