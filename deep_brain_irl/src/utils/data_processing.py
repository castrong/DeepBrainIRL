# Core imports
from copy import deepcopy
import math
import matplotlib.pyplot as plt

# Installed imports
import numpy as np
from sklearn.decomposition import PCA

# Local imports
from deep_brain_irl.src.utils.delayer import Delayer

seconds_per_TR = 2.0045


def vehicle_index_to_col(index):
    return 4 + 4*index

def pedestrian_index_to_col(index, k_closest_vehicles):
    return 4 + 4*k_closest_vehicles + 4*index

def centerline_col(k_closest_vehicles, k_closest_pedestrians):
    return 4 + 4*k_closest_vehicles + 4*k_closest_pedestrians

def tr_to_raw_index(parser, TR, fps=None):
    """
        Convert TR to raw index of the other states in the parser (e.g. player positions). 
        For images, this index will need to be scaled up by the FPS of the images / FPS of the data in the parser. This can 
        be accomplished by passing in fps_image for fps.   
    """
    if fps == None:
        fps = parser.FPS

    # Note this is raw accessed by parser.playerPosition, and not by parser.GetPlayerPositions() which shifts by the firstTRFrame
    return math.floor((parser.firstTRFrame + TR * seconds_per_TR * parser.FPS) * fps/parser.FPS)

def raw_index_to_tr(parser, index, fps=None):
    """
        Convert index of the other states in the parser (e.g. player positions) into a TR. 
        Rounds down to the nearest TR.  
    """
    if fps == None:
        fps = parser.FPS

    index_parser = index * parser.FPS / fps

    start_index = parser.firstTRFrame
    seconds_from_start = (index_parser - start_index) / parser.FPS
    # Would behavior here make more sense as round or int (to round down always)? 
    # if you imagine changing the raw index, when do you want it to flip between TRs?
    return math.floor(seconds_from_start / seconds_per_TR)


def stack_demonstrations(demonstrations):
    state_matrices = [demonstration[0] for demonstration in demonstrations]
    control_matrices = [demonstration[1] for demonstration in demonstrations]
    return np.vstack(state_matrices), np.vstack(control_matrices)

def fix_centerline_direction(state_matrix, k_closest_vehicles, k_closest_pedestrians):
    """ 
        A function that will change the centerline to be in the direction of 
        the heading that the ego vehicle starts at. 
    """
    # We want to always represent us as driving up in the right lane (versus driving backwards in the left lane)
    # we'll try to do this by having which point in the line segment be first vs. second be chosen based on the 
    # current heading of the object 
    new_matrix = deepcopy(state_matrix)

    # Assert that the whole state matrix has the same centerline, which is the last 4 elements of each row
    line_col = centerline_col(k_closest_vehicles, k_closest_pedestrians)
    assert np.all(state_matrix[:, line_col:line_col+4] == state_matrix[0, line_col:line_col+4])
    centerline = [state_matrix[0, line_col:line_col+2], state_matrix[0, line_col+2:line_col+4]]
    diff = centerline[1] - centerline[0]

 
    # If the heading and the direction of the road are in opposite directions, swap them
    # and update the centerline 
    theta_0 = state_matrix[0, 2]
    heading_vector = np.array([np.cos(theta_0), np.sin(theta_0)])
    if heading_vector @ diff > 0:
        temp0, temp1 = centerline[0], centerline[1]
        centerline[0], centerline[1] = temp1, temp0
        new_matrix[:, line_col:line_col+4] = np.concatenate(centerline)

    return new_matrix



def line_frame(state_matrix, k_closest_vehicles, k_closest_pedestrians):
    """ 
        Create a new state matrix where the x, y, theta for the ego vehicle, 
        other vehicles and pedestrians are all in the 
    """
    new_matrix = deepcopy(state_matrix)

    # Assert that the whole state matrix has the same centerline, which is the last 4 elements of each row
    line_col = centerline_col(k_closest_vehicles, k_closest_pedestrians)
    assert np.all(state_matrix[:, line_col:line_col+4] == state_matrix[0, line_col:line_col+4])
    centerline = [state_matrix[0, line_col:line_col+2], state_matrix[0, line_col+2:line_col+4]]

    # Convert the ego vehicle state
    new_matrix[:, 0:4] = object_state_world_to_line_segment_frame(state_matrix[:, 0:4], centerline)

    # Convert the nearby vehicles
    for i in range(k_closest_vehicles):
        col_index = vehicle_index_to_col(i)
        cur_vehicle_state = state_matrix[:, col_index:col_index+4]
        new_matrix[:, col_index:col_index+4] = object_state_world_to_line_segment_frame(cur_vehicle_state, centerline)

    # Convert the nearby pedestrians
    for i in range(k_closest_pedestrians):
        col_index = pedestrian_index_to_col(i, k_closest_vehicles)
        cur_pedestrian_state = state_matrix[:, col_index:col_index+4]
        new_matrix[:, col_index:col_index+4] = object_state_world_to_line_segment_frame(cur_pedestrian_state, centerline)

    # Convert the line segment
    line_index = centerline_col(k_closest_vehicles, k_closest_pedestrians)
    new_matrix[:, line_index:line_index+4] = [0, 0, 0, np.linalg.norm(centerline[1] - centerline[0])]

    return new_matrix

def object_state_world_to_line_segment_frame(state_matrix, line_segment):
    """ 
        Convert an n x 4 state matrix to the frame given by a line segment.
    """
    # Get your x_hat, y_hat of the new frame from the line segment
    x1, y1 = line_segment[0]
    x2, y2 = line_segment[1]
    diff = np.array([x2 - x1, y2 - y1])

    y_hat = diff / np.linalg.norm(diff)

    # We're in a left-handed coordinate system, and going to rotate to y_hat pointing up, x_hat pointing left. 
    # this is counterclockwise, and since it is a left handed coord. system y --> x, x --> -y, same as clockwise in a right-handed coordinate system.
    # Rotate y_hat 90 degrees CCW in left-handed coord system to get x_hat 
    x_hat = np.array([y_hat[1], -y_hat[0]])

    # Store your rotation matrix 
    R = np.array([x_hat, y_hat]) # put them in as rows of the matrix

    # Find the angle of rotation as well. 
    x_hat_angle = np.arctan2(x_hat[1], x_hat[0])

    # Convert your position columns to the line frame, note origin of the line segment frame is x1, y1 of the line segment 
    new_state_matrix = np.zeros(state_matrix.shape)
    new_state_matrix[:, 0:2] = (state_matrix[:, 0:2] - np.array([x1, y1])) @ (R.T)
    new_state_matrix[:, 2] = state_matrix[:, 2] - x_hat_angle # angle measured CW from left-pointing x-axis. so amount we rotated that x-axis CCW down by --> reduction in angle. 
    new_state_matrix[:, 3] = state_matrix[:, 3]
    
    return new_state_matrix 

def object_state_line_segment_to_world_frame(state_matrix, line_segment):
    # Get your x_hat, y_hat of the new frame from the line segment
    x1, y1 = line_segment[0]
    x2, y2 = line_segment[1]
    diff = np.array([x2 - x1, y2 - y1])
    y_hat = diff / np.linalg.norm(diff)

    # Rotate y_hat 90 degrees clockwise to get x_hat 
    x_hat = np.array([y_hat[1], -y_hat[0]])

    # Store your rotation matrix 
    R = np.array([x_hat, y_hat]) # put them in as rows of the matrix
    # Find the angle of rotation as well 
    x_hat_angle = np.arctan2(x_hat[1], x_hat[0])

    # Convert your position columns to the world frame, note origin of the line segment frame is x1, y1 of the line segment 
    new_state_matrix = np.zeros(state_matrix.shape)
    new_state_matrix[:, 0:2] = (state_matrix[:, 0:2] @ R) + np.array([x1, y1])

    # Convert your angle columns to the new frame by shifting by x_hat_angle appropriately
    new_state_matrix[:, 2] = state_matrix[:, 2] + x_hat_angle
    # Your speed column won't actually need to change
    new_state_matrix[:, 3] = state_matrix[:, 3]

    return new_state_matrix


def line_frame_to_network_input(state_line_frame, k_closest_vehicles, k_closest_pedestrians, delays=[0]):
    """ 
        Given the state in a line frame, convert it to the input to the network.
        This will be: ego x, ego theta,
                      vehicle x, vehicle y - ego y, vehicle theta, vehicle speed, ...,
                      pedestrian x, pedestrian y, pedestrian theta, pedestrian speed, ...
    """

    network_input = np.zeros((state_line_frame.shape[0], 2+4*k_closest_vehicles+4*k_closest_pedestrians))
    network_input[:, 0] = state_line_frame[:, 0] # x
    network_input[:, 1] = state_line_frame[:, 2] # theta

    for i in range(k_closest_vehicles):
        network_input[:, 2+4*i] = state_line_frame[:, 4+4*i] # vehicle x 
        network_input[:, 2+4*i+1] = state_line_frame[:, 4+4*i+1] - state_line_frame[:, 1] # vehicle y relative to ego y  
        network_input[:, 2+4*i+2] = state_line_frame[:, 4+4*i+2] # vehicle theta
        network_input[:, 2+4*i+3] = state_line_frame[:, 4+4*i+3] # vehicle speed

    for i in range(k_closest_pedestrians):
        network_input[:, 2+4*k_closest_vehicles+4*i] = state_line_frame[:, 4+4*k_closest_vehicles+4*i] # pedestrian x
        network_input[:, 2+4*k_closest_vehicles+4*i+1] = state_line_frame[:, 4+4*k_closest_vehicles+4*i+1] - state_line_frame[:, 1] # pedestrian y relative to ego y
        network_input[:, 2+4*k_closest_vehicles+4*i+2] = state_line_frame[:, 4+4*k_closest_vehicles+4*i+2] # pedestrian theta
        network_input[:, 2+4*k_closest_vehicles+4*i+3] = state_line_frame[:, 4+4*k_closest_vehicles+4*i+3] # pedestrian speed

    # Do appropriate delaying on network_input
    delayer = Delayer(delays)
    delayer.fit(network_input)
    network_input = delayer.transform(network_input)

    return network_input

def network_output_to_line_frame(state_line_frame_0, state_line_frame_1, network_output):
    """ 
        Given a previous matrix of states in the line frame (state_line_frame_1), 
        and the ground truth state_line_frame from the time of the prediction, as well as
        the corresponding predictions from the network, return the new matrix of states in the line frame.

        The network output is given by the network as: delta_x, delta_y, delta_theta, delta_speed
    """
    # Update the ego state in state_line_frame_1 to correspond to the prediction
    new_state_line_frame = deepcopy(state_line_frame_1)
    new_state_line_frame[:, 0:4] = state_line_frame_1[:, 0:4] + network_output[:, 0:4]
    return new_state_line_frame


"""
    Functions for prepping data for training
"""
def prediction_input_and_output(X, prediction_steps, cols_to_predict):
    """ 
        Take in a num_samples x num_features matrix X. Return a matrix Y
        such that it's X with the first prediction_steps rows removed and only the cols_to_predict, and 
        return an X with the last prediction_steps removed, corresponding to 
        each row in X being used to predict another row in X prediction_steps later 
    """
    Y = X[prediction_steps:, cols_to_predict]
    new_X = X[:-prediction_steps, :]
    return new_X, Y


def line_frame_to_training_matrices(state_matrix_line_frame, k_closest_vehicles, k_closest_pedestrians, prediction_steps, delays=[0], include_brain_data=False, interpolated_brain_data_dict=None, brain_lookahead=np.array([0]), all_ids=None):
    """ 
        Each input output pair we'd like to be:
            input: a row from state_matrix_line_frame but without the ego y position, 
                    and converting the vehicle and pedestrian y positions to be relative to the ego y position.
                    also, remove the centerline state from the row. 

            output: the line frame delta_x, delta_y, delta_theta, and delta_speed between the row
                    in the input and a row prediction_steps later
    """

    # Pull out the appropriate parts as the input
    X_train = line_frame_to_network_input(state_matrix_line_frame, k_closest_vehicles, k_closest_pedestrians, delays=delays)

    # The label is the difference between the ego x and y now and prediction_steps later
    max_prediction = np.max(prediction_steps)
    num_points = X_train.shape[0] - max_prediction
    y_dim = 2*len(prediction_steps) + (interpolated_brain_data_dict[all_ids[0]].shape[1] * len(brain_lookahead) if include_brain_data else 0)

    if num_points < 0:
        return np.zeros((0, X_train.shape[1])), np.zeros((0, y_dim))
    
    Y_train = np.zeros((num_points, 2*len(prediction_steps)))
    for i in range(num_points):
        for j in range(len(prediction_steps)):
            Y_train[i, 2*j:2*(j+1)] = state_matrix_line_frame[i+prediction_steps[j], 0:2] - state_matrix_line_frame[i, 0:2]
            
    # Appropriately truncate X_train to account for you can't start predicting 
    # prediction_steps until you're prediction_steps in.
    X_train = X_train[0:num_points, :]

    if include_brain_data:
        # Pull out the appropriate part of the brain data for this demonstration
        id_index = round(state_matrix_line_frame[0, -1])
        # assert the last row of state_matrix_line_frame is all equal
        assert(np.all(state_matrix_line_frame[:, -1] == id_index))

        # Pull out the full set of brain data for this run 
        interpolated_brain_data = interpolated_brain_data_dict[all_ids[id_index]]       

        # Now, we're going to go through and get the brain data for the indices of the run 
        # corresponding to the demonstration that we have
        demonstration_brain_data = np.zeros((state_matrix_line_frame.shape[0], interpolated_brain_data.shape[1]))
        for i in range(state_matrix_line_frame.shape[0]):
            sample_index = round(state_matrix_line_frame[i, -2]) # the index we've stored in our dataset
            demonstration_brain_data[i, :] = interpolated_brain_data[sample_index, :]

        # create a delayer for the brain data
        brain_delayer = Delayer(-np.array(brain_lookahead)) # negative to look ahead 
        brain_delayer.fit(interpolated_brain_data)
        lookahead_brain_data = brain_delayer.transform(interpolated_brain_data) 

        # Cut off the last max_prediction rows of the brain data to match the height of Y_train. 
        lookahead_brain_data = lookahead_brain_data[0:num_points, :]

        # Add on brain data to the Y_train
        Y_train = np.hstack((Y_train, lookahead_brain_data))
    
    return X_train, Y_train



""" 
    Functions for processing the brain data.
"""
def pca(X, num_components, return_explained_variance=False, plot_variance=True, output_dir="./Output/"):
    # Actually do the fitting
    fit_pca = PCA(n_components=num_components)  
    fit_pca.fit(X) 
    
    if plot_variance:
        plt.figure()
        plt.bar(range(1,len(fit_pca.explained_variance_ratio_)+1), 100*fit_pca.explained_variance_ratio_)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance (%)")
        plt.title("Explained Variance (%) vs. Principal Component")
        print("Total percent explained variance: ", 100*np.sum(fit_pca.explained_variance_ratio_))

        plt.savefig(output_dir + "/PCA Explained Variance.png")
    return fit_pca

def restrict_to_ROIs(brain_data, ROIs):
    """ 
        Options for ROIs: 

        FEF - visual attention, eye movements
        FFA - responds to faces 
        IPS - perceptual-motor coordination (directing eye movements, reaching)
        OPA - navigating through spaces visually while walking
        PPA - scene and place recognition, as well as navigation
        RSC - potentially role in mediating between perceptual and memory
        V1 - first layer of processing visual information
        V2 - higher level / more specialized processing of visual
        V3 - higher level / more specialized processing of visual

    """
    ROIMasks = []
    for ROI in ROIs:
        ROIMasks.append(np.load(f"./Data/ROI Masks/{ROI}.npy"))
    
    # Include a voxel from any of the masks
    if len(ROIs) <= 1:
        combined_mask = ROIMasks[0]
    else:
        combined_mask = ROIMasks[0]
        for mask in ROIMasks[1:]:
            combined_mask = np.logical_or(combined_mask, mask)

    return brain_data[:, combined_mask]

def interpolate_brain_data(brain_data, parser, num_samples):
    brain_data_dim = brain_data.shape[1]
    interpolated_brain_data = np.zeros((num_samples, brain_data_dim))
    for i in range(num_samples):
        # Get the brain data from the TR before this point and the TR after
        tr = raw_index_to_tr(parser, i)
        # Fill in with zeros before the brain data actually starts
        if tr < 0:
            interpolated_brain_data[i, :] = np.zeros(brain_data_dim)
        # Fill in with 0s after the brain data actually ends
        if tr >= brain_data.shape[0] - 1:
            interpolated_brain_data[i, :] = np.zeros(brain_data_dim)
        else:
            brain_data_before = brain_data[tr, :]
            brain_data_after = brain_data[tr + 1, :]

            # find the samples corresponding to those trs
            sample_tr_before = tr_to_raw_index(parser, tr)
            sample_tr_after = tr_to_raw_index(parser, tr + 1)

            # calculate the interpolation weights using that
            weight_after = (i - sample_tr_before) / (sample_tr_after - sample_tr_before)
            weight_before = (sample_tr_after - i) / (sample_tr_after - sample_tr_before)

            # Do the interpolation
            interpolated_brain_data[i, :] = weight_before * brain_data_before + weight_after * brain_data_after
        
    return interpolated_brain_data

