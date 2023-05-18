""" 
    Functions for loading data and some data re-arrangement

"""


import numpy as np
import os
from Driving import NavigationDemoLogParser
from Driving.NavigationDemoLogParser import NavigationDemoLogParser


def load_demonstrations_file(filename):
    """ Load a .npz file containing a list of demonstrations.
        Returns a list of trajectories, each of which is a tuple containing a
        len_trajectory x num_states state matrix and a len_trajectory x num_controls control matrix. 
    """
    demonstrations = np.load(filename, allow_pickle=True)['demonstrations']
    return demonstrations

def load_scenario_label(filename):
    """ 
    """
    data = np.load(filename)
    timestamps = data['timestamps']
    labels = data['labels']
    return timestamps, labels

def load_centerlines(filename):
    """ 
    """
    data = np.load(filename)
    return data['lines']

def load_log_data(filenames):
    """
        Helper function for loading data.
        Returns a list of NavigationDemoLogParser objects corresponding to each
        file in filenames.  
    """
    # Load your simulation data using the NavigationDemoLogParser
    navigation_parsers = [NavigationDemoLogParser(filename, autoParse=True) for filename in filenames]
    return navigation_parsers


def load_brain_data(filenames):
    """
        Return a list of matrices corresponding to the loaded brain data.
    """
    # load data and features
    brain_data = [np.load(filename) for filename in filenames]
    return brain_data

def load_splits(filenames):
    """ 
        Loads and returns matrices corresponding to the predictions from each of Tianjiao's feature spaces. 
        The split is shaped as [feature space, TR, voxel] 
    """
    splits = [np.load(filename) for filename in filenames]
    return splits

def get_splits_metadata():
    index_to_labels = ['Motion-Energy uncorrected', 'Motion-Energy recentered', 'Eyetracking',
                       'Depth', 'Affordance', 'Prompt Semantics Frame', 'Prompt Semantics Gaze',
                       'Head Direction', 'Controls', 'Turns Space Phase', 'Turns Time Phase', 'Future Path',
                       'Route Time Phase', 'Route Space Phase', 'Destination Vector Log', 'Destination-anchored Vector',
                       'Destination Grid Representation', 'Path Distance Remaining', 'Path Distance Elapsed',
                       'Beeline Distance Remaining', 'Beeline Distance Elapsed', 'Path Integration Egocentric', 
                       'Path Integration Allocentric', 'Grid Cells', 'Road Graph', 'Vehicles Spatial', 
                       'Pedestrians Spatial', 'Spatial Semantics', 'Route Time Binned', 'Route Space Binned', 
                       'Gaze Grid', 'Gaze Direction', 'Scene Structure', 'Place Fields']

    labels_to_index = {label: index for index, label in enumerate(index_to_labels)}

    fixation_related = ['Gaze Grid', 'Eyetracking']
    low_level_vision = ['Motion-Energy uncorrected', 'Motion-Energy recentered']
    high_level_vision = ['Affordance', 'Spatial Semantics', 'Prompt Semantics Frame', 'Prompt Semantics Gaze', 'Scene Structure', 'Depth']
    motor = ['Controls']
    motor_plans = ['Turns Space Phase', 'Turns Time Phase', 'Future Path']
    goal_related = ['Destination-anchored Vector', 'Destination Grid Representation', 'Destination Vector Log']
    path_integration = ['Path Integration Allocentric', 'Path Integration Egocentric', 'Beeline Distance Elapsed', 'Path Distance Elapsed']
    tracking_progress = ['Beeline Distance Remaining', 'Path Distance Remaining',
                         'Route Time Binned', 'Route Space Binned', 'Route Time Phase', 'Route Space Phase']
    cognitive_map = ['Place Fields', 'Pedestrians Spatial', 'Vehicles Spatial', 'Road Graph', 'Gaze Direction', 'Head Direction', 'Grid Cells']

    category_to_feature_spaces = {'fixation_related': fixation_related, 'low_level_vision': low_level_vision, 'high_level_vision': high_level_vision, 
                                  'motor': motor, 'motor_plans': motor_plans, 'goal_related': goal_related, 'path_integration': path_integration, 
                                  'tracking_progress': tracking_progress, 'cognitive_map': cognitive_map}

    return index_to_labels, labels_to_index, category_to_feature_spaces

def filter_brain_data(ids, data_path, categories_to_remove, out_path):
    index_to_labels, labels_to_index, category_to_feature_spaces = get_splits_metadata()
    feature_spaces = [element for category in categories_to_remove for element in category_to_feature_spaces[category]]

    for id in ids:
        print("starting id: ", id)
        # Load the brain data
        brain_data_filename = os.path.join(data_path, f'{id}_combined data.npy')
        brain_data = load_brain_data([brain_data_filename])[0]

        # Load the splits
        split_filename = os.path.join(data_path, f'{id}_split.npy')
        split = load_splits([split_filename])[0]

        # remove NaNs from split and replace with 0
        split = np.nan_to_num(split)

        print("finished loading")

        for feature_space in feature_spaces:
            print("removing ", feature_space)
            index = labels_to_index[feature_space]
            brain_data = brain_data - split[index, :, :]

        np.save(os.path.join(out_path, f'{id}_combined data_filtered_removed_{categories_to_remove}.npy'), brain_data)

#filter_brain_data(all_ids, "./Data/BrainData/", ["fixation_related", "low_level_vision", "high_level_vision", "motor", "motor_plans"], "./Data/temp/")