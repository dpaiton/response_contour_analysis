"""
Utility funcions for managing tensorflow models

Authors: Santiago Cadena, Dylan Paiton
"""

import numpy as np

def get_normalized_activations(model, target_neuron_ids, contour_dataset, get_activation_function, activation_function_kwargs={}):
    """
    Parameters:
        target_neuron_ids [list of ints] with shape [num_target_neurons] indicating which neuron index for activations
        contour_dataset [list of list of ndarray] with shapes [num_target_neurons][num_comparisons_per_target][num_datapoints, datapoint_length]
        get_activation_function [python function] which can be called to get the activations from a model for a given input image
        activation_function_kwargs [dict] other keyword arguments to be passed to get_activation_function()
    Returns:
        ndarray with shape [num_target_neurons, num_comparisons_per_target, num_datapoints_x, num_datapoints_y]
    """
    activations_list = []
    for target_index, neuron_index in enumerate(target_neuron_ids):
        activations_sub_list = []
        for comparison_index, datapoints in enumerate(contour_dataset[target_index]):
            if np.any(np.isnan(datapoints)):
                print('WARNING:From model_handling/get_normalized_activations: nan in contour_dataset matrix for '
                    +f'target_index={target_index}')
            activations = get_activation_function(model, np.squeeze(datapoints), neuron_index,
                **activation_function_kwargs)
            if np.any(np.isnan(activations)):
                print('WARNING:From model_handling/get_normalized_activations: nan in activity vector for '
                    +f'neuron_index={neuron_index}, comparison_index={comparison_index}')
            # renormalize activations to be between 0 & 1
            activations = activations - activations.min() # minimum = 0
            activations_max = activations.max() # maximum value must be >= 0
            if activations_max > 0.0:
                activations = activations / activations_max # maximum = 1
            else:
                print('WARNING: model_handling/get_normalized_activations: Maximum value of activations for '
                    +f'neuron_index={neuron_index}, comparison_index={comparison_index}, is {activations_max}')
            num_images = datapoints.shape[0]
            activations = activations.reshape(int(np.sqrt(num_images)), int(np.sqrt(num_images)))
            activations_sub_list.append(activations)
        activations_list.append(np.stack(activations_sub_list, axis=0))
    all_activations = np.stack(activations_list, axis=0)
    return all_activations
