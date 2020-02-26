"""
Utility funcions for managing tensorflow models

Authors: Santiago Cadena, Dylan Paiton
"""

import numpy as np

def get_normalized_activations(model, target_neuron_ids, contour_dataset, get_activation_function, activation_function_kwargs={}):
    """
    contour_dataset should have shape [num_target_neurons][num_comparisons_per_target][num_datapoints, datapoint_length]
    Parameters:
        get_activation_function [python function] which can be called to get the activations from a model for a given input image
    Returns:
        ndarray with shape [num_target_neurons, num_comparisons_per_target, num_datapoints_x, num_datapoints_y]
    TODO: allow for batch size specification
    """
    activations_list = []
    for target_index, neuron_index in enumerate(target_neuron_ids):
        activity_sub_list = []
        for comparison_index, datapoints in enumerate(contour_dataset[target_index]):
            num_images = datapoints.shape[0]
            activations = get_activation_function(model, np.squeeze(datapoints), neuron_index,
                **activation_function_kwargs)
            activity_max = np.amax(np.abs(activations))
            activations = activations / (activity_max + 0.00001)
            activations = activations.reshape(int(np.sqrt(num_images)), int(np.sqrt(num_images)))
            activity_sub_list.append(activations)
        activations_list.append(np.stack(activity_sub_list, axis=0))
    return np.stack(activations_list, axis=0)
