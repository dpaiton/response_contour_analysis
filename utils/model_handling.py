"""
Utility funcions for managing tensorflow models

Authors: Santiago Cadena, Dylan Paiton
"""

import numpy as np


def get_activations(model, images):
    """
    Returns the activations from a model for given input images
    Parameters:
        model:  an object from the ConvNet class
        images: an array with NumImages x W x H
    Returns:
        activations: a vector of length #neurons
    """
    if len(images.shape) == 3:
        images = images[:, :, :, np.newaxis]
    activations = model.prediction.eval(session=model.session,
        feed_dict={model.images: images, model.is_training: False})
    return activations


def get_activations_cell(model, images, neuron):
    """
    Returns the activations from a model for given input images
    Parameters:
        model:  an object from the ConvNet class
        images: an array with NumImages x W x H
        neuron: int that points to the neuron index
    Returns:
        activations: a vector of length #neurons
    """
    if len(images.shape) == 3:
        images = images[:, :, :, np.newaxis]
    activations = model.prediction[:, neuron].eval(session=model.session,
        feed_dict={model.images: images, model.is_training: False})
    return activations

def get_normalized_activations(model, target_neuron_ids, contour_dataset):
    """
    contour_dataset should have shape [num_target_neurons][num_comparisons_per_target][num_datapoints, datapoint_length]
    Parameters:
    Returns:
        ndarray with shape [num_target_neurons, num_comparisons_per_target, num_datapoints_x, num_datapoints_y]
    TODO: allow for batch size specification
    """
    activations_list = []
    for target_index, neuron_index in enumerate(target_neuron_ids):
        activity_sub_list = []
        for comparison_index, datapoints in enumerate(contour_dataset[target_index]):
            num_images = datapoints.shape[0]
            activations = get_activations_cell(model, datapoints, neuron_index)
            activity_max = np.amax(np.abs(activations))
            activations = activations / (activity_max + 0.00001)
            activations = activations.reshape(int(np.sqrt(num_images)), int(np.sqrt(num_images)))
            activity_sub_list.append(activations)
        activations_list.append(np.stack(activity_sub_list, axis=0))
    return np.stack(activations_list, axis=0)