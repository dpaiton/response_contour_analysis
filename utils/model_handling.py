"""
Utility funcions for managing pytorch models

Authors: Dylan Paiton, Santiago Cadena
"""

import torch
import numpy as np

def unit_activation(model, images, target_neuron, compute_grad=True):
    model.zero_grad()
    if type(images) is np.ndarray:
        numpy = True
        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        images = torch.from_numpy(images).to(device)
    else:
        numpy = False
    if compute_grad:
        activations = model(images)[:, target_neuron]
    else:
        with torch.no_grad():
            activations = model(images)[:, target_neuron]
    if numpy:
        return activations.detach().cpu().numpy()
    else:
        return activations


def unit_activation_and_gradient(model, image, target_neuron):
    if not image.requires_grad:
        image.requires_grad = True
    activations = unit_activation(model, image, target_neuron)
    grad = torch.autograd.grad(activations, image)[0]
    return activations, grad


def normalize_single_neuron_activations(single_neuron_activations):
    """
    Renormalize activations to be between 0 & 1
    Parameters:
        single_neuron_activations [np.ndarray] of shape [num_inputs]
    Returns:
        single_neuron_activations [np.ndarray] normalized version of input
    """
    single_neuron_activations = single_neuron_activations - single_neuron_activations.min() # minimum = 0
    activations_max = single_neuron_activations.max() # maximum value must be >= 0
    if activations_max > 0.0:
        single_neuron_activations = single_neuron_activations / activations_max # maximum = 1
    else:
        print('WARNING: model_handling/normalize_single_neuron_activations: Maximum value of activations '
            +f'is {activations_max}')
    return single_neuron_activations


def normalize_ensemble_activations(activations):
    """
    Renormalize activations per neuron to be between 0 & 1
    Parameters:
        activations [np.ndarray] of shape [num_inputs, num_neurons]
    Returns:
        normalized_activations [np.ndarray] normalized version of input
    """
    normalized_activations = np.zeros_like(activations)
    num_inputs, num_neurons = activations.shape
    for neuron_idx in range(num_neurons):
        normalized_activations[:, neuron_idx] = normalize_single_neuron_activations(activations[:, neuron_idx])
    return normalized_activations
    

def get_contour_dataset_activations(model, contour_dataset, target_model_ids, get_activation_function, normalize=True, activation_function_kwargs={}):
    """
    Parameters:
        target_model_ids [list of ints] with shape [num_target_neurons] indicating which neuron index for activations
        contour_dataset [list of list of ndarray] with shapes [num_target_neurons][num_comparisons_per_target][num_datapoints, datapoint_length]
        get_activation_function [python function] which can be called to get the [np.ndarray] activations from a model for a given input image
        activation_function_kwargs [dict] other keyword arguments to be passed to get_activation_function()
    Returns:
        ndarray with shape [num_target_neurons, num_target_planes, num_comparison_planes, num_datapoints_y, num_datapoints_x]
    """
    activations_list = []
    for target_index, target_dataset in enumerate(contour_dataset):
        activations_sub_list = []
        for datapoints in target_dataset:
            if np.any(np.isnan(datapoints)):
                print('WARNING:From model_handling/get_normalized_activations: nan in contour_dataset matrix for '
                    +f'target_index={target_index}')
                import IPython; IPython.embed(); raise SystemExit
            num_images = datapoints.shape[0]
            if target_model_ids is not None:
                neuron_index = target_model_ids[target_index]
                activations = get_activation_function(model, datapoints, neuron_index, **activation_function_kwargs)
                if normalize:
                    activations = normalize_single_neuron_activations(activations)
                activations = activations.reshape(1, int(np.sqrt(num_images)), int(np.sqrt(num_images)))
            else:
                activations = get_activation_function(model, datapoints, **activation_function_kwargs)
                if normalize:
                    activations = normalize_ensemble_activations(activations)
                activations = activations.reshape(activations.shape[1], int(np.sqrt(num_images)), int(np.sqrt(num_images)))
            activations_sub_list.append(activations)
        activations_list.append(np.stack(activations_sub_list, axis=1))
    all_activations = np.stack(activations_list, axis=1)
    return all_activations