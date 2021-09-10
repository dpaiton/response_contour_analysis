"""
Utility funcions for principal curvature analysis of iso-response contours

Authors: Dylan Paiton, Matthias Kümmerer
"""
import os, sys

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import response_contour_analysis.utils.dataset_generation as data_utils
import response_contour_analysis.utils.histogram_analysis as hist_funcs
import response_contour_analysis.utils.model_handling as model_utils

def vector_f(f, x, orig_shape):
    """make f operate on and return vectors"""
    x = x.reshape(orig_shape)
    value, gradient = f(x)
    return value.item(), gradient.flatten()

def sr1_hessian_iter(f, point, distance, n_points, initial_scale=1e-6, random_walk=True, learning_rate=1.0, r=1e-8, return_points=False, progress=True):
    """
    Generator for SR1 approximation of hessian. See sr1_hessian docs for more information
    """
    # We initialize with a hessian matrix with slight positive curvature
    device = point.device
    hessian_approximation = torch.eye(np.prod(point.shape), device=device) * initial_scale
    x_0 = point.flatten() # we need the data points as vectors
    f0, gradient_0 = vector_f(f, x_0, point.shape)
    x_k_minus_1 = x_0
    gradient_k_minus_1 = gradient_0
    if progress:
        gen = tqdm(range(n_points), leave=True)
    else:
        gen = range(n_points)
    for i in gen:
        x_k = x_0 + torch.randn(len(x_k_minus_1), device=device) / np.sqrt(len(x_k_minus_1)) * distance
        delta_x_k = x_k - x_k_minus_1
        f_k, gradient_k = vector_f(f, x_k, point.shape)
        y_k = gradient_k - gradient_k_minus_1
        rank_1_vector = y_k - hessian_approximation @ delta_x_k
        denominator = torch.dot(rank_1_vector, delta_x_k)
        threshold = r * torch.linalg.norm(delta_x_k) * torch.linalg.norm(rank_1_vector)
        if torch.abs(denominator) > threshold:
            with torch.no_grad():
                hessian_update = (
                    torch.outer(rank_1_vector, rank_1_vector)
                    / denominator
                )
                hessian_approximation += learning_rate * hessian_update
        if return_points:
            yield hessian_approximation, x_k.detach().cpu().numpy()
        else:
            yield hessian_approximation
        if random_walk:
            x_k_minus_1 = x_k
            gradient_k_minus_1 = gradient_k

def sr1_hessian(f, point, distance, n_points, **kwargs):
    """
    Computes SR1 approximation of hessian
    Parameters:
        f [function] returns the activation and gradient of the model for a given input
        point [pytorch tensor] single datapoint where the hessian will be computed
        n_points [int] number of points to use for the hessian approximation
        distance [float] average euclidean distance between initial point and sampled points
        kwargs:
        random_walk [bool]
            True: updates will be made along a random walk around initial point
            False: updates will always be made between inital point and random point
        r [float] tolerance level
        learning_rate [float] this is multiplied with the update variable before adding to the Hessian approximation
        initial_scale [float] this is multiplied with the identity matrix for the initial Hessian approximation
        return_points [bool]
            True: return all of the points used to approximate the hessian
            False: do not return all of the points used to approximate the hessian
        progress [bool] whether or not to include a tqdm progress bar
    """
    if kwargs['return_points']:
        output_points = []
    for sr1_output in sr1_hessian_iter(f, point, distance, int(n_points), **kwargs):
        if kwargs['return_points']:
            output_points.append(sr1_output[1])
    if kwargs['return_points']:
        return sr1_output[0], output_points
    else:
        return sr1_output

def taylor_approximation(start_point, new_point, activation, gradient, hessian):
    '''
    computes 2nd order taylor approximation of a forward function
        i.e. $$ y = f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^{T}\mathbf{H}(\mathbf{x}) \Delta \mathbf{x} $$
    Parameters:
        start_point [torch array] original point where activation, gradient, and hessian were computed
        new_point [torch array] new point where 2nd order approximation will be applied
        activation [torch array] scalar activaiton of the function at the start_point
        gradient [torch array] column  vector (shape is [input_size, 1]) first order gradient of function at start_point
        hessian [torch array] matrix  (shape is [input_size, input_size]) second order gradient of function at start_point
    Outputs:
        approx_output [torch array] second order taylor approximation of the model output
    '''
    delta_input = (new_point.flatten() - start_point.flatten())[:, None] # column  vector
    f0 = activation
    f1 = torch.matmul(gradient.T, delta_input).item()
    f2 = torch.matmul(torch.matmul(delta_input.T, hessian), delta_input).item()
    approx_output = f0 + f1 + (f2/2)
    return approx_output


def hessian_approximate_response(f, points, hessian):
    """
    Parameters:
        f [function] returns the activation and gradient of the model for a given input
        points [pytorch tensor] datapoints where the model is to be approximated
            points[0,...] should index the original datapoint.
        hessian [pytorch tensor] hessian of the model at the given point
    """
    f_0, gradient_0 = vector_f(f, points[0, ...], [1]+list(points.shape[1:]))
    approx_responses = torch.zeros(points.shape[0]).to(hessian.device)
    for stim_idx in range(points.shape[0]):
        x_k = points[stim_idx, ...][None, ...]
        approx_output = taylor_approximation(points[0, ...], x_k, f_0, gradient_0[:, None], hessian)
        approx_responses[stim_idx] = approx_output
    return approx_responses


def plane_hessian_error(model, hessian, image, abscissa, ordinate, experiment_params, verbose=False):
    '''
    TODO: allow user to specify a smaller window of images to compute error on
    '''
    plane_absissa = [data_utils.l2_normalize(abscissa)] # horizontal axes for the planes
    plane_ordinate = [data_utils.l2_normalize(ordinate)] # vertical axes for the planes
    experiment_params['normalize_activity_map'] = False
    contour_dataset, response_images, iso_curvatures, iso_fits, iso_contours = hist_funcs.polynomial_iso_response_curvature(
        model, plane_absissa, plane_ordinate, experiment_params)
    neuron_id = target_plane_id = comp_plane_id = 0
    yx_pts = (contour_dataset['y_pts'].copy(), contour_dataset['x_pts'].copy())
    proj_vects = (
        contour_dataset['proj_target_vect'][target_plane_id][comp_plane_id],
        contour_dataset['proj_comparison_vect'][target_plane_id][comp_plane_id],
        contour_dataset['proj_orth_vect'][target_plane_id][comp_plane_id],
    )
    response_image = torch.from_numpy(response_images[neuron_id, target_plane_id, comp_plane_id, ...]).to(experiment_params['device'])
    num_images_per_edge = int(np.sqrt(experiment_params['num_images']))
    cv_slope = proj_vects[1][1] / proj_vects[1][0]
    stim_images = data_utils.inject_data(
        contour_dataset['proj_matrix'][target_plane_id][comp_plane_id],
        contour_dataset['proj_datapoints'],
        experiment_params['image_scale'],
        experiment_params['data_shape']
    )
    torch_stim_images = torch.from_numpy(stim_images).to(experiment_params['device'])
    num_images_per_edge = int(np.sqrt(experiment_params['num_images']))
    #stim_images = stim_images.reshape(num_images_per_edge, num_images_per_edge, *stim_images.shape[1:])
    act_func = lambda x: model_utils.unit_activation_and_gradient(model, x, experiment_params['target_model_id'])
    activation, gradient = act_func(image)
    activation = activation.item()
    gradient = gradient.flatten()[:, None]
    approx_response_image = hessian_approximate_response(act_func, torch_stim_images, hessian)
    approx_response_image = approx_response_image.reshape(num_images_per_edge, num_images_per_edge)
    approx_error = (response_image - approx_response_image)
    if verbose:
        return (response_image, approx_response_image, stim_images, yx_pts, proj_vects, iso_curvatures)
    else:
        return approx_error


def get_shape_operator(pt_grad, pt_hess):
    device = pt_grad.device
    dtype = pt_grad.dtype
    if pt_grad.ndim == 1:
        pt_grad = pt_grad[:, None] # row vector
    normalization_factor = torch.sqrt(torch.linalg.norm(pt_grad)**2 + 1)
    identity_matrix = torch.eye(len(pt_grad), dtype=dtype).to(device)
    metric_tensor = identity_matrix + torch.matmul(pt_grad, pt_grad.T)
    shape_operator = - torch.linalg.solve(metric_tensor, pt_hess) / normalization_factor
    return shape_operator


def local_response_curvature(pt_grad, pt_hess):
    '''
    shape_operator - [M, M] dimensional array
    principal_curvatures - [M] dimensional array of curvatures in ascending order
    principal_directions - [M,M] dimensional array,
        where principal_directions[:, i] is the vector corresponding to principal_curvatures[i]
    '''
    shape_operator = get_shape_operator(pt_grad, pt_hess)
    principal_curvatures, principal_directions = torch.linalg.eig(shape_operator)
    principal_curvatures = torch.real(principal_curvatures)
    principal_directions = torch.real(principal_directions)
    sort_indices = torch.argsort(principal_curvatures, descending=True)
    return shape_operator, principal_curvatures[sort_indices], principal_directions[:, sort_indices]