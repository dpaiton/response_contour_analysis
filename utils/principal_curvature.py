"""
Utility funcions for principal curvature analysis of iso-response contours

Authors: Dylan Paiton, Matthias Kümmerer
"""
import os, sys

import numpy as np
from scipy.linalg import orth, null_space
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import response_contour_analysis.utils.dataset_generation as data_utils
import response_contour_analysis.utils.histogram_analysis as hist_utils
import response_contour_analysis.utils.model_handling as model_utils


def hyperboloid_graph(x_vals, y_vals, a, c):
    z_sq = c**2 * (x_vals**2 / a**2 + y_vals**2 / a**2 - 1)
    if type(x_vals) == type(torch.Tensor()):
        return torch.sqrt_(z_sq)
    else:
        return np.sqrt(z_sq)


def hyperboloid_mesh(a, c, step_size, num_points):
    pt_range = (num_points - 1) * step_size
    start_pt = 0 - pt_range / 2
    end_pt = 0 + pt_range / 2
    x_pts = np.linspace(start_pt, end_pt, num_points)
    y_pts = np.linspace(start_pt, end_pt, num_points)
    x_mesh, y_mesh = np.meshgrid(x_pts, y_pts)
    return x_mesh, y_mesh


def hyperboloid(a, c, step_size, num_points):
    """generate single sheet hyperboloid"""
    x_mesh, y_mesh = hyperboloid_mesh(a, c, step_size, num_points)
    z_vals = hyperboloid_graph(x_mesh, y_mesh, a, c)
    return x_mesh, y_mesh, z_vals


class pytorch_hyperboloid(nn.Module):
    def __init__(self, a, c):
        super(pytorch_hyperboloid, self).__init__()
        self.a = a
        self.c = c
    
    def forward(self, data):
        z_vals = hyperboloid_graph(x_vals=data[:, 0], y_vals=data[:, 1], a=self.a, c=self.c)
        return z_vals.reshape((data.shape[0], 1)) #1 neuron


def numeric_pt_grad_hess(model, x_pt, y_pt, dtype, device, sr1_kwargs):
    act_func = lambda image: model_utils.unit_activation_and_gradient(model, image, target_neuron=0)
    x_tensor = torch.tensor([x_pt, y_pt], dtype=dtype).to(device)[None, :]
    x = torch.autograd.Variable(x_tensor, requires_grad=True)
    z, pt_grad = act_func(x)
    pt_hess = sr1_hessian(act_func, x, **sr1_kwargs)
    return pt_grad, pt_hess


def numeric_grad_hess(model, x_vals, y_vals, dtype, device, sr1_kwargs):
    grad = torch.zeros(x_vals.shape + (2,), dtype=dtype).to(device)
    hess = torch.zeros(x_vals.shape + (4,), dtype=dtype).to(device)
    for x_idx in tqdm(range(len(x_vals[0, :])), leave=True):
        for y_idx in range(len(y_vals[0, :])):
            pt_grad, pt_hess = numeric_pt_grad_hess(
                model, x_vals[x_idx, y_idx], y_vals[x_idx, y_idx], dtype, device, sr1_kwargs)
            grad[x_idx, y_idx, :] = pt_grad
            hess[x_idx, y_idx, :] = pt_hess.flatten()
    return grad, hess


def autodiff_pt_grad_hess(x_pt, y_pt, a, c, dtype, device='cpu'):
    x_tensor = torch.tensor([x_pt, y_pt]).to(dtype).to(device)
    x = torch.autograd.Variable(x_tensor, requires_grad=True).to(device)
    z = hyperboloid_graph(x_vals=x[0], y_vals=x[1], a=a, c=c)
    x_1grad, = torch.autograd.grad(z, x, create_graph=True, retain_graph=True)
    x_2grad0, = torch.autograd.grad(x_1grad[0], x, create_graph=True)
    x_2grad1, = torch.autograd.grad(x_1grad[1], x, create_graph=True)
    x_2grad = torch.cat((x_2grad0, x_2grad1), dim=0)
    return x_1grad, x_2grad


def autodiff_grad_hess(x_vals, y_vals, a, c, dtype, device):
    grad = torch.zeros(x_vals.shape + (2,)).to(device)
    hess = torch.zeros(x_vals.shape + (4,)).to(device)
    for x_idx in tqdm(range(len(x_vals[0, :])), leave=True):
        for y_idx in range(len(y_vals[0, :])):
            x_1grad, x_2grad = autodiff_pt_grad_hess(x_vals[x_idx, y_idx], y_vals[x_idx, y_idx], a, c, dtype, device)
            grad[x_idx, y_idx, :] = x_1grad
            hess[x_idx, y_idx, :] = x_2grad
    return grad, hess


def hyperboloid_gauss_mean_curvature(grad, hess):
    gauss_curvature = np.zeros([len(grad), len(grad)])
    mean_curvature = np.zeros([len(grad), len(grad)])
    np_grad = grad.detach().cpu().numpy()
    for x_idx in range(len(grad)):
        for y_idx in range(len(grad)):
            if np.all(np.isfinite(np_grad[x_idx, y_idx, :])):
                pt_grad = grad[x_idx, y_idx, :]
                pt_hess = hess[x_idx, y_idx, :].reshape((2,2))
                shape_operator, principal_curvatures, principal_directions = local_response_curvature_graph(pt_grad, pt_hess)
                gauss_curvature[x_idx, y_idx] = np.prod(principal_curvatures.detach().cpu().numpy())
                mean_curvature[x_idx, y_idx] = np.mean(principal_curvatures.detach().cpu().numpy())
    return gauss_curvature, mean_curvature


def vector_f(f, x, orig_shape):
    """make f operate on and return vectors"""
    x = x.reshape(orig_shape)
    value, gradient = f(x)
    return value.item(), gradient.flatten()


def sr1_hessian_iter(f, point, distance, n_points, initial_scale=1e-6, random_walk=True, learning_rate=1.0, r=1e-8, lr_decay=False, return_points=False, progress=True):
    """
    Generator for SR1 approximation of hessian. See sr1_hessian docs for more information
    TODO: Add LR decay
    """
    # We initialize with a hessian matrix with slight positive curvature
    device = point.device
    dtype = point.type()
    hessian_approximation = (torch.eye(np.prod(point.shape), device=device) * initial_scale).type(dtype)
    x_0 = point.flatten() # we need the data points as vectors
    f0, gradient_0 = vector_f(f, x_0, point.shape)
    x_k_minus_1 = x_0
    gradient_k_minus_1 = gradient_0
    if progress:
        gen = tqdm(range(n_points), leave=True)
    else:
        gen = range(n_points)
    for i in gen:
        if lr_decay and i == n_points//2:
            learning_rate = learning_rate * 0.1
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
    """
    computes 2nd order taylor approximation of a forward function
        i.e. $$y = f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^{T}\mathbf{H}(\mathbf{x}) \Delta \mathbf{x}$$
    Parameters:
        start_point [torch array] original point where activation, gradient, and hessian were computed
        new_point [torch array] new point where 2nd order approximation will be applied
        activation [torch array] scalar activaiton of the function at the start_point
        gradient [torch array] column  vector (shape is [input_size, 1]) first order gradient of function at start_point
        hessian [torch array] matrix  (shape is [input_size, input_size]) second order gradient of function at start_point
    Outputs:
        approx_output [torch array] second order taylor approximation of the model output
    """
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
    """
    TODO: allow user to specify a smaller window of images to compute error on
    """
    plane_absissa = [data_utils.l2_normalize(abscissa)] # horizontal axes for the planes
    plane_ordinate = [data_utils.l2_normalize(ordinate)] # vertical axes for the planes
    experiment_params['normalize_activity_map'] = False
    contour_dataset, response_images, iso_curvatures, iso_fits, iso_contours = hist_utils.polynomial_iso_response_curvature(
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


def get_shape_operator_graph(pt_grad, pt_hess):
    device = pt_grad.device
    dtype = pt_grad.dtype
    pt_hess = pt_hess.type(dtype)
    if pt_grad.ndim == 1:
        pt_grad = pt_grad[:, None] # row vector
    normalization_factor = torch.sqrt(torch.linalg.norm(pt_grad)**2 + 1)
    identity_matrix = torch.eye(len(pt_grad), dtype=dtype).to(device)
    metric_tensor = identity_matrix + torch.matmul(pt_grad, pt_grad.T)
    shape_operator = - torch.linalg.solve(metric_tensor, pt_hess) / normalization_factor
    return shape_operator


def get_shape_operator_level_set(pt_grad, pt_hess, coordinate_transformation=None):
    """
    compute grad of implicit function g: a=(x_0, ... x_{n-2}) \to b=x_{n-1} (zero-indexed)
    this will gives us a coordinate system of the iso response surface in the coordinates
    x_0, ... x_{n-2}
    """
    device = pt_grad.device
    dtype = pt_grad.dtype
    pt_hess = pt_hess.type(dtype)
    if pt_grad.ndim == 1:
        pt_grad = pt_grad[:, None] # col vector
    # transformation _to_ new coordinates
    if coordinate_transformation is None:
        # we choose the coordinates such that the gradient of f is
        # only in the last dimension, which makes the
        # implicit function work best
        pt_grad_numpy = pt_grad.detach().cpu().numpy().T
        null_space_basis = null_space(pt_grad_numpy)
        pt_grad_numpy_normed = pt_grad_numpy / np.linalg.norm(pt_grad_numpy.flatten())
        new_basis = np.hstack((null_space_basis, pt_grad_numpy_normed.T))
        coordinate_transformation = torch.tensor(new_basis.T, dtype=dtype, device=device)
    else:
        coordinate_transformation = coordinate_transformation.type(dtype)
    pt_grad = torch.matmul(coordinate_transformation, pt_grad)
    pt_hess = torch.matmul(
        coordinate_transformation,
        torch.matmul(pt_hess, coordinate_transformation.T)
    )
    # convenience variables
    pt_grad_a = pt_grad[:-1]
    pt_grad_b = pt_grad[-1:]
    pt_hess_aa = pt_hess[:-1, :-1]
    pt_hess_ab = pt_hess[:-1, -1:]
    pt_hess_bb = pt_hess[-1:, -1:]
    # consistency tests
    if pt_grad_b == 0:
        # this should never happen in DNN cases
        raise ValueError('singular gradient, you need a different coordinate system')
    if torch.abs(pt_grad_b[0, 0]) < 1e-7:
        # this should never happen in DNN cases
        print('close to singular gradient, you might need a different coordinate system', pt_grad_b)
    # g is the implicit function from x_1, x_{n-1} to x_n
    grad_g = -pt_grad_a / pt_grad_b
    embedding_differential = torch.vstack((
        torch.eye(len(grad_g)).to(device),
        grad_g.T
    ))
    embedding_differential = torch.matmul(coordinate_transformation.T, embedding_differential)
    # alternative formulation:
    # hess_g = (
    #     (-1 / pt_grad_b) * (
    #         pt_hess_aa + 
    #         pt_hess_ab.T * grad_g.T
    #     ) +
    #     (1 / pt_grad_b ** 2) * pt_grad_a * (
    #         pt_hess_ab.T +
    #         pt_hess_bb * grad_g.T
    #     )
    # )
    hess_g = (-1 / pt_grad_b) * (
        pt_hess_ab.T * (grad_g + grad_g.T)
        +
        pt_hess_bb * grad_g * grad_g.T
        +
        pt_hess_aa
    )
    if pt_grad_b > 0: # make sure the normal points in the right direction
        grad_g = -grad_g
        hess_g = -hess_g
    normalization_factor = torch.sqrt(torch.linalg.norm(grad_g)**2 + 1)
    identity_matrix = torch.eye(len(grad_g), dtype=dtype).to(device)
    metric_tensor = identity_matrix + torch.matmul(grad_g, grad_g.T)
    shape_operator = - torch.linalg.solve(metric_tensor, hess_g) / normalization_factor
    return shape_operator, embedding_differential, metric_tensor


def get_shape_operator_moosavi(pt_grad, pt_hess):
    """
    Adapted from descriptions given in:
    SM Moosavi-Dezfooli, A Fawzi, O Fawzi, P Frossard, S Soatto (2018) - Robustness of Classifiers to Universal Perturbations: A Geometric Perspective
    and
    SM Moosavi-Dezfooli (2019) - Geometry of Adversarial Robustness of Deep Networks: Methods and Applications
    """
    device = pt_grad.device
    dtype = pt_grad.dtype
    identity_matrix = torch.eye(len(pt_grad), dtype=dtype).to(device)
    projection_operator = identity_matrix - torch.matmul(pt_grad, pt_grad.T)
    norm_constant = 1 / torch.linalg.norm(pt_grad)
    shape_operator = norm_constant * torch.matmul(
        projection_operator,
        torch.matmul(pt_hess, projection_operator.T)
    )
    return shape_operator


def get_shape_operator_golden(pt_grad, pt_hess):
    """
    Returns the shape operator, principal directions, principal curvature
    code adapted to pytorch from
        https://github.com/jamesgolden1/bias_free_denoising/blob/manifold_metric/curvature/hyperboloid_single_sheet_curvature_compare.ipynb
    Parameters:
        pt_grad [pytorch tensor] gradient vector for the input point
        pt_hess [pytorch tensor] hessian matrix for the input point
    """
    device = pt_grad.device
    dtype = pt_grad.dtype
    # Append gradient vector as extra col to identity matrix
    identity_matrix = torch.eye(len(pt_grad), dtype=dtype).to(device)
    embedding_differential = torch.zeros([len(pt_grad), len(pt_grad)+1], dtype=dtype).to(device)
    embedding_differential[:len(pt_grad), :len(pt_grad)] = identity_matrix.clone().detach()
    embedding_differential[:, len(pt_grad)] = pt_grad
    # Take inner product of this matrix with its transpose
    first_fundamental = torch.matmul(embedding_differential, embedding_differential.T)
    # Compute the normal vector to the manifold at the point of interest
    normal = torch.cat(
        (torch.matmul(identity_matrix.clone().detach(), pt_grad),
         torch.tensor([-1]).to(device)), dim=0)
    unit_normal = normal / torch.linalg.norm(normal)
    # Scale Hessian by the last element of the unit normal vector
    second_fundamental = torch.reshape(pt_hess.flatten() * unit_normal[-1], first_fundamental.shape)
    # Compute shape operator matrix = FF\SF
    shape_operator = torch.linalg.solve(first_fundamental, second_fundamental)
    return shape_operator


def get_principal_curvatures(shape_operator):
    """
    Performs an eigen decomposition of the shape operator
    Parameters:
        shape_operator - [M, M] dimensional array
    Returns:
        principal_curvatures - sorted eigenvalues of shape_operator
        principal_directions - sorted eigenvectors of shape_operator
            where principal_directions[:, i] is the vector corresponding to principal_curvatures[i]
    """
    dtype = shape_operator.dtype
    principal_curvatures, principal_directions = torch.linalg.eig(shape_operator)
    principal_curvatures = torch.real(principal_curvatures).type(dtype)
    principal_directions = torch.real(principal_directions).type(dtype)
    sort_indices = torch.argsort(principal_curvatures, descending=True)
    return principal_curvatures[sort_indices], principal_directions[:, sort_indices]


def local_response_curvature_graph(pt_grad, pt_hess):
    """
    Parameters:
        pt_grad: defining function gradient
        pt_hess: defining function hessian
    Outputs:
        shape_operator - [M, M] dimensional array
        principal_curvatures - [M] dimensional array of curvatures in ascending order
        principal_directions - [M,M] dimensional array,
            where principal_directions[:, i] is the vector corresponding to principal_curvatures[i]
    """
    shape_operator = get_shape_operator_graph(pt_grad, pt_hess)
    principal_curvatures, principal_directions = get_principal_curvatures(shape_operator)
    return shape_operator, principal_curvatures, principal_directions


def local_response_curvature_level_set(pt_grad, pt_hess, projection_subspace_of_interest=None, coordinate_transformation=None):
    """
    Parameters:
        pt_grad: defining function gradient
        pt_hess: defining function hessian
        projection_subspace_of_interest: [k, M] matrix. projection from ambient space to a subspace for which we are interested
          in the curvature. Curvature will be computed for the projection of the subspace of interest
          into the isoresponse surface.
        coordinate_transformation: orthogonal [M, M] matrix from input space into a new coordinate system. The last coordinate will
          be used to parametrize the decision boundary
    Outputs:
        shape_operator - [M-1, M-1] dimensional array
        principal_curvatures - [M-1] dimensional array of curvatures in ascending order
        principal_directions - [M,M-1] dimensional array,
            where principal_directions[:, i] is the vector corresponding to principal_curvatures[i]
    """
    dtype = pt_grad.dtype
    device = pt_grad.device
    shape_operator, embedding_differential, metric_tensor = get_shape_operator_level_set(pt_grad, pt_hess, coordinate_transformation=coordinate_transformation)
    if projection_subspace_of_interest is not None:
        projection_from_isosurface = projection_subspace_of_interest[:, :-1].type(dtype)
        # even if the projection was orthogonal originally, after we deleted the last column it might not be anymore
        # therefore we have to reorthogonalize
        projection_from_isosurface = orth(projection_from_isosurface.detach().cpu().numpy().T).T
        projection_from_isosurface = torch.tensor(projection_from_isosurface, dtype=dtype, device=device)
        # restrict shape operator to subspace of interest. This is the correct endomorphism for the restriced second fundamental form,
        # since the first projection cancels with the transposed projection that is part of the restricted metric.
        shape_operator = torch.matmul(torch.matmul(projection_from_isosurface, shape_operator), projection_from_isosurface.T)
    principal_curvatures, principal_directions = get_principal_curvatures(shape_operator)
    ## TODO: Verify with MK that it is ok to do this projection post-sorting
    # we need to norm the directions wrt to the metric, not the canonical scalar product
    if projection_subspace_of_interest is not None:
        _metric_tensor = torch.matmul(torch.matmul(projection_from_isosurface, metric_tensor), projection_from_isosurface.T)
    else:
        _metric_tensor = metric_tensor.type(dtype)
    direction_norms = torch.tensor([
        torch.dot(direction, torch.matmul(_metric_tensor, direction)) for direction in principal_directions.T
    ], dtype=dtype, device=device)
    principal_directions /= torch.sqrt(direction_norms)[None, :]
    if projection_subspace_of_interest is not None:
        principal_directions = torch.matmul(projection_from_isosurface.T.type(dtype), principal_directions)
    principal_directions = torch.matmul(embedding_differential.type(principal_directions.dtype), principal_directions)
    return shape_operator, principal_curvatures, principal_directions


def local_response_curvature_alternates(pt_grad, pt_hess, so_type='moosavi'):
    """
    Alternative published methods for computing local response curvature.
    if so_type == 'moosavi', then shape operator is computed using descriptions from:
        SM Moosavi-Dezfooli, A Fawzi, O Fawzi, P Frossard, S Soatto (2018) - Robustness of Classifiers to Universal Perturbations: A Geometric Perspective
        and
        SM Moosavi-Dezfooli (2019) - Geometry of Adversarial Robustness of Deep Networks: Methods and Applications
    
    and if so_type == 'golden', then the shape operator is computed from:
        JR Golden, KP Vilankar, DJ Field (2019) - Selective and Invariant Features of Neural Response Surfaces Measured with Principal Curvature
    
    and if so_type == 'lee_level' or 'lee_graph', then operator is computed from:
        DM Paiton, D Schultheiss, M Kümmerer, Z Cranko, M Bethge (2021) - The Geometry of Adversarial Subspaces
    """
    if so_type.lower() == 'moosavi':
        shape_operator = get_shape_operator_moosavi(pt_grad, pt_hess)
    elif so_type.lower() == 'golden':
        shape_operator = get_shape_operator_golden(pt_grad, pt_hess)
    elif so_type.lower() == 'lee_level':
        shape_operator = get_shape_operator_level_set(pt_grad, pt_hess)[0]
    elif so_type.lower() == 'lee_graph':
        shape_operator = get_shape_operator_graph(pt_grad, pt_hess)
    else:
        assert False
    principal_curvatures, principal_directions = get_principal_curvatures(shape_operator)
    return shape_operator, principal_curvatures, principal_directions