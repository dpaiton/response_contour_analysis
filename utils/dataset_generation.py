"""
Utility funcions for generating datasets for response analysis

Authors: Dylan Paiton, Santiago Cadena
"""

import numpy as np
import scipy
import skimage
import torch


def remap_axis_index_to_dataset_index(axis_index, axis_min, axis_max, num_images):
    """
    Map axis_index from [axis_min, axis_max] to [0, num_images-1]
    Parameters:
        axis_index [float] location along axis to represent, must be between axis_min and axis_max
        axis_min [float] minimum vlaue of projection axis
        axis_max [float] maximum value of projection axis
        num_images [int] number of images (i.e. discrete bins) that lie along an axis
    Outputs:
        dataset_index [int] output in terms of discrete bin location from 0 to num_images-1
    """
    assert axis_index >= axis_min, (f'dataset_generation/remap_axis_index_to_dataset_index:'
        +'ERROR: axis_index = {axis_index} must be greater than or equal to axis_min = {axis_min}.')
    assert axis_index <= axis_max, (f'dataset_generation/remap_axis_index_to_dataset_index:'
        +'ERROR: axis_index = {axis_index} must be less than or equal to axis_max = {axis_max}.')
    norm_axis_index = (axis_index - axis_min) / (axis_max - axis_min) # map axis_index to [0, 1]
    dataset_index = norm_axis_index * (num_images - 1) # map axis_index to [0, num_images-1]
    return int(dataset_index)


def angle_between_vectors(vec0, vec1):
    """
    Returns the cosine angle between two vectors
    Parameters:
        vec_a [np.ndarray] with shape [vector_length, 1]
        vec_b [np.ndarray] with shape [vector_length, 1]
    Outputs:
        angle [float] angle between the two vectors, in radians
    """
    assert vec0.shape == vec1.shape, (f'vec0.shape = {vec0.shape} must equal vec1.shape={vec1.shape}')
    assert vec0.ndim == 2, (f'vec0 should have ndim==2 with secondary dimension=1, not {vec0.shape}')
    inner_products = np.dot(l2_normalize(vec0).T, l2_normalize(vec1))
    inner_products = np.clip(inner_products, -1.0, 1.0)
    angle = np.arccos(inner_products)
    return angle


def torch_angle_between_vectors(vec0, vec1):
    """
    Returns the cosine angle between two vectors
    Parameters:
        vec_a [torch vector] with shape [vector_length, 1]
        vec_b [torch vector] with shape [vector_length, 1]
    Outputs:
        angle [float] angle between the two vectors, in radians
    """
    assert vec0.shape == vec1.shape, (f'vec0.shape = {vec0.shape} must equal vec1.shape={vec1.shape}')
    assert vec0.ndim == 2, (f'vec0 should have ndim==2 with secondary dimension=1, not {vec0.shape}')
    inner_products = torch.matmul(torch_l2_normalize(vec0).T, torch_l2_normalize(vec1))
    inner_products = torch.clip(inner_products, -1.0, 1.0)
    angle = torch.arccos(inner_products)
    return angle


def l2_normalize(array):
    """
    Convert input to a vector, and then divide it by its l2 norm.
    Parameters:
        array [ np.ndarray] vector with shape [vector_length,].
            It can also be a 2D image, which will first be vectorized
    Outputs:
        array [np.ndarray] as same shape as the input and with l2-norm = 1
    """
    original_shape = array.shape
    vector = array.reshape(array.size)
    vector = vector / np.linalg.norm(vector)
    if vector.shape != original_shape:
        vector.reshape(original_shape)
    return vector


def torch_l2_normalize(array):
    """
    Convert input to a vector, and then divide it by its l2 norm.
    Parameters:
        array [torch tensor] vector with shape [vector_length,].
            It can also be a 2D image, which will first be vectorized
    Outputs:
        array [torch tensor] as same shape as the input and with l2-norm = 1
    """
    original_shape = array.shape
    vector = array.reshape(-1)
    vector = vector / torch.linalg.norm(vector)
    if vector.shape != original_shape:
        vector.reshape(original_shape)
    return vector


def define_plane(target_vector, comp_vector):
    """
    Perform a single step of the Gram-Schmidt process
        https://en.wikipedia.org/wiki/Gram-Schmidt_process
    Parameters:
        target_vector [np.ndarray] vector with shape [vector_length,]
        comp_vector [np.ndarray] vector with shape [vector_length,]
    Outputs:
        orth_normed [np.ndarray] column vector with shape [vector_length,] that is orthogonal to target_vector and has unit norm
    """
    t_normed = np.squeeze(l2_normalize(target_vector))
    c_normed = np.squeeze(l2_normalize(comp_vector))
    if np.all(t_normed == c_normed):
        print('ERROR: From dataset_generation/gram_schmidt.py: input target vector and comp vector are equal and must be different.')
        import IPython; IPython.embed(); raise SystemExit
    orth_normed = gram_schmidt(t_normed, c_normed)
    return t_normed, orth_normed


def gram_schmidt(target_normed, comp_normed):
    """
    Perform a single step of the Gram-Schmidt process
        https://en.wikipedia.org/wiki/Gram-Schmidt_process
    Parameters:
        target_normed [np.ndarray] l2 normalized vector with shape [vector_length,]
        comp_normed [np.ndarray] l2 normalized vector with shape [vector_length,]
    Outputs:
        orth_normed [np.ndarray] l2 normalized vector with shape [vector_length,] that is orthogonal to target_vector
    """
    orth_vector = comp_normed - np.dot(comp_normed[:, None].T, target_normed[:, None]) * target_normed
    orth_norm = np.linalg.norm(orth_vector)
    if orth_norm == 0:
        print('WARNING: From dataset_generation/gram_schmidt.py: norm of orthogonal vector is 0.')
    orth_normed = np.squeeze((orth_vector / orth_norm).T) # normalize & transpose to be a row vector
    return orth_normed


def get_proj_matrix(target_vector, comp_vector):
    """
    Computes an orthonormal matrix composed of the target_vector and an orthogonal vector
    Parameters:
        target_vector [np.ndarray] of shape [vector_length,]
        comp_vector [np.ndarray] of shape [vector_length,]
        comp_is_orth [bool] If false, find orth vector that is as close as possible
            to comp_vector using the Gram-Schmidt process
    Outputs:
        projection_matrix [np.ndarray] of shape [2, vector_length] for projecting data into and out of pixel space
    """
    normed_target_vector, orth_vector = define_plane(target_vector, comp_vector)
    proj_matrix = np.stack([normed_target_vector, orth_vector], axis=0)
    return proj_matrix


def project_data(proj_matrix, datapoints, image_scale=1.0):
    """
    Project ND datapoints into 2D plane
    Parameters:
        proj_matrix [np.ndarray] of shape [2, N] for projecting data to the 2D plane
        datapoitns [np.ndarray] of shape [num_datapoints, N] high dimensional data
    outputs:
        proj_datapoints [np.ndarray] of shape [num_datapoints, 2] projected data
    """
    num_images, data_length = datapoints.shape
    proj_datapoints = np.dot(datapoints / image_scale, proj_matrix.T).astype(np.float32)
    return proj_datapoints


def inject_data(proj_matrix, proj_datapoints, image_scale=1.0, data_shape=None):
    """
    Inject 2D datapoints into ND
    Parameters:
        proj_matrix [np.ndarray] of shape [2, N] for injecting data into higher dimensions
        proj_datapoints [np.ndarray] of shape [num_datapoints, 2] 2D points to be injected
            for example a meshgrid of coordinate locations
        image_scale [float] final norm of datapoints
            for example the average norm of the training images
        data_shape [list]
            if provided, then the output will be shaped [num_datapoints]+data_shape
            if None, then the output will be shaped [num_datapoints, int(sqrt(N)), int(sqrt(N))]
    Outputs:
        datapoints [np.ndarray] of shape [num_datapoints, N] injected data
    """
    input_dim, data_length = proj_matrix.shape
    num_datapoints = proj_datapoints.shape[0]
    datapoints = np.dot(proj_datapoints, proj_matrix).astype(np.float32)
    if data_shape is None:
        data_edge = int(np.sqrt(data_length))
        datapoints = datapoints.reshape((num_datapoints, 1, data_edge, data_edge))
    else:
        datapoints = datapoints.reshape([num_datapoints] + list(data_shape))
    datapoints *= image_scale # rescale datapoint norm
    return datapoints


def get_datamesh(yx_range, num_images):
    """
    Returns a meshgrid of points within the specified range
    Parameters:
        yx_range [list of tuple] indicating [(x_axis_min, x_axis_max), (y_axis_min, y_axis_max)]
        num_images [int] indicating how many images to compute from each plane. This must have an even square root.
    Outputs:
        proj_datapoints [np.ndarray] of shape [num_datapoints, 2]
    """
    y_range = yx_range[0]
    x_range = yx_range[1]
    x_pts = np.linspace(x_range[0], x_range[1], int(np.sqrt(num_images)))
    y_pts = np.linspace(y_range[0], y_range[1], int(np.sqrt(num_images)))
    X_mesh, Y_mesh = np.meshgrid(x_pts, y_pts)
    proj_datapoints = np.stack([X_mesh.reshape(num_images), Y_mesh.reshape(num_images)], axis=1)
    return proj_datapoints, x_pts, y_pts


def one_to_many_angles(vec_a, vec_list_b):
    """
    Returns cosine angle from one vector to a list of vectors
    Parameters:
        vec_a [np.ndarray] l2 normalized vector with shape [vector_length,]
        vec_list_b [list of np.ndarray] list of l2 normalized vectors with shape [num_vectors][vector_length,]
    Outputs:
        angles [list of floats] angle between vec_a and each of the vectors in vec_list_b, in radians
    """
    angles = []
    for vec_b in vec_list_b:
        angles.append(angle_between_vectors(vec_a, vec_b))
    return angles


def all_to_all_angles(list_of_vectors):
    """
    Compute the angle in degrees between all pairs of vectors
    Parameters:
        list_of_vectors [list] list of vectors (e.g. images) to compute the angle bewteen
    Outputs:
        vect_angles [np.ndarray] lower triangle of plot matrix only, as a vector in raster order
        angle_matrix [np.ndarray] of shape [num_vectors, num_vectors] with all angles between
            basis functions in the lower triangle and upper triangle is set to -1
    """
    num_vectors = len(list_of_vectors)
    vector_length = list_of_vectors[0].size
    indices = np.tril_indices(num_vectors, 1)
    vect_size = len(indices[0])
    vect_angles = np.zeros(vect_size)
    angle_matrix = np.zeros((num_vectors, num_vectors))
    for angleid, (nid0, nid1) in enumerate(zip(*indices)):
        vec0 = list_of_vectors[nid0].reshape((vector_length, 1))
        vec1 = list_of_vectors[nid1].reshape((vector_length, 1))
        angle = angle_between_vectors(vec0, vec1) * (180 / np.pi) # degrees
        vect_angles[angleid] = angle
        angle_matrix[nid0, nid1] = angle
    angle_matrix[angle_matrix==0] = -1
    return vect_angles, angle_matrix


def find_orth_vect(matrix):
    """
    Given an orthonormal matrix, find a new unit vector that is orthogonal
    Parameters:
        matrix [np.ndarray] matrix of shape [num_vectors, vector_length] whose columns are each orthonormal vectors
    Outputs:
        orth_vect [np.ndarray] unit vector of shape [vector_length,] that is orthogonal to
            all of the vectors in the input matrix
    """
    rand_vect = np.random.rand(matrix.shape[1], 1)
    new_matrix = np.append(matrix, rand_vect.T, axis=0)
    candidate_vect = np.zeros(matrix.shape[0]+1)
    candidate_vect[-1] = 1
    orth_vect = np.linalg.lstsq(new_matrix, candidate_vect, rcond=None)[0] # [0] indexes lst-sqrs solution
    orth_vect = np.squeeze(l2_normalize(orth_vect))
    return orth_vect


def get_rand_orth_vectors(target_vector, num_orth_directions):
    """
    Given an input vector, construct a matrix of normalized column vectors that are orthogonal to the input
    Parameters:
        target_vector [np.ndarray] initial vector of shape [vector_length,]
        num_orth_directions [int] number of orthogonal vectors to construct
    Outputs:
        rand_vectors [np.ndarray] output matrix of shape [num_orth_directions, vector_length] containing vectors
        that are all orthogonal to target_vector
    """
    rand_vectors = target_vector[None, :] # matrix of alternate vectors
    for orth_idx in range(num_orth_directions):
        tmp_vect = find_orth_vect(rand_vectors)
        rand_vectors = np.append(rand_vectors, tmp_vect[None,:], axis=0)
    return rand_vectors[1:, :]


def get_rand_target_neuron_ids(num_target_ids, num_neurons):
    """
    Compute list of random integers corresponding to IDs for target neurons.
    Parameters:
        num_target_ids [int] number of integer IDs to return
        num_neurons [int] total number of neurons to select from
    Outputs:
        rand_ids [list] of len num_target_ids containing random integers selected from range(num_neurons)
    """
    assert num_target_ids < num_neurons, (
        'Input variable num_target_ids=%g must be less than num_neurons=%g.'%(num_target_ids, num_neurons))
    return list(np.random.choice(range(num_neurons), num_target_ids, replace=False))


def compute_rand_vectors(target_vectors, num_comparisons=1):
    """
    Calculate all random orthogonal vectors for a selection of target vectors
    Parameters:
        target_vectors [list] list of target_vector candidates
            (e.g. maximally activating images for neurons)
        num_comparisons [int] number of comparison planes to use for each target neuron
    Outputs:
        norm_target_vectors [list] of normalized vectors of shape [vector_length,] to be used for data generation
        rand_orth_vectors [list of np.ndarrays] each element in the list (one for each target vector) contains
            a matrix of random orthogonal (to each other & to the target vector) & normalized vectors
            with shape [num_comparisons, vector_length]
    """
    norm_target_vectors = []
    rand_orth_vectors = []
    for target_vector in target_vectors:
        norm_target_vectors.append(l2_normalize(target_vector.reshape(target_vector.size))) # shape is [vector_length,]
        rand_orth_vectors.append(get_rand_orth_vectors(target_vector, num_comparisons))
    return (norm_target_vectors, rand_orth_vectors)


def compute_comp_vectors(all_vectors, target_vector_ids, min_angle=5, num_comparisons=1, comp_method='closest'):
    """
    For each target neuron, build dataset of selected orthogonal vectors
        We first select a comparison vector from all_vectors that has the smallest inner-angle
        onto a given target vector. Orthogonal vectors are selected such that the defined plane has a
        maximal inner-product with both the target vector and the comparison vector.
    Parameters:
        all_vectors [list] list of target_vector candidates
            (e.g. maximally activating images for neurons)
        target_vector_ids [list] list of ints for indexing all_vectors
        min_angle [float] minimum allowable angle for comparison vectors
        num_comparisons [int] number of comparison planes to use for each target neuron
        comp_method [str] can be 'closest' or 'rand' to compute comparison vectors that are
            closest in angle to the target vector or random, respectively
    Outputs:
        comparison_vector_ids [list of list] [num_targets][num_comparisons_per_target]
        normed_target_vectors [list] of normalized vectors to be used for data generation
        comparison_vectors [list of np.ndarrays] each element in the list (one for each target vector) contains
            a matrix of non-random orthogonal (to each other & to the target vector) & normalized vectors
            with shape [num_comparisons, vector_length]. The vectors are computed from other entries in
            all_vectors using the Gram-Schmidt process.
    TODO: Should extra_indices be 1) random orth vectors, 2) random vectors selected from all_vectors, 3) left as-is, or 4) parameterized to select 1-3
    """
    total_num_vectors = len(all_vectors)
    vector_length = all_vectors[0].size
    # angle_matrix is a [total_num_vectors, total_num_vectors] ndarray that
    # gives the angle between all pairs of target vectors
    angle_matrix = all_to_all_angles(all_vectors)[1]
    num_below_min = np.count_nonzero(angle_matrix < min_angle) # many angles are -1 or 0
    # sorted_angle_indices is an [N, 2] ndarray,
    # where N = total_num_vectors**2 - num_below_min and '2' indexes each axis of angle_matrix
    sorted_angle_indices = np.stack(np.unravel_index(np.argsort(angle_matrix.ravel()),
        angle_matrix.shape), axis=1)[num_below_min:, :]
    # Compute indices of comparison vectors for each target vector
    comparison_vector_ids = [] # list of lists [num_targets][num_comparisons_per_target]
    normed_target_vectors = []
    comparison_vectors = []
    for target_neuron_id in target_vector_ids:
        target_vector = l2_normalize(all_vectors[target_neuron_id])
        normed_target_vectors.append(target_vector)
        target_neuron_locs = np.argwhere(sorted_angle_indices[:, 0] == target_neuron_id)
        # high_angle_neuron_ids are indices for all neurons that have a high angle (above min_angle) with the target neuron
        high_angle_neuron_ids = np.squeeze(sorted_angle_indices[target_neuron_locs, 1])
        # If there are not enough high angle neurons to satisfy num_comparisons then fill out with other target vectors
        extra_indices = []
        for index in range(total_num_vectors):
            if index not in high_angle_neuron_ids:
                if index != target_neuron_id:
                    extra_indices.append(index)
        if len(extra_indices) > 0:
            try:
                sub_comparison_vector_ids = np.concatenate((np.atleast_1d(high_angle_neuron_ids),
                    np.array(extra_indices)))
            except:
              print('ERROR:iso_response_analysis: concatenation failed - likely one of the arrays is size 0')
              import IPython; IPython.embed(); raise SystemExit
        else:
            sub_comparison_vector_ids = high_angle_neuron_ids
        if comp_method == 'closest':
            sub_comparison_vector_ids = sub_comparison_vector_ids[:num_comparisons] # down select the closest ones
        elif comp_method == 'rand':
            shuffled_indices = np.random.choice(range(len(sub_comparison_vector_ids)), num_comparisons, replace=False)
            sub_comparison_vector_ids = sub_comparison_vector_ids[shuffled_indices] # pick them at random
        else:
            assert False, (f'comp_method must be "closest" or "rand", not {comp_method}.')
        # Build out matrix of comparison vectors from the computed IDs
        comparison_vector_matrix = target_vector.T[:, None] # matrix of alternate vectors
        for comparison_vector_id in sub_comparison_vector_ids:
            if(comparison_vector_id != target_neuron_id):
                comparison_vector = np.squeeze(l2_normalize(all_vectors[comparison_vector_id]).T)
                comparison_vector_matrix = np.append(comparison_vector_matrix, comparison_vector[:,None], axis=1)
        comparison_vector_ids.append(sub_comparison_vector_ids)
        comparison_vectors.append(comparison_vector_matrix.T[1:,:])
    return (comparison_vector_ids, normed_target_vectors, comparison_vectors)


def compute_specified_vectors(all_vectors, target_vector_ids, comparison_vector_ids):
    """
    Compile a dataset of normalized target and comparison vectors using specified list indices
    Parameters:
        all_vectors [list] list of target_vector candidates
            (e.g. maximally activating images for neurons)
        target_vector_ids [list] list of ints for indexing all_vectors
        comparison_vector_ids [list of list] [num_targets][num_comparisons_per_target] list of ints for indexing all_vectors
    Outputs:
        normed_target_vectors [list] of normalized vectors to be used for data generation
        comparison_vectors [list of np.ndarrays] each element in the list (one for each target vector) contains
            a matrix of non-random orthogonal (to each other & to the target vector) & normalized vectors
            with shape [num_comparisons, vector_length]. The vectors are computed from other entries in
            all_vectors using the Gram-Schmidt process.
    """
    normed_target_vectors = []
    comparison_vectors = []
    for target_index, target_neuron_id in enumerate(target_vector_ids):
        target_vector = l2_normalize(all_vectors[target_neuron_id])
        normed_target_vectors.append(target_vector)
        # Build out matrix of comparison vectors from the computed IDs
        comparison_vector_matrix = target_vector.T[:, None] # matrix of alternate vectors
        for comparison_vector_id in comparison_vector_ids[target_index]:
            if(comparison_vector_id != target_neuron_id):
                comparison_vector = all_vectors[comparison_vector_id]
                comparison_vector = np.squeeze(l2_normalize(comparison_vector).T)
                comparison_vector_matrix = np.append(comparison_vector_matrix,
                    comparison_vector[:,None], axis=1)
            else:
                assert False, ('Comparison vector ID cannot equal target vector ID')
        comparison_vectors.append(comparison_vector_matrix.T[1:,:])
    return (normed_target_vectors, comparison_vectors)


def get_contour_dataset(target_vectors, comparison_vectors, yx_range, num_images, image_scale=1.0, data_shape=None, return_datapoints=True):
    """
    Parameters:
        target_vectors [list] of normalized target vectors
        comparison_vectors [list; or list of list] of normalized comparison vectors
            if type is list then each entry should be an np.ndarray comparison vector to be combined with all target vectors
            if type is list of list then it is assumed that the a comparison vector is provided for each target vector
        yx_range [list of tuple] indicating [(x_axis_min, x_axis_max), (y_axis_min, y_axis_max)]
        num_images [int] indicating how many images to compute from each plane. This must have an even square root.
        image_scale [float] indicating desired length of the image vectors.
            Each normalized image vector will be multiplied by image_scale after being injected into image space.
        return_datapoints [bool] if True, return the datapoints injected into the input space
    Returns:
        out_dict [dict] containing the following keys:
            proj_target_vect - target vector, projected onto the 2D plane
            proj_comparison_vect - comparison vector, projected onto the 2D plane
            proj_orth_vect - orthogonal vector, computed using the Gram-Schmidt process
            orth_vect - orthogonal vector in image space
            proj_datapoints - datapoint locations in 2D plane
            x_pts - linear interpolation between x_range[0] and x_range[1]
            y_pts - linear interpolation between y_range[0] and y_range[1]
        datapoints has shape [num_target_neurons][num_comparisons_per_target (or num_planes)][num_datapoints, datapoint_length]
    """
    proj_datapoints, x_pts, y_pts = get_datamesh(yx_range, num_images)
    all_datapoints = []
    out_dict = {
        'orth_vect': [],
        'proj_target_vect': [],
        'proj_comparison_vect': [],
        'proj_orth_vect': [],
        'proj_matrix': [],
        'proj_datapoints': proj_datapoints,
        'x_pts': x_pts,
        'y_pts': y_pts
    }
    if type(comparison_vectors[0]) is list: # each target vector has pre-defined comparison vectors
        target_comp_zip = zip(target_vectors, comparison_vectors)
    elif type(comparison_vectors[0]) is np.ndarray: # generator combines each comparison vector with each target vector
        target_comp_zip = ((target_vector, comparison_vectors) for target_vector in target_vectors)
    else:
        assert False, ('comparison vectors must be of type "list of list" or "list of np.ndarray", not "%s"'%(type(comparison_vectors)))
    for target_vect, target_comp_vects in target_comp_zip:
        orth_vect_sub_list = []
        proj_target_vect_sub_list = []
        proj_comparison_vect_sub_list = []
        proj_orth_vect_sub_list = []
        proj_matrix_sub_list = []
        datapoints_sub_list = []
        for comparison_vect in target_comp_vects: # Each contour plane for the population study
            comparison_vect = np.squeeze(comparison_vect)
            proj_matrix = get_proj_matrix(target_vect, comparison_vect)
            orth_vect = np.squeeze(proj_matrix[1,:])
            if return_datapoints:
                datapoints_sub_list.append(inject_data(proj_matrix, proj_datapoints, image_scale, data_shape))
            else:
                datapoints_sub_list.append(None)
            orth_vect_sub_list.append(orth_vect)
            proj_target_vect_sub_list.append(np.dot(proj_matrix, target_vect).T) #project
            proj_comparison_vect_sub_list.append(np.dot(proj_matrix, comparison_vect).T) #project
            proj_orth_vect_sub_list.append(np.dot(proj_matrix, orth_vect).T) #project
            proj_matrix_sub_list.append(proj_matrix)
        all_datapoints.append(datapoints_sub_list)
        out_dict['orth_vect'].append(orth_vect_sub_list)
        out_dict['proj_target_vect'].append(proj_target_vect_sub_list)
        out_dict['proj_comparison_vect'].append(proj_comparison_vect_sub_list)
        out_dict['proj_orth_vect'].append(proj_orth_vect_sub_list)
        out_dict['proj_matrix'].append(proj_matrix_sub_list)
    if return_datapoints:
        out_dict['all_datapoints'] = all_datapoints
    return out_dict


def hilbert_amplitude(data, padding=None):
    """
    Compute Hilbert amplitude envelope of data matrix
    Inputs:
        data: [np.ndarray] of shape [num_data, num_rows, num_cols]
        padding: [list of int] specifying how much 0-padding to use for FFT along the row & column dimension, respectively
            default is the closest power of 2 of maximum(num_rows, num_cols)
    Outputs:
        envelope: [np.ndarray] same shape as data, contains Hilbert envelope
    TODO: Bug when num_data = 1
    """
    cart2pol = lambda x,y: (np.arctan2(y,x), np.hypot(x, y))
    num_data, num_rows, num_cols = data.shape
    if padding is None or max(padding) <= largest_edge_size:
        # Amount of zero padding for fft2 (closest power of 2)
        Ny = np.int(2**(np.ceil(np.log2(num_rows))))
        Nx = np.int(2**(np.ceil(np.log2(num_cols))))
    else:
        Ny = np.int(padding[0])
        Nx = np.int(padding[1])
    # Analytic signal envelope for data
    # (Hilbet transform of each image)
    envelope = np.zeros((num_data, num_rows, num_cols), dtype=complex)
    # Fourier transform of data
    f_data = np.zeros((num_data, Ny, Nx), dtype=complex)
    # Filtered Fourier transform of data
    # Hilbert filters
    hil_filt = np.zeros((num_data, Ny, Nx))
    # Grid for creating filter
    freqsx = (2 / Nx) * np.pi * np.arange(-Nx / 2.0, Nx / 2.0)
    freqsy = (2 / Ny) * np.pi * np.arange(-Ny / 2.0, Ny / 2.0)
    (mesh_fx, mesh_fy) = np.meshgrid(freqsx, freqsy)
    (theta, r) = cart2pol(mesh_fx, mesh_fy)
    for data_idx in range(num_data):
        # Grab single datapoint
        datapoint = (data - np.mean(data, axis=0, keepdims=True))[data_idx, ...]
        # Convert datapoint into DC-centered Fourier domain
        f_datapoint = np.fft.fftshift(np.fft.fft2(datapoint, [Ny, Nx]))
        f_data[data_idx, ...] = f_datapoint
        # Find indices of the peak amplitude
        max_ys = np.abs(f_datapoint).argmax(axis=0) # Returns row index for each col
        max_x = np.argmax(np.abs(f_datapoint).max(axis=0))
        # Convert peak amplitude location into angle in freq domain
        fx_ang = freqsx[max_x]
        fy_ang = freqsy[max_ys[max_x]]
        theta_max = np.arctan2(fy_ang, fx_ang)
        # Define the half-plane with respect to the maximum
        ang_diff = np.abs(theta - theta_max)
        idx = (ang_diff > np.pi).nonzero()
        ang_diff[idx] = 2.0 * np.pi - ang_diff[idx]
        hil_filt[data_idx, ...] = (ang_diff < np.pi / 2.).astype(int)
        # Create analytic signal from the inverse FT of the half-plane filtered datapoint
        abf = np.fft.ifft2(np.fft.fftshift(hil_filt[data_idx, ...] * f_datapoint))
        envelope[data_idx, ...] = abf[0:num_rows, 0:num_cols]
    return envelope


def gaussian_fit(pyx):
    """
    Compute the expected mean & covariance matrix for a 2-D gaussian fit of input distribution
    Inputs:
        pyx: [np.ndarray] of shape [num_rows, num_cols] that indicates the probability function to fit
    Outputs:
        mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
        cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
    """
    assert pyx.ndim == 2, (
        'Input must have 2 dimensions specifying [num_rows, num_cols]')
    mean = np.zeros((1, 2), dtype=np.float32) # [mu_y, mu_x]
    for idx in np.ndindex(pyx.shape): # [y, x] ticks columns (x) first, then rows (y)
        mean += np.asarray([pyx[idx]*idx[0], pyx[idx]*idx[1]])[None, :]
    cov = np.zeros((2,2), dtype=np.float32)
    for idx in np.ndindex(pyx.shape): # ticks columns first, then rows
        cov += np.dot((idx-mean).T, (idx-mean))*pyx[idx] # typically an outer-product
    return (np.squeeze(mean), cov)


def generate_gaussian(shape, mean, cov):
    """
    Generate a Gaussian PDF from given mean & cov
    Inputs:
        shape: [tuple] specifying (num_rows, num_cols)
        mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
        cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
    Outputs:
        tuple containing (Gaussian PDF, grid_points used to generate PDF)
            grid_points are specified as a tuple of (y,x) points
    """
    y_size, x_size = shape
    y = np.linspace(0, y_size, np.int32(np.floor(y_size)))
    x = np.linspace(0, x_size, np.int32(np.floor(x_size)))
    mesh_x, mesh_y = np.meshgrid(x, y)
    pos = np.empty(mesh_x.shape + (2,)) #x.shape == y.shape
    pos[:, :, 0] = mesh_y; pos[:, :, 1] = mesh_x
    gauss = scipy.stats.multivariate_normal(mean=mean, cov=cov)
    return (gauss.pdf(pos), (mesh_y, mesh_x))


def get_gauss_fit(prob_map, num_attempts=1, perc_mean=0.33):
    """
    Returns a gaussian fit for a given probability map
    Fitting is done via robust regression, where a fit is
    continuously refined by deleting outliers num_attempts times
    Inputs:
        prob_map [np.ndarray] set of 2-D probability map to be fit
        num_attempts: Number of times to fit & remove outliers
        perc_mean: All probability values below perc_mean*mean(gauss_fit) will be
            considered outliers for repeated attempts
    Outputs:
        gauss_fit: [np.ndarray] specifying the 2-D Gaussian PDF
        grid: [tuples] containing (y,x) points with which the Gaussian PDF can be plotted
        gauss_mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
        gauss_cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
    """
    assert prob_map.ndim==2, (
        'get_gauss_fit: ERROR: Input prob_map must have 2 dimension specifying [num_rows, num_cols')
    assert np.isfinite(prob_map).all(), (
        'get_gauss_fit: ERROR: prob_map must be finite')
    assert prob_map.max() - prob_map.min() != 0, (
        'get_gauss_fit: ERROR: prob_map cannot have one value for all entries')
    if num_attempts < 1:
        num_attempts = 1
    gauss_success = False
    while not gauss_success:
        tmp_prob_map = prob_map.copy()
        try:
            for i in range(num_attempts):
                map_min = np.min(tmp_prob_map)
                tmp_prob_map -= map_min
                map_sum = np.sum(tmp_prob_map)
                if map_sum != 1.0:
                    tmp_prob_map /= map_sum
                gauss_mean, gauss_cov = gaussian_fit(tmp_prob_map)
                assert np.isfinite(gauss_mean).all(), 'get_gauss_fit: ERROR: Gauss mean value is not finite'
                assert np.isfinite(gauss_cov).all(), 'get_gauss_fit: ERROR: Gauss cov value is not finite'
                gauss_fit, grid = generate_gaussian(tmp_prob_map.shape, gauss_mean, gauss_cov)
                gauss_fit = (gauss_fit * map_sum) + map_min
                if i < num_attempts-1:
                    gauss_mask = gauss_fit.copy()
                    gauss_mask[np.where(gauss_mask<perc_mean*np.mean(gauss_mask))] = 0
                    gauss_mask[np.where(gauss_mask>0)] = 1
                    tmp_prob_map *= gauss_mask
            gauss_success = True
        except np.linalg.LinAlgError: # Usually means cov matrix is singular
            print(f'get_gauss_fit: Failed to fit Gaussian at attempt {i}, trying again.'+
            '\n  To avoid this try decreasing perc_mean.')
            num_attempts = i-1
            if num_attempts <= 0:
                assert False, (
                  'get_gauss_fit: np.linalg.LinAlgError - Unable to fit gaussian.')
    return (gauss_fit, grid, gauss_mean, gauss_cov)


def construct_mask_from_mean_cov(gauss_mean, gauss_cov, shape, confidence=0.90):
    evals, evecs = np.linalg.eigh(gauss_cov)
    sort_indices = np.argsort(evals)[::-1]
    largest_eigval = evals[sort_indices][0]
    smallest_eigval = evals[sort_indices][-1]
    angle = np.arctan2(*evecs[:, sort_indices][0])
    chisq_val = scipy.stats.chi2.ppf(confidence, 2)
    height = chisq_val * np.sqrt(smallest_eigval) # b
    width = chisq_val * np.sqrt(largest_eigval) # a
    mask = np.zeros(shape)
    rr, cc = skimage.draw.ellipse(
        gauss_mean[0],
        gauss_mean[1],
        height/2, # radius
        width/2, # radius
        shape=shape,
        rotation=angle)
    mask[rr, cc] = 1
    return mask, [height, width], angle


def mask_then_normalize(vector, mask, mask_threshold):
    """
    Parameters:
        mask [np.ndarray] mask to zero out vector values with shape [vector_rows, vector_cols] or [vector_length,]
        vector [np.ndarray] vector with shape [vector_rows, vector_cols] or [vector_length,].
    Outputs:
        vector [np.ndarray] masked vector with shape [vector_length,] and l2-norm = 1
    """
    mask = mask.flatten()
    vector = vector.flatten()
    assert mask.size == vector.size, (
        f'mask size = {mask.size} must equal vector size = {vector.size}')
    mask /= mask.max()
    mask[mask<mask_threshold] = 0
    mask[mask>0] = 1
    vector = np.multiply(mask, vector)
    vector = vector / np.linalg.norm(vector)
    return vector


def poly_mask_image(image, yx_range, vector_slope, edge_length):
    ((y_min, y_max), (x_min, x_max)) = yx_range
    ranges = [y_max - y_min, x_max - x_min]
    mins = [y_min, x_min]
    cv_endpoint = y_max/vector_slope
    conv_pts = lambda li : [int((li[i] - mins[i]) / ranges[i] * edge_length) for i in range(len(li))]
    all_pts = [
        conv_pts([0, 0]), # [y, x]
        conv_pts([0, x_max]),
        conv_pts([y_max, x_max]),
        conv_pts([y_max, cv_endpoint]),
        conv_pts([0, 0])
    ]
    polygon = np.array(all_pts)
    mask = skimage.draw.polygon2mask(image.shape, polygon)
    image[~mask] = 0
    return image
