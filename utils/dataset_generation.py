"""
Utility funcions for generating datasets for response analysis

Authors: Dylan Paiton, Santiago Cadena
"""

import numpy as np


def gram_schmidt(target_vector, comp_vector):
    """
    Perform a single step of the Gram-Schmidt process 
    https://en.wikipedia.org/wiki/Gram-Schmidt_process
    Parameters:
        target_vector [np.ndarray] of shape [num_pixels,]
        comp_vector [np.ndarray] of shape [num_pixels,]
    Outputs:
        orth_norm [np.ndarray] of shape [num_pixels,] that is orthogonal to target_vector and has unit norm
    """
    t_norm = np.squeeze((target_vector / np.linalg.norm(target_vector)).T)
    c_norm = np.squeeze((comp_vector / np.linalg.norm(comp_vector)).T)
    orth_vector = c_norm - np.dot(c_norm[:,None].T, t_norm[:,None]) * t_norm
    orth_norm = np.squeeze((orth_vector / np.linalg.norm(orth_vector)).T)
    return orth_norm


def get_proj_matrix(target_vector, comp_vector, comp_is_orth=False):
    """
    Find a projection basis that is orthogonal to target_vector 
    Uses a single step of the Gram-Schmidt process
    Parameters:
        target_vector [np.ndarray] of shape [num_pixels,]
        comp_vector [np.ndarray] of shape [num_pixels,]
        comp_is_orth [bool] If false, find orth vector that is as close as possible to comp_vector
    Outputs:
        projection_matrix [tuple] containing [ax_1, ax_2] for projecting data into the 2d array
    """
    if comp_is_orth:
        orth_vector = comp_vector
    else:
        orth_vector = gram_schmidt(target_vector, comp_vector)
    proj_matrix = np.stack([target_vector, orth_vector], axis=0)
    return proj_matrix


def find_orth_vect(matrix):
    """
    Given an orthonormal matrix, find a new unit vector that is orthogonal
    Parameters:
        matrix [np.ndarray] matrix whose columns are each orthonormal vectors
    Outputs:
        orth_vect [np.ndarray] unit vector that is orthogonal to all of the vectors in the input matrix
    """
    rand_vect = np.random.rand(matrix.shape[0], 1)
    new_matrix = np.hstack((matrix, rand_vect))
    candidate_vect = np.zeros(matrix.shape[1]+1)
    candidate_vect[-1] = 1
    orth_vect = np.linalg.lstsq(new_matrix.T, candidate_vect, rcond=None)[0] # [0] indexes lst-sqrs solution
    orth_vect = np.squeeze((orth_vect / np.linalg.norm(orth_vect)).T)
    return orth_vect


def get_rand_orth_vectors(target_vector, num_orth_directions):
    """
    Given a vector, construct a matrix of shape [num_orth_directions, vector_length] of vectors
    that are orthogonal to the input
    Parameters:
      target_vector [np.ndarray] initial vector
      num_orth_directions [int] number of orthogonal vectors to construct
    Outputs:
      rand_vectors [np.ndarray] output matrix of shape [num_orth_directions, vector_length] containing vectors
      that are all orthogonal to target_vector
    """
    rand_vectors = target_vector.T[:,None] # matrix of alternate vectors
    for orth_idx in range(num_orth_directions):
        tmp_vect = find_orth_vect(rand_vectors)
        rand_vectors = np.append(rand_vectors, tmp_vect[:,None], axis=1)
    return rand_vectors.T[1:, :]


def get_image_angles(images):
    """
    Compute the angle in degrees between all pairs of basis functions in bf_stats
    Parameters:
        images [list] list of images to compute the angle bewteen
    Outputs:
        image_angles [np.ndarray] lower triangle of plot matrix only, as a vector in raster order
        plot_matrix [np.ndarray] of shape [num_images, num_images] with all angles between
            basis functions in the lower triangle and upper triangle is set to -1
    """
    num_images = len(images)
    num_pixels = images[0].size
    indices = np.tril_indices(num_images, 1)
    vect_size = len(indices[0])
    image_angles = np.zeros(vect_size)
    plot_matrix = np.zeros((num_images, num_images))
    for angleid, (nid0, nid1) in enumerate(zip(*indices)):
        im0 = images[nid0].reshape((num_pixels, 1))
        im1 = images[nid1].reshape((num_pixels, 1))
        inner_products = np.dot((im0 / np.linalg.norm(im0)).T, (im1 / np.linalg.norm(im1)))
        inner_products[inner_products>1.0] = 1.0
        inner_products[inner_products<-1.0] = -1.0
        angle = np.arccos(inner_products)
        image_angles[angleid] = angle * (180 / np.pi)
        plot_matrix[nid0, nid1] = angle * (180 / np.pi)
    plot_matrix[plot_matrix==0] = -1
    return image_angles, plot_matrix


def get_rand_target_neuron_ids(num_target_ids, num_neurons):
    """
    """
    assert num_target_ids < num_neurons, (
        "Input variable 'num_target_ids' must be less than %g, not %g."%(num_neurons, num_target_ids))
    return list(np.random.choice(range(num_neurons), num_target_ids, replace=False))


def compute_rand_iso_vectors(neuron_meis, target_neuron_ids, num_comparisons=1):
    """
    Calculate all projection vectors for each target neuron
    For each target neuron, build dataset of random orthogonal vectors
    Parameters:
        neuron_meis [list] list of images for maximally activating images for neurons
        target_neuron_ids [list] list of ints for indexing neuron_meis
        num_comparisons [int] number of comparison planes to use for each target neuron
    Outputs:
        target_vectors [list] of normalized vectors to be used for data generation
        rand_orth_vectors [list of np.ndarrays] each element in the list (one for each target vector) contains a matrix of random orthogonal (to each other & to the target vector) & normalized vectors with shape [num_comparisons, num_pixels]
    """
    num_neurons = len(neuron_meis)
    num_pixels = neuron_meis[0].size
    target_vectors = []
    rand_orth_vectors = []
    for neuron_idx, target_neuron_id in enumerate(target_neuron_ids):
        target_vector = neuron_meis[target_neuron_id]
        target_vector = target_vector.reshape(neuron_meis[target_neuron_id].size)
        target_vector = target_vector / np.linalg.norm(target_vector)
        target_vectors.append(target_vector)
        rand_orth_vectors.append(get_rand_orth_vectors(target_vector, num_comparisons))
    return (target_vectors, rand_orth_vectors)


def compute_comp_iso_vectors(neuron_meis, min_angle, target_neuron_ids, num_comparisons=1):
    """
    Calculate all projection vectors for each target neuron
    For each target neuron, build dataset of selected orthogonal vectors
    Parameters:
        neuron_meis [list] list of images for maximally activating images for neurons
        target_neuron_ids [list] list of ints for indexing neuron_meis
        num_comparisons [int] number of comparison planes to use for each target neuron
    Outputs:
        comparison_neuron_ids [list of list] [num_targets][num_comparisons_per_target]
        target_vectors [list] of normalized vectors to be used for data generation
        comparison_vectors [list of np.ndarrays] each element in the list (one for each target vector) contains a matrix of non-random orthogonal (to each other & to the target vector) & normalized vectors with shape [num_comparisons, num_pixels]. The vectors are computed from other MEIs using the Gram-Schmidt process.
    """
    num_neurons = len(neuron_meis)
    num_pixels = neuron_meis[0].size
    neuron_angles, plot_matrix = get_image_angles(neuron_meis)
    num_above_min = np.count_nonzero(plot_matrix<min_angle) # many angles are -1 or 0
    sorted_angle_indices = np.stack(np.unravel_index(np.argsort(plot_matrix.ravel()),
        plot_matrix.shape), axis=1)[num_above_min:, :]
    comparison_neuron_ids = [] # list of lists [num_targets][num_comparisons_per_target]
    target_vectors = []
    comparison_vectors = []
    for neuron_idx, target_neuron_id in enumerate(target_neuron_ids):
        target_vector = neuron_meis[target_neuron_id]
        # Reshape & rescale target vector
        target_vector = target_vector.reshape(neuron_meis[target_neuron_id].size)
        target_vector = target_vector / np.linalg.norm(target_vector)
        target_vectors.append(target_vector)
        target_neuron_locs = np.argwhere(sorted_angle_indices[:,0] == target_neuron_id)
        low_angle_neuron_ids = np.squeeze(sorted_angle_indices[target_neuron_locs, 1])
        extra_indices = []
        for index in range(num_neurons):
            if index not in low_angle_neuron_ids:
                if index != target_neuron_id:
                    extra_indices.append(index)
        if len(extra_indices) > 0:
            try:
                sub_comparison_neuron_ids = np.concatenate((np.atleast_1d(low_angle_neuron_ids),
                    np.array(extra_indices)))
            except:
              print("ERROR:iso_response_analysis: concatenation failed - likely one of the arrays is size 0")
              import IPython; IPython.embed(); raise SystemExit
        else:
            sub_comparison_neuron_ids = low_angle_neuron_ids
        sub_comparison_neuron_ids = sub_comparison_neuron_ids[:num_comparisons]
        comparison_vector_matrix = target_vector.T[:,None] # matrix of alternate vectors
        for comparison_neuron_id in sub_comparison_neuron_ids:
            if(comparison_neuron_id != target_neuron_id):
                comparison_vector = neuron_meis[comparison_neuron_id]
                comparison_vector = comparison_vector.reshape(num_pixels)
                comparison_vector = np.squeeze((comparison_vector / np.linalg.norm(comparison_vector)).T)
                comparison_vector_matrix = np.append(comparison_vector_matrix, comparison_vector[:,None], axis=1)
        comparison_neuron_ids.append(sub_comparison_neuron_ids)
        comparison_vectors.append(comparison_vector_matrix.T[1:,:])
    return (comparison_neuron_ids, target_vectors, comparison_vectors)


def get_contour_dataset(target_vectors, comparison_vectors, x_range, y_range, num_images, image_scale=1):
    """
    Parameters:
    Returns:
        datapoints has shape [num_target_neurons][num_comparisons_per_target (or num_planes)][num_datapoints, datapoint_length]
    """
    x_pts = np.linspace(x_range[0], x_range[1], int(np.sqrt(num_images)))
    y_pts = np.linspace(y_range[0], y_range[1], int(np.sqrt(num_images)))
    X_mesh, Y_mesh = np.meshgrid(x_pts, y_pts)
    proj_datapoints = np.stack([X_mesh.reshape(num_images), Y_mesh.reshape(num_images)], axis=1)
    all_datapoints = []
    out_dict = {
        "proj_target_neuron": [],
        "proj_comparison_neuron": [],
        "proj_orth_vect": [],
        "orth_vect": [],
        "proj_datapoints": proj_datapoints,
        "x_pts": x_pts,
        "y_pts": y_pts}
    for target_vect, all_comparison_vects in zip(target_vectors, comparison_vectors):
        proj_target_neuron_sub_list = []
        proj_comparison_neuron_sub_list = []
        proj_orth_vect_sub_list = []
        orth_vect_sub_list = []
        datapoints_sub_list = []
        num_comparison_vects = all_comparison_vects.shape[0]
        for comparison_vect_idx in range(num_comparison_vects): # Each contour plane for the population study
            comparison_vect = np.squeeze(all_comparison_vects[comparison_vect_idx, :])
            proj_matrix = get_proj_matrix(target_vect, comparison_vect, comp_is_orth=False)
            orth_vect = np.squeeze(proj_matrix[1,:])
            proj_target_neuron_sub_list.append(np.dot(proj_matrix, target_vect).T) #project
            proj_comparison_neuron_sub_list.append(np.dot(proj_matrix, comparison_vect).T) #project
            proj_orth_vect_sub_list.append(np.dot(proj_matrix, orth_vect).T) #project
            orth_vect_sub_list.append(orth_vect)
            datapoints = np.stack([np.dot(proj_matrix.T, proj_datapoints[data_id,:])
                for data_id in range(num_images)], axis=0) #inject
            num_datapoints, data_length = datapoints.shape
            data_edge = int(np.sqrt(data_length))
            datapoints = datapoints.reshape((num_datapoints, data_edge, data_edge, 1))
            datapoints = datapoints * image_scale # rescale datapoints
            datapoints_sub_list.append(datapoints)
        all_datapoints.append(datapoints_sub_list)
        out_dict["proj_target_neuron"].append(proj_target_neuron_sub_list)
        out_dict["proj_comparison_neuron"].append(proj_comparison_neuron_sub_list)
        out_dict["proj_orth_vect"].append(proj_orth_vect_sub_list)
        out_dict["orth_vect"].append(orth_vect_sub_list)
    return out_dict, all_datapoints