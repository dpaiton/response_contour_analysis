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
        target_vector [np.ndarray] vector with shape [vector_length,]
        comp_vector [np.ndarray] vector with shape [vector_length,]
    Outputs:
        orth_norm [np.ndarray] column vector with shape [vector_length,] that is orthogonal to target_vector and has unit norm
    """
    t_norm = np.squeeze((target_vector / np.linalg.norm(target_vector)))
    c_norm = np.squeeze((comp_vector / np.linalg.norm(comp_vector)))
    orth_vector = c_norm - np.dot(c_norm[:,None].T, t_norm[:,None]) * t_norm # column vector
    orth_norm = np.squeeze((orth_vector / np.linalg.norm(orth_vector)).T)
    return orth_norm


def get_proj_matrix(target_vector, comp_vector, comp_is_orth=False):
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
    orth_vect = np.squeeze(orth_vect / np.linalg.norm(orth_vect))
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
        "Input variable num_target_ids=%g must be less than num_neurons=%g."%(num_target_ids, num_neurons))
    return list(np.random.choice(range(num_neurons), num_target_ids, replace=False))


def compute_rand_iso_vectors(target_vectors, num_comparisons=1):
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
        target_vector = target_vector.reshape(target_vector.size) # shape is [vector_length,]
        norm_target_vectors.append(target_vector / np.linalg.norm(target_vector))
        rand_orth_vectors.append(get_rand_orth_vectors(target_vector, num_comparisons))
    return (norm_target_vectors, rand_orth_vectors)


def get_vector_angles(list_of_vectors):
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
        inner_products = np.dot((vec0 / np.linalg.norm(vec0)).T, (vec1 / np.linalg.norm(vec1)))
        inner_products[inner_products>1.0] = 1.0
        inner_products[inner_products<-1.0] = -1.0
        angle = np.arccos(inner_products)
        vect_angles[angleid] = angle * (180 / np.pi)
        angle_matrix[nid0, nid1] = angle * (180 / np.pi)
    angle_matrix[angle_matrix==0] = -1
    return vect_angles, angle_matrix


def compute_comp_iso_vectors(all_target_vectors, target_neuron_ids, min_angle=5, num_comparisons=1):
    """
    For each target neuron, build dataset of selected orthogonal vectors
        We first select a comparison vector from all_target_vectors that has the smallest inner-angle
        onto a given target vector. Orthogonal vectors are selected such that the defined plane has a
        maximal inner-product with both the target vector and the comparison  vector.
    Parameters:
        all_target_vectors [list] list of target_vector candidates
            (e.g. maximally activating images for neurons)
        target_neuron_ids [list] list of ints for indexing all_target_vectors
        min_angle [float] minimum allowable angle for comparison vectors
        num_comparisons [int] number of comparison planes to use for each target neuron
    Outputs:
        comparison_neuron_ids [list of list] [num_targets][num_comparisons_per_target]
        normed_target_vectors [list] of normalized vectors to be used for data generation
        comparison_vectors [list of np.ndarrays] each element in the list (one for each target vector) contains
            a matrix of non-random orthogonal (to each other & to the target vector) & normalized vectors
            with shape [num_comparisons, vector_length]. The vectors are computed from other entries in
            all_target_vectors using the Gram-Schmidt process.
    TODO: Should extra_indices be 1) random orth vectors, 2) random vectors selected from all_target_vectors, 3) left as-is, or 4) parameterized to select 1-3
    """
    num_target_vectors = len(all_target_vectors)
    vector_length = all_target_vectors[0].size
    # angle_matrix is a [num_target_vectors, num_target_vectors] ndarray that
    # gives the angle between all pairs of target vectors
    angle_matrix = get_vector_angles(all_target_vectors)[1]
    num_below_min = np.count_nonzero(angle_matrix<min_angle) # many angles are -1 or 0
    # sorted_angle_indices is an [N, 2] ndarray,
    # where N = num_target_vectors**2 - num_below_min and '2' indexes each axis of angle_matrix
    sorted_angle_indices = np.stack(np.unravel_index(np.argsort(angle_matrix.ravel()),
        angle_matrix.shape), axis=1)[num_below_min:, :]
    # Compute indices of comparison vectors for each target vector
    comparison_neuron_ids = [] # list of lists [num_targets][num_comparisons_per_target]
    normed_target_vectors = []
    comparison_vectors = []
    for target_neuron_id in target_neuron_ids:
        target_vector = all_target_vectors[target_neuron_id]
        # Reshape & rescale target vector
        target_vector = target_vector.reshape(target_vector.size)
        target_vector = target_vector / np.linalg.norm(target_vector)
        normed_target_vectors.append(target_vector)
        target_neuron_locs = np.argwhere(sorted_angle_indices[:, 0] == target_neuron_id)
        # high_angle_neuron_ids gives the indices of all neurons that have a high angle (above min_angle) with the target neuron
        high_angle_neuron_ids = np.squeeze(sorted_angle_indices[target_neuron_locs, 1])
        # If there are not enough high angle neurons to satisfy num_comparisons then fill out with other target vectors
        extra_indices = []
        for index in range(num_target_vectors):
            if index not in high_angle_neuron_ids:
                if index != target_neuron_id:
                    extra_indices.append(index)
        if len(extra_indices) > 0:
            try:
                sub_comparison_neuron_ids = np.concatenate((np.atleast_1d(high_angle_neuron_ids),
                    np.array(extra_indices)))
            except:
              print("ERROR:iso_response_analysis: concatenation failed - likely one of the arrays is size 0")
              import IPython; IPython.embed(); raise SystemExit
        else:
            sub_comparison_neuron_ids = high_angle_neuron_ids
        sub_comparison_neuron_ids = sub_comparison_neuron_ids[:num_comparisons]
        # Build out matrix of comparison vectors from the computed IDs
        comparison_vector_matrix = target_vector.T[:,None] # matrix of alternate vectors
        for comparison_neuron_id in sub_comparison_neuron_ids:
            if(comparison_neuron_id != target_neuron_id):
                comparison_vector = all_target_vectors[comparison_neuron_id]
                comparison_vector = comparison_vector.reshape(vector_length)
                comparison_vector = np.squeeze((comparison_vector / np.linalg.norm(comparison_vector)).T)
                comparison_vector_matrix = np.append(comparison_vector_matrix, comparison_vector[:,None], axis=1)
        comparison_neuron_ids.append(sub_comparison_neuron_ids)
        comparison_vectors.append(comparison_vector_matrix.T[1:,:])
    return (comparison_neuron_ids, normed_target_vectors, comparison_vectors)


def get_contour_dataset(target_vectors, comparison_vectors, x_range, y_range, num_images, image_scale=1):
    """
    Parameters:
        target_vectors [list] of normalized target vectors
        comparison_vectors [list] of normalized comparison vectors
        x_range [tuple] indicating (min, max) for x axis
        y_range [tuple] indicating (min, max) for y axis
        num_images [int] indicating how many images to compute from each plane. This must have an even square root.
        image_scale [float] indicating desired length of the image vectors.
            Each normalized image vector will be multiplied by image_scale after being injected into image space.
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
    x_pts = np.linspace(x_range[0], x_range[1], int(np.sqrt(num_images)))
    y_pts = np.linspace(y_range[0], y_range[1], int(np.sqrt(num_images)))
    X_mesh, Y_mesh = np.meshgrid(x_pts, y_pts)
    proj_datapoints = np.stack([X_mesh.reshape(num_images), Y_mesh.reshape(num_images)], axis=1)
    all_datapoints = []
    out_dict = {
        "proj_target_vect": [],
        "proj_comparison_vect": [],
        "proj_orth_vect": [],
        "orth_vect": [],
        "proj_datapoints": proj_datapoints,
        "x_pts": x_pts,
        "y_pts": y_pts}
    for target_vect, all_comparison_vects in zip(target_vectors, comparison_vectors):
        proj_target_vect_sub_list = []
        proj_comparison_vect_sub_list = []
        proj_orth_vect_sub_list = []
        orth_vect_sub_list = []
        datapoints_sub_list = []
        num_comparison_vects = all_comparison_vects.shape[0]
        for comparison_vect_idx in range(num_comparison_vects): # Each contour plane for the population study
            comparison_vect = np.squeeze(all_comparison_vects[comparison_vect_idx, :])
            proj_matrix = get_proj_matrix(target_vect, comparison_vect, comp_is_orth=False) # works fine even if it is orthogonal
            orth_vect = np.squeeze(proj_matrix[1,:])
            proj_target_vect_sub_list.append(np.dot(proj_matrix, target_vect).T) #project
            proj_comparison_vect_sub_list.append(np.dot(proj_matrix, comparison_vect).T) #project
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
        out_dict["proj_target_vect"].append(proj_target_vect_sub_list)
        out_dict["proj_comparison_vect"].append(proj_comparison_vect_sub_list)
        out_dict["proj_orth_vect"].append(proj_orth_vect_sub_list)
        out_dict["orth_vect"].append(orth_vect_sub_list)
    return out_dict, all_datapoints
