"""
Utility funcions for summarizing response contours

Authors: Dylan Paiton, Santiago Cadena
"""

import numpy as np
from skimage import measure


def get_bins(all_curvatures, num_bins=50):
    """
    Compute bin edges & centers for histograms by finding min and max across all values given
        The output will always have a bin centered at 0
    Parameters:
        all_curvatures [list of floats] curvature values to be binned
        num_bins [int] number of evenly-spaced bins
    Outputs:
        bins [np.ndarray] of size [num_bins,] with a range that includes the min and max of all_curvatures
    """
    max_curvature = np.amax(all_curvatures)
    min_curvature = np.amin(all_curvatures)
    bin_width = (max_curvature - min_curvature) / (num_bins-1) # subtract 1 to leave room for the zero bin
    bin_centers = [0.0]
    while min(bin_centers) > min_curvature:
        bin_centers.append(bin_centers[-1]-bin_width)
    bin_centers = bin_centers[::-1]
    while max(bin_centers) < max_curvature:
        bin_centers.append(bin_centers[-1]+bin_width)
    bin_lefts = bin_centers - (bin_width / 2)
    bin_rights = bin_centers + (bin_width / 2)
    bins = np.append(bin_lefts, bin_rights[-1])
    return bins


def iso_response_curvature_poly_fits(activations, target, target_is_act=True):
    """
    Parameters:
        activations [tuple] first element is the comp activations returned from get_normalized_activations and the secdond element is rand activations
        target [float] target activity for finding iso-response contours OR target position along x axis for finding the target activity
            if target_is_act is false, then target refers to a position on the x axis from -1 (left most point) to 1 (right most point)
        target_is_act [bool] if True, then the 'target' parameter is an activation value, else the 'target' parameter is the x axis position
    Outputs:
        curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated iso-response curvature coefficient for the given neuron and plane
        fits [list of lists] of lengths [num_neurons, num_planes] which contain the polyval line fits for the given computed coefficients
    """
    curvatures = []; fits = []
    for neuron_id in range(activations.shape[0]):
        sub_curvatures = []; sub_fits = []
        for plane_id in range(activations.shape[1]):
            activity = activations[neuron_id, plane_id, ...]
            num_y, num_x = activity.shape
            if target_is_act:
                target_act = target
            else:
                target_pos = int(num_x * ((target + 1) / 2)) # map [-1, 1] to [0, num_x]
                target_act = activity[num_y//2, target_pos]
            ## mirror top half of activations to only measure curvature in the upper right quadrant
            activity[:int(num_y/2), :] = activity[int(num_y/2):, :][::-1, :]
            ## compute curvature
            contours = measure.find_contours(activity, target_act)[0]
            x_vals = contours[:,1]
            y_vals = contours[:,0]
            coeffs = np.polynomial.polynomial.polyfit(y_vals, x_vals, deg=2)
            sub_curvatures.append(coeffs[-1])
            sub_fits.append(np.polynomial.polynomial.polyval(y_vals, coeffs))
        curvatures.append(sub_curvatures)
        fits.append(sub_fits)
    return (curvatures, fits)


def response_attenuation_curvature_poly_fits(activations, target_act, x_pts, proj_datapoints):
    """
    Parameters:
        activations [tuple] first element is the comp activations returned from get_normalized_activations and the secdond element is rand activations
        x_pts [np.ndarray] of size (num_images,) that were the points used for the contour dataset
        proj_datapoints [np.ndarray] of stacked vectorized datapoint locations (from a mesh grid) for the 2D contour dataset
        target_act [float] target activity for finding iso-response contours
    Outputs:
        curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated response attenuation curvature coefficient for the given neuron and plane
        fits [list of lists] of lengths [num_neurons, num_planes] which contain the polyval line fits for the given computed coefficients
        sliced_activity [list of lists] of lengths [num_neurons, num_planes] which contain a vector of activity values for the chosen x value
    """
    curvatures = []; fits = []; sliced_activity = []
    for neuron_index in range(activations.shape[0]):
        sub_curvatures = []; sub_fits = []; sub_sliced_activity = []
        for orth_index in range(activations.shape[1]):
            activity = activations[neuron_index, orth_index, ...] # [y, x]
            x_act = np.squeeze(activity[:, 0])
            closest_target_act = x_act[np.abs(x_act - target_act).argmin()] # find a location to take a slice
            x_target_index = np.argwhere(x_act == closest_target_act)[0].item() # find the index along x axis
            x_target = x_pts[x_target_index] # find the x value at this index
            slice_indices = np.where(proj_datapoints[:, 0] == x_target)[0]
            slice_datapoints = proj_datapoints[slice_indices, :][:, :] # slice grid
            sub_sliced_activity.append(activity.reshape([-1])[slice_indices][:])
            coeffs = np.polynomial.polynomial.polyfit(slice_datapoints[:, 1],
                sub_sliced_activity[-1], deg=2) # [c0, c1, c2], where p = c0 + c1x + c2x^2
             # TODO: Previously we multiplied this value by -1, b ut then the fits are weird -
             #  an we multiply the data by -1 to invert it and then the coeffs are correct when
             #  they come out of polyval?
            sub_curvatures.append(coeffs[2])
            sub_fits.append(np.polynomial.polynomial.polyval(slice_datapoints[:, 1], coeffs))
        curvatures.append(sub_curvatures)
        fits.append(sub_fits)
        sliced_activity.append(sub_sliced_activity)
    return (curvatures, fits, sliced_activity)


def compute_curvature_poly_fits(activations, contour_dataset, target_act):
    """
    Parameters:
        activations [tuple] first element is the comp activations returned from get_normalized_activations and the secdond element is rand activations
        contour_dataset [dict] that must contain keys "x_pts" and "proj_datapoints" from utils/dataset_generation.get_contour_dataset()
          x_pts [np.ndarray] of size (num_images,) that were the points used for the contour dataset
          proj_datapoints [np.ndarray] of stacked vectorized datapoint locations (from a mesh grid) for the 2D contour dataset
        target_act [float] target activity for finding iso-response contours
    outputs:
        iso_curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated iso-response curvature coefficient for the given neuron and plane
        attn_curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated response attenuation curvature coefficient for the given neuron and plane
    """
    iso_curvatures = iso_response_curvature_poly_fits(activations, target_act)[0]
    attn_curvatures = response_attenuation_curvature_poly_fits(activations, target_act,
        contour_dataset["x_pts"], contour_dataset["proj_datapoints"])[0]
    return (iso_curvatures, attn_curvatures)


def get_normalized_hist(curvatures, bins):
    """
    Compute normalized histogram for curvature values
    Parameters:
        curvatures [nested list of floats] that is indexed by [comparison plane id][bin index]
        bins [sequence of scalars] defines a monotonically increasing array of bin edges, including the rightmost edge
    Outputs:
        hist [np.ndarray] normalized histogram values
        bin_edges [np.ndarray] histogram bin edges of len = len(hist)+1
    """
    flat_curvatures = [item for item in curvatures]
    hist, bin_edges = np.histogram(flat_curvatures, bins, density=False)
    hist = hist / len(flat_curvatures)
    return hist, bin_edges


def get_bins_from_all_curvatures(type_curvatures, num_bins):
    """
    Compute uniform bins for all curvature plots
    Parameters:
        type_curvatures [nested list of floats] that is indexed by
            [dataset type]
            [target neuron id]
            [comparison plane id]
        num_bins [int] number of evenly-spaced bins
    Outputs:
        bins [np.ndarray] of size [num_bins,] with a range that includes the min and max of all_curvatures
    """
    all_curvatures = []
    for dataset_curvatures in type_curvatures:
        for neuron_curvatures in dataset_curvatures:
            all_curvatures += neuron_curvatures
    bins = get_bins(all_curvatures, num_bins)
    return bins


def compute_curvature_hists(curvatures, num_bins):
    """
    Compute histograms for all curvatures with shared support across curvature types
    Parameters:
        curvatures [nested list of floats] that is indexed by
            [curvature type]
            [dataset type]
            [target neuron id]
            [comparison plane id]
        num bins [int] number of bins to use for all histograms
    Outputs:
        all_hists [nested list of histogram values] that is indexed by
            [curvature type]
            [dataset type]
            [target neuron id]
        all_bin_edges [list] bin edges for each curvature type
    """
    all_bins = [get_bins_from_all_curvatures(type_curvature, num_bins)
        for type_curvature in curvatures]
    all_hists = []; all_bin_edges = []
    for type_curvatures, type_bins in zip(curvatures, all_bins):
        type_sub_hists = []
        for dataset_curvatures in type_curvatures:
            dataset_sub_hists = []
            for neuron_curvatures in dataset_curvatures:
                neuron_hist, type_bin_edges = get_normalized_hist(neuron_curvatures, type_bins)
                dataset_sub_hists.append(neuron_hist)
            type_sub_hists.append(dataset_sub_hists)
        all_hists.append(type_sub_hists)
        all_bin_edges.append(type_bin_edges)
    return [all_hists, all_bin_edges]
