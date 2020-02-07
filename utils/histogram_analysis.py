"""
Utility funcions for summarizing response contours

Authors: Dylan Paiton, Santiago Cadena
"""

import numpy as np
from skimage import measure


def get_bins(all_curvatures, num_bins=50):
    """
    compute bin edges & centers for histograms
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


def compute_curvature_fits(activations, contour_dataset, target_act):
    """
    activations [tuple] first element is the comp activations returned from get_normalized_activations and the secdond element is rand activations
    contour_dataset can be rand_contour_dataset or comp_contour_dataset.
        It assumes that rand_contour_dataset["x_pts"] equals comp_contour_dataset["x_pts"]
        It assumes that rand_contour_dataset["proj_datapoints"] equals comp_contour_dataset["proj_datapoints"]
    Some possible outputs have been commented because they are not currently used, but they might be useful in the future.
    """
    (comp_activations, rand_activations) = activations
    (num_comp_target_neurons, num_comp_planes, num_points_y, num_points_x) = comp_activations.shape
    (num_rand_target_neurons, num_rand_planes, num_points_y, num_points_x) = rand_activations.shape
    # Iso response curvature
    iso_comp_curvatures = []; iso_rand_curvatures = []
    activations_and_curvatures = (
        (comp_activations, iso_comp_curvatures),
        (rand_activations, iso_rand_curvatures)
        )
    for activations, curvatures in activations_and_curvatures:
        for neuron_id in range(num_comp_target_neurons):
            sub_curvatures = []
            for plane_id in range(num_comp_planes):
                activity = activations[neuron_id, plane_id, ...]
                ## mirror top half of activations to only measure curvature in the upper right quadrant
                num_y, num_x = activity.shape 
                activity[:int(num_y/2), :] = activity[int(num_y/2):, :][::-1, :]
                ## compute curvature
                contours = measure.find_contours(activity, target_act)[0]
                x_vals = contours[:,1]
                y_vals = contours[:,0]
                coeffs = np.polynomial.polynomial.polyfit(y_vals, x_vals, deg=2)
                sub_curvatures.append(coeffs[-1])
            curvatures.append(sub_curvatures)
    # Attenuation curvature
    x_pts = contour_dataset["x_pts"]
    proj_datapoints = contour_dataset["proj_datapoints"]
    attn_comp_curvatures = []; attn_rand_curvatures = []#; attn_comp_fits = []; attn_comp_sliced_activity = []; attn_rand_fits = []; attn_rand_sliced_activity = []
    for neuron_index in range(num_rand_target_neurons):
        sub_comp_curvatures = []; sub_comp_sliced_activity = []; sub_rand_curvatures = [] #; sub_comp_fits = []; sub_rand_fits = []
        sub_rand_sliced_activity = []
        for orth_index in range(num_rand_planes): # comparison vector method
            comp_activity = comp_activations[neuron_index, orth_index, ...] # [y, x]
            x_act = np.squeeze(comp_activity[:, 0])
            closest_target_act = x_act[np.abs(x_act - target_act).argmin()] # find a location to take a slice
            x_target_index = np.argwhere(x_act == closest_target_act)[0].item() # find the index along x axis
            x_target = x_pts[x_target_index] # find the x value at this index
            slice_indices = np.where(proj_datapoints[:, 0] == x_target)[0]
            slice_datapoints = proj_datapoints[slice_indices, :][:, :] # slice grid
            sub_comp_sliced_activity.append(comp_activity.reshape([-1])[slice_indices][:])
            
            coeff = np.polynomial.polynomial.polyfit(slice_datapoints[:, 1],
                sub_comp_sliced_activity[-1], deg=2) # [c0, c1, c2], where p = c0 + c1x + c2x^2
            sub_comp_curvatures.append(-coeff[2]) # multiply by -1 so that positive coeff is "more" curvature
            #sub_comp_fits.append(np.polynomial.polynomial.polyval(slice_datapoints[:, 1], coeff))
        for orth_index in range(num_rand_planes): # random vector method
            rand_activity = rand_activations[neuron_index, orth_index, ...].reshape([-1])
            sub_rand_sliced_activity.append(rand_activity[slice_indices][:])
            coeff = np.polynomial.polynomial.polyfit(slice_datapoints[:, 1],
                sub_rand_sliced_activity[-1], deg=2)
            sub_rand_curvatures.append(-coeff[2])
            #sub_rand_fits.append(np.polynomial.polynomial.polyval(slice_datapoints[:, 1], coeff))
        attn_comp_curvatures.append(sub_comp_curvatures)
        #attn_comp_fits.append(sub_comp_fits)
        #attn_comp_sliced_activity.append(sub_comp_sliced_activity)
        attn_rand_curvatures.append(sub_rand_curvatures)
        #attn_rand_fits.append(sub_rand_fits)
        #attn_rand_sliced_activity.append(sub_rand_sliced_activity)
    iso_curvatures = (iso_comp_curvatures, iso_rand_curvatures)
    attn_curvatures = (attn_comp_curvatures, attn_rand_curvatures)
    return (iso_curvatures, attn_curvatures)


def compute_curvature_hists(curvatures, num_bins):
    """
    """
    ((iso_comp_curvatures, iso_rand_curvatures), (attn_comp_curvatures, attn_rand_curvatures)) = curvatures
    num_neurons = len(iso_comp_curvatures)
    # Compute uniform bins for both iso-curvature plots and both attenuation-curvature plots
    iso_all_curvatures = []
    attn_all_curvatures = []
    for neuron_index in range(num_neurons):
        iso_all_curvatures += iso_comp_curvatures[neuron_index]
        iso_all_curvatures += iso_rand_curvatures[neuron_index]
        attn_all_curvatures += attn_comp_curvatures[neuron_index]
        attn_all_curvatures += attn_rand_curvatures[neuron_index]
    iso_bins = get_bins(iso_all_curvatures, num_bins)
    attn_bins = get_bins(attn_all_curvatures, num_bins)
    iso_comp_hist = []; iso_rand_hist = []; attn_comp_hist = []; attn_rand_hist = []
    for target_id in range(len(iso_comp_curvatures)):
        # Iso-response histogram
        flat_comp_curvatures = [item for item in iso_comp_curvatures[target_id]]
        comp_hist, iso_bin_edges = np.histogram(flat_comp_curvatures, iso_bins, density=False)
        iso_comp_hist.append(comp_hist / len(flat_comp_curvatures))
        flat_rand_curvatures = [item for item in iso_rand_curvatures[target_id]]
        rand_hist, _ = np.histogram(flat_rand_curvatures, iso_bins, density=False)
        iso_rand_hist.append(rand_hist / len(flat_rand_curvatures))
        # Response attenuation histogram
        flat_comp_curvatures = [item for item in attn_comp_curvatures[target_id]]
        comp_hist, attn_bin_edges = np.histogram(flat_comp_curvatures, attn_bins, density=False)
        attn_comp_hist.append(comp_hist / len(flat_comp_curvatures))
        flat_rand_curvatures = [item for item in attn_rand_curvatures[target_id]]
        rand_hist, _ = np.histogram(flat_rand_curvatures, attn_bins, density=False)
        attn_rand_hist.append(rand_hist / len(flat_rand_curvatures))
    return ((iso_comp_hist, iso_rand_hist), (attn_comp_hist, attn_rand_hist), (iso_bin_edges, attn_bin_edges))