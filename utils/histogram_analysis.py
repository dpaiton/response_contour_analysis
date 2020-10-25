"""
Utility funcions for summarizing response contours

Authors: Dylan Paiton, Santiago Cadena
"""

import numpy as np
import numpy.polynomial.polynomial as poly
from skimage import measure


def iso_response_curvature_poly_fits(activations, target_act, xy_scale=[1, 1], measure_loc='all'):
    """
    Parameters:
        activations [np.ndarray] first index is the target neuron, second index is the data plane
        target_act [float] target activity for finding iso-response contours
        xy_scale [list of ints] x and y (respectively) scale factors for remapping activations to the data domain
        measure_loc [str] Choose location to measure iso-response contours
            this assumes the original range of X and Y values were from -K to K for some integer, K
            'all' -> (default) do not modify the activity map
            'left' -> only measure contours left of the (x-) middle
            'right' -> only measure contours right of the (x-) middle
            'top' -> only measure contours above the (y-) middle
            'bottom' -> only measure contours below the (y-) middle
            'upperright' -> measure contours in the top right quadrant 
    Outputs:
        curvatures [list of lists] of lengths [num_neurons][num_planes] which contain the estimated iso-response curvature coefficient for the given neuron and plane
        fits [list of lists] of lengths [num_neurons][num_planes] which contain the polyval line fits for the given computed coefficients
    Notes:
        this function uses scikit.measure.find_contours, documented here:
        https://scikit-image.org/docs/0.16.x/api/skimage.measure.html#skimage.measure.find_contours
    """
    contours_list = []; curvatures = []; fits = []
    for neuron_id in range(activations.shape[0]):
        sub_contours = []; sub_curvatures = []; sub_fits = []
        for plane_id in range(activations.shape[1]):
            activity = activations[neuron_id, plane_id, ...]
            if activity.max() > 0.0:
                #if measure_upper_right: # mirror top half of activations to bottom half
                #    activity[:int(num_y/2), :] = activity[int(num_y/2):, :][::-1, :] 
                num_y, num_x = activity.shape
                if measure_loc == 'left':
                    #activity = activity[:, :num_x//2] # remove data for x>0
                    activity[:, num_x//2:] = 0 # remove data for x>0
                elif measure_loc == 'right':
                    #activity = activity[:, num_x//2:] # remove data for x<0
                    activity[:, :num_x//2] = 0 # remove data for x<0
                elif measure_loc == 'top':
                    #activity = activity[num_y//2:, ] # remove data for y<0
                    activity[:num_y//2, ] = 0 # remove data for y<0
                elif measure_loc == 'bottom':
                    #activity = activity[:num_y//2, ] # remove data for y>0
                    activity[num_y//2:, ] = 0 # remove data for y>0
                elif measure_loc == 'upperright':
                    #activity = activity[num_y//2:, num_x//2:] # remove data for y<0 & x<0
                    activity = activity[:num_y//2, :num_x//2] = 0 # remove data for y<0 & x<0
                if np.abs(activity.max()) <= 1e-10:
                    print(
                        f'WARNING: iso_response_curvature_poly_fits: After isolating {measure_loc},' 
                        +'maximum value of activations for '
                        +f'neuron_index={neuron_id}, comparison_index={plane_id}, is {activity.max}'
                    )
                    sub_contours.append(np.nan)
                    sub_curvatures.append(np.nan)
                    sub_fits.append(np.nan)
                    continue
                activity = activity - activity.min() # renormalize to be between 0 and 1
                activity = activity / activity.max()
                
                try: # compute curvature
                    contours = measure.find_contours(activity, target_act)[-1] # if multiple contours are found, use the right-most
                    contours = np.concatenate(contours, axis=0) # Grab all of them together
                except:
                    print('ERROR: iso_response_curvature_poly_fits: Unable to find iso-response contours...')
                    import IPython; IPython.embed(); raise SystemExit
                    
                # Rescale contour location to actual data range
                x_contour_pts = contours[:,1] * xy_scale[0] - (num_x * xy_scale[0] / 2)
                y_contour_pts = contours[:,0] * xy_scale[1] - (num_y * xy_scale[1] / 2)
                
                # only use contour points for x > 0, which helps with circular contours
                #x_pos = np.argwhere(x_contour_pts>0) 
                #if x_pos.size > 10: # TODO: We shouldn't need this if/else. Need to investigate what is going on when x_pos.size == 0 or 1. It basically means there are only curves in the left half, which shouldn't happen?
                #    x_contour_pts_half = np.squeeze(x_contour_pts[x_pos])
                #    y_contour_pts_half = np.squeeze(y_contour_pts[x_pos])
                #else:
                #    x_contour_pts_half = x_contour_pts
                #    y_contour_pts_half = y_contour_pts
                x_contour_pts_half = x_contour_pts
                y_contour_pts_half = y_contour_pts
                
                # polyfit assumes the convexity is up/down, while we will have left/right
                # To get around this, we swap the x and y axis for the polyfit and then swap back
                # polyfit returns [c0, c1, c2], where p = c0 + c1x + c2x^2
                try:
                    coeffs = poly.polyfit(x=y_contour_pts_half, y=x_contour_pts_half, deg=2) 
                except:
                    print('ERROR: iso_response_curvature-poly_fits: polyfit failed.')
                    import IPython; IPython.embed(); raise SystemExit
                sub_contours.append((x_contour_pts, y_contour_pts))
                sub_curvatures.append(coeffs[-1])
                sub_fits.append((poly.polyval(y_contour_pts_half, coeffs), y_contour_pts_half))
            else:
                print('WARNING: iso_response_curvature_poly_fits: Maximum value of activations for '
                    +f'neuron_index={neuron_id}, comparison_index={plane_id}, is {activity.max}')
                sub_contours.append(np.nan)
                sub_curvatures.append(np.nan)
                sub_fits.append(np.nan)
        contours_list.append(sub_contours)
        curvatures.append(sub_curvatures)
        fits.append(sub_fits)
    return (curvatures, fits, contours_list)


def response_attenuation_curvature_poly_fits(activations, target_act, x_pts, y_pts):
    """
    Parameters:
        activations [tuple] first element is the comp activations returned from get_normalized_activations and the secdond element is rand activations
        x_pts [np.ndarray] of size (num_images,) that were the x points used for the contour dataset
        y_pts [np.ndarray] of size (num_images,) that were the y points used for the contour dataset
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
            x_act = np.squeeze(activity[0, :])
            closest_target_act = x_act[np.abs(x_act - target_act).argmin()] # find a location to take a slice
            try:
                x_target_index = np.argwhere(x_act == closest_target_act)[0].item() # find the index along x axis
            except:
                import IPython; IPython.embed(); raise SystemExit
            x_target = x_pts[x_target_index] # find the x value at this index
            num_y, num_x = activity.shape
            num_images = num_y * num_x
            X_mesh, Y_mesh = np.meshgrid(x_pts, y_pts)
            proj_datapoints = np.stack([X_mesh.reshape(num_images), Y_mesh.reshape(num_images)], axis=1)
            slice_indices = np.where(proj_datapoints[:, 0] == x_target)[0]
            slice_datapoints = proj_datapoints[slice_indices, :][:, :] # slice grid
            slice_activity = activity.reshape([-1])[slice_indices][:]
            sub_sliced_activity.append(slice_activity)
            x_vals = slice_datapoints[:, 1]
            y_vals = slice_activity * -1 # flip so that response attenuation gives positive curvature
            # polyfit returns [c0, c1, c2], where p = c0 + c1x + c2x^2
            coeffs = poly.polyfit(x_vals, y_vals, deg=2)
            sub_curvatures.append(coeffs[-1])
            sub_fits.append((x_vals, poly.polyval(x_vals, coeffs)))
        curvatures.append(sub_curvatures)
        fits.append(sub_fits)
        sliced_activity.append(sub_sliced_activity)
    return (curvatures, fits, sliced_activity)


def compute_curvature_poly_fits(activations, contour_dataset, target_act, measure_loc='all'):#, measure_upper_right=False):
    """
    Parameters:
        activations [np.ndarray] returned from get_normalized_activations
        contour_dataset [dict] that must contain keys "x_pts" and "y_pts" from utils/dataset_generation.get_contour_dataset()
          x_pts [np.ndarray] of size (num_images,) that were the points used for the contour dataset
          proj_datapoints [np.ndarray] of stacked vectorized datapoint locations (from a mesh grid) for the 2D contour dataset
        target_act [float] target activity for finding iso-response contours
        measure_loc [str] Choose location to measure iso-response contours
            this assumes the original range of X and Y values were from -K to K for some integer, K
            'all' -> (default) do not modify the activity map
            'left' -> only measure contours left of the (x-) middle
            'right' -> only measure contours right of the (x-) middle
            'top' -> only measure contours above the (y-) middle
            'bottom' -> only measure contours below the (y-) middle
            'upperright' -> measure contours in the top right quadrant 
        measure_upper_right [bool] if True, only measure the curvature in the upper right quadrant of each (renormalized) activity map
    outputs:
        iso_curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated iso-response curvature coefficient for the given neuron and plane
        attn_curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated response attenuation curvature coefficient for the given neuron and plane
    """
    num_target, num_planes, num_y, num_x = activations.shape 
    x_pts, y_pts = (contour_dataset['x_pts'], contour_dataset['y_pts'])
    x_range = max(x_pts) - min(x_pts)
    y_range = max(y_pts) - min(y_pts)
    x_scale_factor =  x_range / num_x
    y_scale_factor =  y_range / num_y
    iso_curvatures, iso_fits, iso_contours = iso_response_curvature_poly_fits(
        activations,
        target_act,
        [x_scale_factor, y_scale_factor],
        measure_loc)
        #measure_upper_right)
    attn_curvatures, attn_fits, attn_sliced_activity = response_attenuation_curvature_poly_fits(
        activations,
        target_act,
        x_pts,
        y_pts)
    return (iso_curvatures, attn_curvatures)


def get_relative_hist(curvatures, bins):
    """
    Compute relative histogram for curvature values,
        where the hist value indicates the number of items in that bin divided by the total number of items
    Parameters:
        curvatures [list of floats] curvature fit values to be histogrammed
        bins [sequence of scalars] defines a monotonically increasing array of bin edges, including the rightmost edge
    Outputs:
        hist [np.ndarray] normalized histogram values
        bin_edges [np.ndarray] histogram bin edges of len = len(hist)+1
    """
    hist, bin_edges = np.histogram(curvatures, bins, density=False)
    hist = hist / len(curvatures)
    return hist, bin_edges


def get_bins(num_bins, min_val, max_val):
    """
    Compute bin edges & centers for histograms
        The output will always have a bin centered at 0
    Parameters:
        num_bins [int] number of evenly-spaced bins
        min_val [int] set to force a minimum bound
        max_val [int] set to force a maximum bound
    Outputs:
        bins [np.ndarray] of size [num_bins,] with a range that includes the specified min and max
    """
    bin_width = (max_val - min_val) / (num_bins-1) # subtract 1 to leave room for the zero bin
    bin_centers = [0.0]
    while min(bin_centers) > min_val:
        bin_centers.append(bin_centers[-1] - bin_width)
    bin_centers = bin_centers[::-1]
    while max(bin_centers) < max_val:
        bin_centers.append(bin_centers[-1] + bin_width)
    bin_centers = np.array(bin_centers)
    bin_lefts = bin_centers - (bin_width / 2)
    bin_rights = bin_centers + (bin_width / 2)
    bins = np.append(bin_lefts, bin_rights[-1])
    return bins


def get_bins_from_curvatures(all_curvatures, num_bins=50):
    """
    Compute bin edges & centers for histograms by finding min and max across all values given
        The output will always have a bin centered at 0
    Parameters:
        all_curvatures [list of floats] curvature values to be binned
        num_bins [int] number of evenly-spaced bins
        min_val [int] set to force a minimum bound
        max_val [int] set to force a maximum bound
    Outputs:
        bins [np.ndarray] of size [num_bins,] with a range that includes the min and max of all_curvatures
    """
    min_val = np.amin(all_curvatures)
    max_val = np.amax(all_curvatures)
    return get_bins(num_bins, min_val, max_val)


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
    bins = get_bins_from_curvatures(all_curvatures, num_bins)
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
                neuron_hist, type_bin_edges = get_relative_hist(neuron_curvatures, type_bins)
                dataset_sub_hists.append(neuron_hist)
            type_sub_hists.append(dataset_sub_hists)
        all_hists.append(type_sub_hists)
        all_bin_edges.append(type_bin_edges)
    return [all_hists, all_bin_edges]


def compute_curvature_fits(curvatures, dist_name='gennorm'):
    """
    Compute fits for all curvatures with shared support across curvature types
    Parameters:
        curvatures [nested list of floats] that is indexed by
            [curvature type]
            [dataset type]
            [target neuron id]
            [comparison plane id]
        dist_name [str] name of the distribution for fitting.
            Must be one of the continuous distributions listed on:
            https://docs.scipy.org/doc/scipy/reference/stats.html
    Outputs:
        all_fits [nested list of (param) tuples] that is indexed by
            [curvature type]
            [dataset type]
            [target neuron id]
    """
    dist = getattr(scipy.stats, dist_name)
    all_fits = []
    for type_curvatures in curvatures:
        type_sub_fits = []
        for dataset_curvatures in type_curvatures:
            dataset_sub_fits = []
            for neuron_curvatures in dataset_curvatures:
                params = dist.fit(neuron_curvatures)
                dataset_sub_fits.append(params)
            type_sub_fits.append(dataset_sub_fits)
        all_fits.append(type_sub_fits)
    return all_fits, dist