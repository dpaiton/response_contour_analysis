"""
Utility funcions for summarizing response contours

Authors: Dylan Paiton, Santiago Cadena
"""

import numpy as np
import numpy.polynomial.polynomial as poly
from skimage import measure


def iso_response_curvature_poly_fits(activations, target, target_is_act=True, yx_scale=[1, 1]):
    """
    Parameters:
        activations [np.ndarray] first index is the target neuron, second index is the data plane
        target [float] target activity for finding iso-response contours OR target position along x axis for finding the target activity
            if target_is_act is false, then target refers to a position on the x axis from -1 (left most point) to 1 (right most point)
        target_is_act [bool] if True, then the 'target' parameter is an activation value, else the 'target' parameter is the x axis position
        yx_scale [list of ints] y and x (respectively) scale factors for remapping activations to the data domain
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
                num_y, num_x = activity.shape
                if target_is_act:
                    target_act = target
                else:
                    # map [-1, 1] to [0, num_x-1]
                    target_pos = int((num_x - 1) * ((target + 1) / 2)) 
                    target_act = activity[(num_y//2)-1, target_pos]
                #activity[:, :(num_x//2-1)] = 0 # remove data for x<0
                if np.abs(activity.max()) <= 1e-10:
                    print(
                        f'WARNING: iso_response_curvature_poly_fits: After isolating the right half,' 
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
                    contours = measure.find_contours(activity, target_act)
                    contours = np.concatenate(contours, axis=0) # Grab all of them together
                except:
                    print('ERROR: iso_response_curvature_poly_fits: Unable to find iso-response contours...')
                    import IPython; IPython.embed(); raise SystemExit
                    
                # Rescale contour location to actual data range
                y_contour_pts = contours[:,0] * yx_scale[0] - (num_y * yx_scale[0] / 2)
                x_contour_pts = contours[:,1] * yx_scale[1] - (num_x * yx_scale[1] / 2)
                
                # Don't use points that lie along the y axis, to fix edge artifacts
                x_pos = np.argwhere(x_contour_pts>0) 
                if x_pos.size > 3:
                    y_contour_pts_proc = np.squeeze(y_contour_pts[x_pos])
                    x_contour_pts_proc = np.squeeze(x_contour_pts[x_pos])
                else:
                    y_contour_pts_proc = y_contour_pts
                    x_contour_pts_proc = x_contour_pts
                
                # polyfit assumes the convexity is up/down, while we will have left/right
                # To get around this, we swap the x and y axis for the polyfit and then swap back
                # polyfit returns [c0, c1, c2], where p = c0 + c1x + c2x^2
                try:
                    coeffs = poly.polyfit(x=y_contour_pts_proc, y=x_contour_pts_proc, deg=2) 
                except:
                    print('ERROR: iso_response_curvature-poly_fits: polyfit failed.')
                    import IPython; IPython.embed(); raise SystemExit
                sub_contours.append((x_contour_pts_proc, y_contour_pts_proc))
                sub_curvatures.append(coeffs[-1])
                sub_fits.append((poly.polyval(y_contour_pts_proc, coeffs), y_contour_pts_proc))
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


def response_attenuation_curvature_poly_fits(activations, target, target_is_act, x_pts, y_pts):
    """
    Parameters:
        activations [tuple] first element is the comp activations returned from get_normalized_activations and the secdond element is rand activations
        x_pts [np.ndarray] of size (num_images,) that were the x points used for the contour dataset
        y_pts [np.ndarray] of size (num_images,) that were the y points used for the contour dataset
        target [float] target activity for finding iso-response contours OR target position along x axis for finding the target activity
            if target_is_act is false, then target refers to a position on the x axis from -1 (left most point) to 1 (right most point)
        target_is_act [bool] if True, then the 'target' parameter is an activation value, else the 'target' parameter is the x axis position
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
            num_y, num_x = activity.shape
            # TODO: Verify this fix was correct. first index was originally 0, which would be along the top
            x_act = np.squeeze(activity[num_y//2, :])
            if target_is_act:
                closest_target_act = x_act[np.abs(x_act - target).argmin()] # find a location to take a slice
            else:
                target_pos = int(num_x * ((target + 1) / 2)) # map [-1, 1] to [0, num_x]
                closest_target_act = x_act[target_pos]
            try:
                x_target_index = np.argwhere(x_act == closest_target_act)[0].item() # find the index along x axis
            except:
                print('Warning: slicing activity matrix failed.')
                import IPython; IPython.embed(); raise SystemExit
            x_target = x_pts[x_target_index] # find the x value at this index
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


def compute_curvature_poly_fits(activations, contour_dataset, target, target_is_act=True, bounds=None):
    """
    Parameters:
        activations [np.ndarray] returned from get_normalized_activations
        contour_dataset [dict] that must contain keys "x_pts" and "y_pts" from utils/dataset_generation.get_contour_dataset()
          x_pts [np.ndarray] of size (num_images,) that were the points used for the contour dataset
          proj_datapoints [np.ndarray] of stacked vectorized datapoint locations (from a mesh grid) for the 2D contour dataset
        target [float] target activity for finding iso-response contours OR target position along x axis for finding the target activity
            if target_is_act is false, then target refers to a position on the x axis from -1 (left most point) to 1 (right most point)
        target_is_act [bool] if True, then the 'target' parameter is an activation value, else the 'target' parameter is the x axis position
        bounds [nested tuple] containing ((y_min, y_max), (x_min, x_max)) for the window within which curvature should be measured
    outputs:
        iso_curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated iso-response curvature coefficient for the given neuron and plane
        attn_curvatures [list of lists] of lengths [num_neurons, num_planes] which contain the estimated response attenuation curvature coefficient for the given neuron and plane
    """
    num_target, num_planes, num_y, num_x = activations.shape 
    yx_pts = (contour_dataset['y_pts'].copy(), contour_dataset['x_pts'].copy())
    y_pts, x_pts = yx_pts
    y_range = max(y_pts) - min(y_pts)
    x_range = max(x_pts) - min(x_pts)
    if bounds is None: 
        y_scale_factor =  y_range / num_y
        x_scale_factor =  x_range / num_x
        start_y = start_x = 0
        end_y = num_y
        end_x = num_x
    else: # assumes input is square, centered around origin, and the bounds are also centered
        y_bounds, x_bounds = bounds
        y_bound_range = max(y_bounds) - min(y_bounds)
        x_bound_range = max(x_bounds) - min(x_bounds)
        y_trim = 0.5 * y_bound_range / y_range
        x_trim = 0.5 * x_bound_range / x_range
        start_y = int(np.floor(y_trim*num_y))
        end_y = int(np.ceil(3*y_trim*num_y))
        start_x = int(np.floor(x_trim*num_x))
        end_x = int(np.ceil(3*x_trim*num_x))
        new_num_y = end_y - start_y
        new_num_x = end_x - start_x
        y_pts_trim = y_pts[start_y:end_y]
        x_pts_trim = x_pts[start_x:end_x]
        new_y_range = max(y_pts_trim) - min(y_pts_trim)
        new_x_range = max(x_pts_trim) - min(x_pts_trim)
        y_scale_factor =  new_y_range / new_num_y
        x_scale_factor =  new_x_range / new_num_x
    iso_curvatures, iso_fits, iso_contours = iso_response_curvature_poly_fits(
        activations[:, :, start_y:end_y, start_x:end_x],
        target,
        target_is_act,
        [y_scale_factor, x_scale_factor])
    attn_curvatures, attn_fits, attn_sliced_activity = response_attenuation_curvature_poly_fits(
        activations[:, :, start_y:end_y, start_x:end_x],
        target,
        target_is_act,
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