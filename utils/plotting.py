"""
Utility funcions for plotting response analysis

Authors: Dylan Paiton, Santiago Cadena
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import proplot as pro


def clear_axis(ax, spines='none'):
    for ax_loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[ax_loc].set_color(spines)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='both', bottom=False, top=False, left=False, right=False)
    return ax


def set_size(width, fraction=1, subplot=[1, 1]):
    """
    Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters:
        width: float
                Width in pts
        fraction: float
                Fraction of the width which you wish the figure to occupy
    Returns:
      fig_dim: tuple
              Dimensions of figure in inches
    Usage:
        figsize = set_size(text_width, fraction=1, subplot=[1, 1])
    Code obtained from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    fig_width_pt = width * fraction # Width of figure
    inches_per_pt = 1 / 72.27 # Convert from pt to inches
    golden_ratio = (5**.5 - 1) / 2 # Golden ratio to set aesthetic figure height
    fig_width_in = fig_width_pt * inches_per_pt # Figure width in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1]) # Figure height in inches
    fig_dim = (fig_width_in, fig_height_in) # Final figure dimensions
    return fig_dim


def plot_group_iso_contours(analysis_dict, num_levels, show_contours=True, targets_comparisons=[2,3], text_width=200, width_fraction=1.0, dpi=100):
    arrow_width = 0.0
    arrow_linewidth = 1
    arrow_headsize = 0.15
    arrow_head_length = 0.15
    arrow_head_width = 0.15
    gs0_hspace = 0.5
    gs0_wspace = -0.6
    phi_k_text_x_offset = 0.6 / width_fraction
    phi_k_text_y_offset = -1.2 / width_fraction
    phi_j_text_x_offset = 0.9 / width_fraction
    phi_j_text_y_offset = 0.3 / width_fraction
    nu_text_x_offset = -0.56 / width_fraction
    nu_text_y_offset = 0.3 / width_fraction
    num_plots_y = targets_comparisons[0] # num target neurons
    num_plots_x = targets_comparisons[1] # num comparison planes
    gs0 = gridspec.GridSpec(num_plots_y, num_plots_x, wspace=gs0_wspace, hspace=gs0_hspace)
    vmin = np.min(analysis_dict['activations'])
    vmax = np.max(analysis_dict['activations'])
    levels = np.linspace(vmin, vmax, num_levels)
    cmap = plt.get_cmap('cividis')
    cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    fig = plt.figure(figsize=set_size(text_width, width_fraction, [num_plots_y, num_plots_x]), dpi=dpi)
    contour_handles = []
    curve_axes = []
    for plot_id in np.ndindex((num_plots_y, num_plots_x)):
        (y_id, x_id) = plot_id
        neuron_index = y_id
        orth_index = x_id
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 1, gs0[plot_id])
        curve_axes.append(clear_axis(fig.add_subplot(inner_gs[0])))
        #curve_axes[-1].set_title(str(plot_id)) # TODO: change title
        # plot colored mesh points
        norm_activity = analysis_dict['activations'][neuron_index, orth_index, ...]
        x_mesh, y_mesh = np.meshgrid(
            analysis_dict['contour_dataset']['x_pts'],
            analysis_dict['contour_dataset']['y_pts'])
        if show_contours:
            contsf = curve_axes[-1].contourf(x_mesh, y_mesh, norm_activity,
                levels=levels, vmin=vmin, vmax=vmax, alpha=1.0, antialiased=True, cmap=cmap)
        else:
            contsf = curve_axes[-1].scatter(x_mesh, y_mesh,
                vmin=vmin, vmax=vmax, cmap=cmap, marker='s', alpha=1.0, c=norm_activity, s=30.0)
        contour_handles.append(contsf)
        # plot target neuron arrow & label
        proj_target = analysis_dict['contour_dataset']['proj_target_vect'][neuron_index][orth_index]
        target_vector_x = proj_target[0].item()
        target_vector_y = proj_target[1].item()
        curve_axes[-1].arrow(0, 0, target_vector_x, target_vector_y,
            width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
            fc='k', ec='k', linestyle='-', linewidth=arrow_linewidth)
        tenth_range_shift = ((max(analysis_dict['x_range']) - min(analysis_dict['x_range']))/10) # For shifting labels
        text_handle = curve_axes[-1].text(
            target_vector_x+(tenth_range_shift*phi_k_text_x_offset),
            target_vector_y+(tenth_range_shift*phi_k_text_y_offset),
            r'$\Phi_{k}$', horizontalalignment='center', verticalalignment='center')
        # plot comparison neuron arrow & label
        proj_comparison = analysis_dict['contour_dataset']['proj_comparison_vect'][neuron_index][orth_index]
        comparison_vector_x = proj_comparison[0].item()
        comparison_vector_y = proj_comparison[1].item()
        curve_axes[-1].arrow(0, 0, comparison_vector_x, comparison_vector_y,
            width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
            fc='k', ec='k', linestyle='-', linewidth=arrow_linewidth)
        text_handle = curve_axes[-1].text(
            comparison_vector_x+(tenth_range_shift*phi_j_text_x_offset),
            comparison_vector_y+(tenth_range_shift*phi_j_text_y_offset),
            r'$\Phi_{j}$', horizontalalignment='center', verticalalignment='center')
        # Plot orthogonal vector Nu
        proj_orth = analysis_dict['contour_dataset']['proj_orth_vect'][neuron_index][orth_index]
        orth_vector_x = proj_orth[0].item()
        orth_vector_y = proj_orth[1].item()
        curve_axes[-1].arrow(0, 0, orth_vector_x, orth_vector_y,
            width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
            fc='k', ec='k', linestyle='-', linewidth=arrow_linewidth)
        text_handle = curve_axes[-1].text(
            orth_vector_x+(tenth_range_shift*nu_text_x_offset),
            orth_vector_y+(tenth_range_shift*nu_text_y_offset),
            r'$\nu$', horizontalalignment='center', verticalalignment='center')
        # Plot axes
        curve_axes[-1].set_aspect('equal')
        curve_axes[-1].plot(analysis_dict['x_range'], [0,0], color='k', linewidth=arrow_linewidth/2)
        curve_axes[-1].plot([0,0], analysis_dict['y_range'], color='k', linewidth=arrow_linewidth/2)
    # Add colorbar
    scalarMap._A = []
    cbar_ax = inset_axes(curve_axes[-1],
        width='5%',
        height='100%',
        loc='lower left',
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=curve_axes[-1].transAxes,
        borderpad=0,
        )
    cbar = fig.colorbar(scalarMap, cax=cbar_ax, ticks=[vmin, vmax])
    cbar.ax.tick_params(labelleft=False, labelright=True, left=False, right=True)
    cbar.ax.set_yticklabels(['{:.0f}'.format(vmin), '{:.0f}'.format(vmax)])
    plt.show()
    return fig, contour_handles


def plot_curvature_histograms(hist_list, label_list, color_list, bin_centers, title, xlabel,
                              text_width=200, width_ratio=1.0, dpi=100):
    """
    hist_list [nested list of floats] containing histogram values that is indexed by
        [curvature type]
        [dataset type]
        [target neuron id]
        [num_bins - 1]
    label_list [nested list of strings] containing labels that shares its structure with hist_list
    color_list [nested list of strings] containing color hash values that shares its structure with hist_list
    bin_centers [nested list of floats] containing bin center values that is indexed by
        [curvature type]
        [num_bins]
    title [string]
    xlabel [string]
    """
    gs0_wspace = 0.5
    hspace_hist = 0.7
    wspace_hist = 0.10
    num_y_plots = 1
    num_x_plots = 1
    fig = plt.figure(figsize=set_size(text_width, width_ratio, [num_y_plots, num_x_plots]), dpi=dpi)
    gs_base = gridspec.GridSpec(num_y_plots, num_x_plots, wspace=gs0_wspace)
    num_hist_y_plots = 2
    num_hist_x_plots = 2
    gs_hist = gridspec.GridSpecFromSubplotSpec(num_hist_y_plots, num_hist_x_plots, gs_base[0],
        hspace=hspace_hist, wspace=wspace_hist)
    orig_ax = fig.add_subplot(gs_hist[0,0])
    axes = []
    for sub_plt_y in range(0, num_hist_y_plots):
        axes.append([])
        for sub_plt_x in range(0, num_hist_x_plots):
            if (sub_plt_x, sub_plt_y) == (0,0):
                axes[sub_plt_y].append(orig_ax)
            else:
                axes[sub_plt_y].append(fig.add_subplot(gs_hist[sub_plt_y, sub_plt_x], sharey=orig_ax))
    all_x_lists = zip(hist_list, label_list, color_list, bin_centers, title)
    for axis_x, (type_hist, sub_label, sub_color, sub_bins, sub_title) in enumerate(all_x_lists):
        max_hist_val = 0.001
        min_hist_val = 100
        all_y_lists = zip(type_hist, sub_label, sub_color, xlabel)
        for axis_y, (dataset_hist, axis_labels, axis_colors, sub_xlabel) in enumerate(all_y_lists):
            axes[axis_y][axis_x].spines['top'].set_visible(False)
            axes[axis_y][axis_x].spines['right'].set_visible(False)
            axes[axis_y][axis_x].set_xticks(sub_bins, minor=True)
            axes[axis_y][axis_x].set_xticks(sub_bins[::int(len(sub_bins)/4)], minor=False)
            axes[axis_y][axis_x].xaxis.set_major_formatter(plticker.FormatStrFormatter('%0.3f'))
            for hist, label, color in zip(dataset_hist, axis_labels, axis_colors):
                axes[axis_y][axis_x].plot(sub_bins, hist, color=color, linestyle='-',
                    drawstyle='steps-mid', label=label)
                axes[axis_y][axis_x].set_yscale('log')
                if np.max(hist) > max_hist_val:
                    max_hist_val = np.max(hist)
                if np.min(hist) < min_hist_val:
                    min_hist_val = np.min(hist)
            axes[axis_y][axis_x].axvline(0.0, color='black', linestyle='dashed', linewidth=1)
            if axis_y == 0:
                axes[axis_y][axis_x].set_title(sub_title)
            axes[axis_y][axis_x].set_xlabel(sub_xlabel)
            if axis_x == 0:
                axes[axis_y][axis_x].set_ylabel('Relative\nFrequency')
                ax_handles, ax_labels = axes[axis_y][axis_x].get_legend_handles_labels()
                legend = axes[axis_y][axis_x].legend(handles=ax_handles, labels=ax_labels,
                    loc='upper right', ncol=1, borderaxespad=0., borderpad=0.,
                    handlelength=0., columnspacing=-0.5, labelspacing=0., bbox_to_anchor=(0.95, 0.95))
                legend.get_frame().set_linewidth(0.0)
                for text, color in zip(legend.get_texts(), axis_colors):
                    text.set_color(color)
                for item in legend.legendHandles:
                    item.set_visible(False)
            if axis_x == 1:
                axes[axis_y][axis_x].tick_params(axis='y', labelleft=False)
        #axes[axis_y][axis_x].set_ylim([0, 1.0])
    plt.show()
    return fig
