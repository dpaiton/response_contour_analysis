# TODO:
# we should have a function that makes the activity maps & separate that from the functions that do curve fits or whatever. Since the activity maps are the most time consuming part of the process.
#

import os
import sys

import pickle
import numpy as np

import utils.model_handling as model_funcs
import utils.dataset_generation as iso_data
import utils.histogram_analysis as hist_funcs

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
if parent_path not in sys.path: sys.path.append(parent_path)
santi_path = parent_path+'/santi_iso_response'
santi_etc_path = os.path.join(santi_path, 'etc')
from santi_iso_response.iso_response import utils as santi_utils

# NOTE:This script uses about 19GB of memory with batch_size=100; num_images=900; num_targets=166; num_comparisons=166

with open(os.path.join(santi_etc_path, 'meis.pkl'), 'rb') as g:
    mei = pickle.load(g)
num_mei = len(mei['images'])
neuron_mei = [mei['images'][idx] for idx in range(num_mei)]
target_neuron_ids = [idx for idx in range(num_mei) if mei['performance'][idx] > 0.01]
total_num_neurons = len(neuron_mei)

experiment_params = dict()
experiment_params['batch_size'] = 100
experiment_params['target_neuron_ids'] = target_neuron_ids
experiment_params['num_comparisons'] = len(target_neuron_ids)
experiment_params['min_angle'] = 15
experiment_params['x_range'] = (-2.0, 2.0)
experiment_params['y_range'] = (-2.0, 2.0)
experiment_params['num_images'] = int(30**2)
experiment_params['image_scale'] = 12
experiment_params['target_activity'] = 0.5
experiment_params['iso_window_bounds'] = ((-1, 1), (-1, 1))
experiment_params['comp_method'] = 'closest'
experiment_params['output_directory'] = parent_path+'/iso_analysis/'
experiment_params['save_prefix'] = 'santi_windowed_'

def get_curvatures_from_target_comparison_vectors(model, target_vectors, comparison_vectors, act_func, kwargs):
    contour_dataset, _ = iso_data.get_contour_dataset(
        target_vectors,
        comparison_vectors,
        kwargs['x_range'],
        kwargs['y_range'],
        kwargs['num_images'],
        kwargs['image_scale'],
        return_datapoints=False
    )
    num_neurons = len(target_vectors)
    num_planes = len(comparison_vectors)
    num_edge_images = int(np.sqrt(kwargs['num_images']))
    response_images = np.zeros((num_neurons, num_planes, num_edge_images, num_edge_images))
    all_iso_curvatures = []
    all_attn_curvatures = []
    all_mei_lengths = []
    for target_index, target_proj_matrix in enumerate(contour_dataset['proj_matrix']):
        sub_mei_lengths = []
        sub_iso_curvatures = []
        sub_attn_curvatures = []
        for plane_index, proj_matrix in enumerate(target_proj_matrix): # 1 plane at a time
            datapoints = iso_data.inject_data(proj_matrix, contour_dataset['proj_datapoints'],
                kwargs['image_scale'])
            activations = model_funcs.get_normalized_activations(
                model,
                [kwargs['target_neuron_ids'][target_index]],
                [[datapoints]],
                act_func
            )
            response_images[target_index, plane_index, ...] = np.squeeze(activations).copy()
            
            
            iso_curvatures, attn_curvatures = hist_funcs.compute_curvature_poly_fits(
                activations,
                contour_dataset,
                kwargs['target_activity'],
                bounds=kwargs['iso_window_bounds']
                #measure_loc='right'
                #measure_upper_right=False
            )
            ## Add code to measure mei length
            #mei = kwargs['mei']
            # mei_lengths = get_mei_lengths(mei, proj_matrix)
            #     -- normalize mei
            #     -- project mei using proj_matrix
            mei_lengths = None
            sub_mei_lengths.append(mei_lengths)
            sub_iso_curvatures.append(iso_curvatures[0][0])
            sub_attn_curvatures.append(attn_curvatures[0][0])
        all_mei_lengths.append(sub_mei_lengths)
        all_iso_curvatures.append(sub_iso_curvatures)
        all_attn_curvatures.append(sub_attn_curvatures)
    out_dict = {
        'mei_lengths':all_mei_lengths,
        'contour_dataset':contour_dataset,
        'iso_curvatures':all_iso_curvatures,
        'response_images': response_images,
        'attn_curvatures':all_attn_curvatures
    }
    return out_dict

def get_curvatures_from_exciting_vectors(model, exciting_vectors, act_func, kwargs):
    iso_vectors = iso_data.compute_comp_vectors(
        exciting_vectors,
        kwargs['target_neuron_ids'],
        kwargs['min_angle'],
        kwargs['num_comparisons'],
        comp_method=kwargs['comp_method']
    )
    comparison_vector_ids = iso_vectors[0]
    target_vectors = iso_vectors[1]
    comparison_vectors = iso_vectors[2]
    kwargs['mei'] = exciting_vectors
    out_dict = get_curvatures_from_target_comparison_vectors(
        model,
        target_vectors,
        comparison_vectors,
        act_func,
        kwargs
    )
    out_dict['comparison_vector_ids'] = comparison_vector_ids
    out_dict['target_vectors'] = target_vectors
    out_dict['comparison_vectors'] = comparison_vectors
    return out_dict

if not os.path.exists(experiment_params['output_directory']):
    os.makedirs(experiment_params['output_directory'])
np.savez(experiment_params['output_directory']+experiment_params['save_prefix']+'meis_params.npz',
    data=experiment_params)

model = santi_utils.load_model()

"""
Maximum Exciting Images
"""
comp_results = get_curvatures_from_exciting_vectors(
    model,
    neuron_mei,
    santi_utils.get_activations_cell,
    experiment_params
)
np.savez(experiment_params['output_directory']+experiment_params['save_prefix']+'meis.npz',
    data=comp_results)
print('Maximum exciting images experiment complete.')

#"""
#Random planes from MEIs
#"""
#iso_vectors = iso_data.compute_rand_vectors(
#    [neuron_mei[idx] for idx in experiment_params['target_neuron_ids']],
#    experiment_params['num_comparisons']
#)
#target_vectors = iso_vectors[0]
#orth_vectors = iso_vectors[1]
#comp_results = get_curvatures_from_target_comparison_vectors(
#    model,
#    target_vectors,
#    orth_vectors,
#    santi_utils.get_activations_cell,
#    experiment_params
#)
#comp_results['target_vectors'] = target_vectors
#comp_results['comparison_vectors'] = orth_vectors
#np.savez(experiment_params['output_directory']+experiment_params['save_prefix']+'rand.npz',
#    data=comp_results)
#print('Random plane experiment complete.')

#"""
#Maximum Exciting Stimuli
#"""
#iid_stim_images = np.squeeze(model.data.train()[0])
#num_stim = iid_stim_images.shape[0]
#stim_activity = santi_utils.get_activations(model, iid_stim_images)
#stim_argsort = np.argsort(stim_activity, axis=0)
#neuron_mes = [] # most exciting stimulus
#for neuron_idx in range(total_num_neurons):
#    most_exciting_stim = iid_stim_images[stim_argsort[0, neuron_idx], ...]
#    most_exciting_stim = iso_data.normalize_vector(most_exciting_stim)
#    neuron_mes.append(most_exciting_stim)
#del iid_stim_images
#"""
#### PROBLEM - Some neurons have the same exciting stim.
#If this is the case we should assign the second most exciting stim.
#But then we have an assignment problem...
#Not sure what to do here.
#"""
#comp_results = get_curvatures_from_exciting_vectors(
#    model,
#    neuron_mes,
#    santi_utils.get_activations_cell,
#    experiment_params
#)
#np.savez(experiment_params['output_directory']+experiment_params['save_prefix']+'stim.npz',
#    data=comp_results)
#print('Maximum exciting stimuli experiment complete.')
