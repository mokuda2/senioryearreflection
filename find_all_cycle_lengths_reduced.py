# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:07:28 2023

@author: kolto

finds all cycle lengths in a fundamental cycle basis, measures the KL divergence
between the distributions of lengths of fundamental cycles in the Variant Target
Model (Chapter 5) and the real-world target network AND between that of a
configuration model (based on the Variant Target Model's degree distribution)
and the real-world target network.

These KL divergences are saved in both text files and pickle objects, as are the
lists of fundamental cycle lengths. 
"""

from family_model_intergenerational_marriage_reduced import *
from get_model_parameters import separate_parts
from KL_divergence import KL_Divergence

import argparse
from distutils import util
import networkx as nx
import numpy as np

import os
import pandas as pd
import pickle

#%%

get_bool = lambda x: bool(util.strtobool(x))  # helper function to parse boolean values

def compare_model_to_configuration_model(out_dir='cycle_distributions_ver2',
                                         save_models=False,
                                         max_tries=5,
                                         num_instantiations=1,
                                         name=None,
                                         num_people=0,
                                         use_connected=False
                                         ):
    if use_connected:
        out_dir += '_connected'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'text_files')):
        os.makedirs(os.path.join(out_dir, 'text_files'))
    if not os.path.exists(os.path.join(out_dir, 'pickle_files')):
        os.makedirs(os.path.join(out_dir, 'pickle_files'))
    filename_cycle_lengths = name + '_' + 'cycle_lengths'

    target_marriage_dist, target_num_marriages, target_prob_inf_marriage, target_prob_finite_marriage, target_child_dist, target_size_goal = get_graph_stats(name)
    # build a nx graph object out of your TARGET pajek file
    target_nodes, target_union_edges, target_child_edges = separate_parts(f'Original_Sources/kinsources-{name}-oregraph.paj', 'A')
    target = nx.Graph()
    target.add_nodes_from(np.arange(1, target_nodes[0]+1))
    target.add_edges_from(target_union_edges)
    target.add_edges_from(target_child_edges)
    # target_degree_distribution = [deg for n, deg in target.degree()]
    target_cycles = nx.cycle_basis(target)
    target_cycle_lengths = [len(cycle) for cycle in target_cycles]
    print(f'found cycle distribtion for target {name}')
    # and also save the target model cycle info
    with open(os.path.join(out_dir, 'pickle_files', f'{name}_target_cycle_lengths.pkl'), 'wb') as f:
        pickle.dump(target_cycle_lengths, f)

    our_model_cycle_dist_KL_divs = []
    configuration_cycle_dist_KL_divs = []
    all_our_model_cycle_lens = []
    all_configuration_model_cycle_lens = []
    for k in range(num_instantiations):
        model_dies_out = True
        tries = 0
        while model_dies_out:
            # try several times to build a model which survives (recaptures the target_size_goal number of nodes)
            our_model_G, our_model_G_connected, this_model_marriage_edges, this_model_marriage_distances, this_model_children_per_couple, model_dies_out, this_output_path = human_family_network(num_people,
                                                                                                                                             target_marriage_dist,
                                                                                                                                             target_prob_finite_marriage,
                                                                                                                                             target_prob_inf_marriage,
                                                                                                                                             target_child_dist,
                                                                                                                                             name,
                                                                                                                                             when_to_stop=target_size_goal,
                                                                                                                                             out_dir=os.path.join(out_dir, 'models'),
                                                                                                                                             save=save_models)
            tries += 1
            if tries > max_tries:
                print(name + ' died out five times with start size {}'.format(num_people))
                break
        if model_dies_out:
            continue
        # convert from directed to undirected
        if use_connected:
            our_model_G = nx.Graph(our_model_G_connected)
        else:
            our_model_G = nx.Graph(our_model_G)

        # now store the degree distributions
        our_model_degree_dist = [deg for n, deg in our_model_G.degree()]  # for contstruction
        configuration_model = nx.configuration_model(our_model_degree_dist, create_using=nx.Graph)  # should I be removing self loops?  Allowing multiedges?

        # now compute the cycle length distribution of each
        # simple_cycles returns lists of nodes, a new list for each cycle
        our_model_cycles = nx.cycle_basis(our_model_G)
        our_model_cycle_lengths = [len(cycle) for cycle in our_model_cycles]
        all_our_model_cycle_lens.append(our_model_cycle_lengths)
        # print(f'found cycle distribtion for target model {name}: {temp}')

        configuration_cycles = nx.cycle_basis(configuration_model)
        configuration_cycle_lengths = [len(cycle) for cycle in configuration_cycles]
        all_configuration_model_cycle_lens.append(configuration_cycle_lengths)
        # print(f'found cycle distribtion for configuration model {name}: {temp}')

        # now compute the KL divergences
        our_model_cycle_dist_KL_divs.append(KL_Divergence(target_cycle_lengths, our_model_cycle_lengths))
        with open(os.path.join(out_dir, 'pickle_files', f'{name}_model_cycle_KL_div.pkl'), 'wb') as f:
            pickle.dump(our_model_cycle_dist_KL_divs, f)
        # and maybe write these numbers out to a text file for quick viewing
        with open(os.path.join(out_dir,  'text_files', name+ '_model_KL_divs' + '.txt'), 'w') as f:
            f.writelines([str(num)+'\n' for num in our_model_cycle_dist_KL_divs])

        configuration_cycle_dist_KL_divs.append(KL_Divergence(target_cycle_lengths, configuration_cycle_lengths))
        with open(os.path.join(out_dir, 'pickle_files', f'{name}_configuration_cycle_KL_div.pkl'), 'wb') as f:
            pickle.dump(configuration_cycle_dist_KL_divs, f)
        # and maybe write these numbers out to a text file for quick viewing
        with open(os.path.join(out_dir,  'text_files', name+ '_configuration_KL_divs' + '.txt'), 'w') as f:
            f.writelines([str(num)+'\n' for num in configuration_cycle_dist_KL_divs])

        # also save the actual lists of cycle lenghts
        with open(os.path.join(out_dir, 'pickle_files', f'{name}_configuration_cycle_lengths.pkl'), 'wb') as f:
            pickle.dump(all_configuration_model_cycle_lens, f)
        with open(os.path.join(out_dir, 'pickle_files', f'{name}_model_cycle_lengths.pkl'), 'wb') as f:
            pickle.dump(all_our_model_cycle_lens, f)




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-network_num', '--network_num', type=int, default=0, help="(int) corresponding to the network's index position in returned from get_graphs_and_names()")
    parser.add_argument('-name', '--name', type=str, default=None, help="give an individual graph name to run, otherwise all listed in data_set_start_sizes will run")
    parser.add_argument('-out_dir', '--out_dir', type=str, default='cycle_distributions', help='directory name to store all results')
    parser.add_argument('-num_instantiations', '--num_instantiations', type=int, default=1, help='number of instantiations PER network name')
    parser.add_argument('-path_to_start_sizes', '--path_to_start_sizes', type=str, default='data_set_start_sizes_reduced.pkl', help='full path to data_set_start_sizes.pkl')
    parser.add_argument('-connected', '--connected', type=get_bool, default=False, help="do you want to ditch the pre-generation 0 structure?")

    args = parser.parse_args()

    network_num = args.network_num
    name = args.name
    out_dir = args.out_dir
    num_instantiations = args.num_instantiations
    path_to_start_sizes = args.path_to_start_sizes
    use_connected = args.connected

    # load the start size
    start_sizes = pd.read_pickle(path_to_start_sizes)
    start_sizes = start_sizes.loc[~start_sizes.avg_start_size.isna(), ['name', 'iters', 'avg_start_size']]
    start_sizes.loc[:, 'avg_start_size'] = start_sizes.loc[:, 'avg_start_size'].astype(int)

    if name is None:
        names = start_sizes.loc[:, 'name'].unique()
        name = names[network_num]

    num_people = start_sizes.loc[start_sizes.name == name, 'avg_start_size'].values[0]
    print('name: ', name)
    print('model start size:', num_people)
    compare_model_to_configuration_model(out_dir=out_dir,
                                         save_models=False,
                                         max_tries=5,
                                         num_instantiations=num_instantiations,
                                         name=name,
                                         num_people=num_people,
                                         use_connected=use_connected)

# %%
