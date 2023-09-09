# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:48:02 2023

@author: kolto

This file is to be run after find_start_sizes.py or find_start_sizes_reduced.py.
It will instantiate and save models, their distance to union distributions, and
their children per household distributions for quick comparison and is useful
when evaluating changes to our model.

by default, results will be save in ./data_set_stats/review_of_start_sizes_{int}.
each model will have its own folder and will also save histograms of its distributions to
./data_set_stats/review_of_start_sizes_{int}/model_specific_results/marriage_distance_histograms
and to
./data_set_stats/review_of_start_sizes_{int}/model_specific_results/child_distance_histograms

num_instantiations is the number of successful model instantiations to include in
box-whisker plots

IF YOU WANT TO USE THE TARGET MODEL (CHAPTER 4) THEN YOU MUST UNCOMMENT
from family_model_intergenerational_marriage import *

IF YOU WANT TO USE THE VARIANT TARGET MODEL (CHAPTER 5) THEN YOU MUST UNCOMMENT
# from family_model_intergenerational_marriage_reduced import *
"""

import pandas as pd
import numpy as np
import os
from get_model_parameters import get_names
from family_model_intergenerational_marriage import *
# from family_model_intergenerational_marriage_reduced import *
from get_model_parameters import find_model_marriage_child_distributions_from_paj, get_marriage_distances_kolton
from KL_divergence import KL_Divergence
import ast
import pickle
from matplotlib import pyplot as plt

def find_files(out_directory, filename, ):
    ver = 1
    output_dir = os.path.join(out_directory, filename + '_')
    file_list = []
    while os.path.exists(output_dir+str(ver)):
        if os.path.exists(output_dir+str(ver)):
            file_list.append(output_dir+str(ver))
        ver += 1

    return file_list

#%%

def evaluate_start_size_searches(start_size_dir='./start_size', save_report_to='data_set_stats', save_most_recent=True):
    report = pd.DataFrame(columns=['name', 'ran_to_completion', 'iters', 'avg_start_size'])

    names = get_names(save_paths=False)
    names.remove('warao')

    # grab the average start size reported (IE the smallest start size to recover target network size >95% of the time)
    completed_networks = set()
    partially_completed_networks = set()
    no_info_networks = set()
    for name in names:
        # attempt to load the pkl file (repeatedly_call_start_size only saves a pkl file if the run went to completion (all iterations specified, without an error))
        file_list = find_files(start_size_dir, name)
        for folder in file_list:
            with open(os.path.join(folder, name+'_start_size.txt')) as f:
                start_size_lists = [ast.literal_eval(row.strip()) for row in f.readlines()]
            num_iters = len(start_size_lists)
            if num_iters == 50:  # sorry for the hardcoding...
                completed_networks.add(name)
                avg_start_size = avg_start_size = int(np.round(np.mean([row[-1] for row in start_size_lists])))
                report = pd.concat([report, pd.Series({'name':name, 'ran_to_completion':True, 'iters':num_iters, 'avg_start_size':avg_start_size}).to_frame().T], ignore_index=True)
            elif num_iters > 0:
                partially_completed_networks.add(name)
                avg_start_size = avg_start_size = int(np.round(np.mean([row[-1] for row in start_size_lists])))
                report = pd.concat([report, pd.Series({'name':name, 'ran_to_completion':False, 'iters':num_iters, 'avg_start_size':avg_start_size}).to_frame().T], ignore_index=True)
            else:
                no_info_networks.add(name)
                report = pd.concat([report, pd.Series({'name':name, 'ran_to_completion':False, 'iters':num_iters, 'avg_start_size':np.nan}).to_frame().T], ignore_index=True)

    # save the start sizes report
    # report = report.drop_duplicates()
    report.loc[:, 'iters'] = pd.to_numeric(report.loc[:, 'iters'])
    report.loc[:, 'avg_start_size'] = pd.to_numeric(report.loc[:, 'avg_start_size'])
    best_run_indices = []
    most_recent_indices = []
    # keep only the version with the most iterations
    temp = report.groupby('name')

    if not save_most_recent:
        for name, group in temp:
            # best_start_size[name] = report.iloc[group.iters.idxmax()].avg_start_size
            best_run_indices.append(group.iters.idxmax())
        report = report.iloc[best_run_indices].copy()
    else:
        for name, group in temp:
            # best_start_size[name] = report.iloc[group.iters.idxmax()].avg_start_size
            most_recent_indices.append(group.index[-1])
        report = report.iloc[most_recent_indices].copy()
    # report.loc[:, 'avg_start_size'] = report.loc[:, 'avg_start_size'].astype(int)
    partially_completed_networks = {k for k in partially_completed_networks if k not in completed_networks}


    out_dir = makeOutputDirectory(save_report_to, 'review_of_start_sizes')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with pd.ExcelWriter(os.path.join(out_dir, 'data_set_start_sizes.xlsx'), engine='xlsxwriter') as writer:
        report.to_excel(writer)
    report.to_pickle(os.path.join(out_dir, 'data_set_start_sizes.pkl'))

    with open(os.path.join(out_dir, 'successful_networks.txt'), 'w') as o:
       for n in sorted(list(completed_networks)):
           o.write(n)
           o.write('\n')
    with open(os.path.join(out_dir, 'successful_networks.pkl'), 'wb') as o:
        pickle.dump(sorted(list(completed_networks)), o)

    with open(os.path.join(out_dir, 'partially_completed_networks.txt'), 'w') as o:
       # those that ran for some but not all specified iterations
        for n in sorted(list(partially_completed_networks)):
           o.write(n)
           o.write('\n')
    with open(os.path.join(out_dir, 'partially_completed_networks.pkl'), 'wb') as o:
        pickle.dump(sorted(list(partially_completed_networks)), o)

    with open(os.path.join(out_dir, 'did_not_run_networks.txt'), 'w') as o:
        # text file was created, but never written to
        for n in sorted(list(no_info_networks)):
            o.write(n)
            o.write('\n')
    with open(os.path.join(out_dir, 'did_not_run_networks.pkl'), 'wb') as o:
        pickle.dump(sorted(list(no_info_networks)), o)


#%%
def measure_KL_div(iters=1, use_latest=True, path_to_start_sizes_pkl=os.path.join('data_set_stats', 'review_of_start_sizes_1'), save_models=False):
    paths = find_files('data_set_stats', 'review_of_start_sizes')
    if use_latest:
        path_to_start_sizes_pkl = paths[-1]
        out_dir = paths[-1]
    else:
        out_dir = path_to_start_sizes_pkl.copy()
    path_to_start_sizes_pkl = os.path.join(path_to_start_sizes_pkl, 'data_set_start_sizes.pkl')
    # load the start size
    start_sizes = pd.read_pickle(path_to_start_sizes_pkl)
    start_sizes = start_sizes.loc[~start_sizes.avg_start_size.isna(), ['name', 'iters', 'avg_start_size']]
    start_sizes.loc[:, 'avg_start_size'] = start_sizes.loc[:, 'avg_start_size'].astype(int)
    start_sizes.loc[:, 'iters'] = pd.to_numeric(start_sizes.loc[:, 'iters'])
    # add columns to be calculated
    start_sizes.loc[:, 'marriage_KL_div'] = np.nan
    start_sizes.loc[:, 'child_KL_div'] = np.nan
    start_sizes.loc[:, 'iterations'] = iters
    names = start_sizes.loc[:, 'name'].unique()
    for name in names:

        num_people = start_sizes.loc[start_sizes.loc[start_sizes.name==name, 'iters'].idxmax(), 'avg_start_size']  # this grabs the (first) run with the greatest number of iterations
        marriage_KL_div = 0.
        child_KL_div = 0.
        target_marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, target_child_dist, size_goal = get_graph_stats(name)
        for i in range(iters):
            G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out = human_family_network(num_people, target_marriage_dist, prob_finite_marriage, prob_inf_marriage, target_child_dist, name, when_to_stop=size_goal, save=save_models)
            marriage_KL_div += KL_Divergence(all_marriage_distances, target_marriage_dist)
            child_KL_div += KL_Divergence(all_children_per_couple, target_child_dist)
        marriage_KL_div /= iters
        child_KL_div /= iters
        start_sizes.loc[start_sizes.name==name, 'marriage_KL_div'] = marriage_KL_div
        start_sizes.loc[start_sizes.name==name, 'child_KL_div'] = child_KL_div

    # now save the output

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with pd.ExcelWriter(os.path.join(out_dir, 'kl_divergences_with_start_sizes.xlsx'), engine='xlsxwriter') as writer:
        start_sizes.to_excel(writer)
    start_sizes.to_pickle(os.path.join(out_dir, 'kl_divergences_with_start_sizes.pkl'))


#%%
def visualize_distribution_histograms(use_latest=True, path_to_start_sizes_pkl=os.path.join('data_set_stats', 'review_of_start_sizes_1'), save_models=True, save_plots=True, marriage_density=True, child_density=True, side_by_side=True):
    paths = find_files('data_set_stats', 'review_of_start_sizes')
    if use_latest:
        path_to_start_sizes_pkl = paths[-1]
        out_dir = paths[-1]
    else:
        out_dir = path_to_start_sizes_pkl.copy()

    if save_models or save_plots:
        out_dir = os.path.join(out_dir, 'model_specific_results')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if save_plots:
        marriage_hist_path = os.path.join(out_dir, 'marriage_distance_histograms')
        if not os.path.exists(marriage_hist_path):
            os.makedirs(marriage_hist_path)

        child_hist_path = os.path.join(out_dir, 'child_distance_histograms')
        if not os.path.exists(child_hist_path):
            os.makedirs(child_hist_path)

    path_to_start_sizes_pkl = os.path.join(path_to_start_sizes_pkl, 'data_set_start_sizes.pkl')
    # load the start size
    start_sizes = pd.read_pickle(path_to_start_sizes_pkl)
    start_sizes = start_sizes.loc[~start_sizes.avg_start_size.isna(), ['name', 'iters', 'avg_start_size']]
    start_sizes.loc[:, 'avg_start_size'] = start_sizes.loc[:, 'avg_start_size'].astype(int)
    names = start_sizes.loc[:, 'name'].unique()

    for name in names:
        num_people = start_sizes.loc[start_sizes.name == name, 'avg_start_size'].values[0]
        target_marriage_dist, target_num_marriages, target_prob_inf_marriage, target_prob_finite_marriage, target_child_dist, target_size_goal = get_graph_stats(name)
        model_dies_out = True
        max_tries = 5
        tries = 0
        while model_dies_out:
            # try several times to build a model which survives (recaptures the target_size_goal number of nodes)
            G, model_marriage_edges, model_marriage_distances, model_children_per_couple, model_dies_out, output_path = human_family_network(num_people,
                                                                                                                                             target_marriage_dist,
                                                                                                                                             target_prob_finite_marriage,
                                                                                                                                             target_prob_inf_marriage,
                                                                                                                                             target_child_dist,
                                                                                                                                             name,
                                                                                                                                             when_to_stop=target_size_goal,
                                                                                                                                             out_dir=out_dir,
                                                                                                                                             save=save_models)
            tries += 1
            if tries > max_tries:
                print(name + ' died out five times with start size {}'.format(num_people))
                break

        # calculate distances/children per couple from the pajek file
        # NOTE: this uses ONLY the ACTUAL network structure and ignores the artificially-imposed distances from our model's initial generation setup
        paj_marriage_distances, paj_num_inf_marriages, paj_percent_inf_marriages, paj_children_per_couple = find_model_marriage_child_distributions_from_paj(os.path.join(output_path, 'model-'+name+'-oregraph.paj'))

        actual_marriage_KL_div = KL_Divergence(model_marriage_distances, target_marriage_dist)
        actual_child_KL_div = KL_Divergence(model_children_per_couple, target_child_dist)

        actual_marriage_KL_div = KL_Divergence(paj_marriage_distances, target_marriage_dist)
        actual_child_KL_div = KL_Divergence(paj_children_per_couple, target_child_dist)

        model_marriage_distances = np.array(model_marriage_distances)
        paj_marriage_distances = np.array(paj_marriage_distances)
        target_marriage_dist = np.array(target_marriage_dist)

        model_percent_inf_marriages = sum(model_marriage_distances == -1) / len(model_marriage_distances)
        target_percent_inf_marriages = sum(target_marriage_dist == -1) / len(target_marriage_dist)
        # paj_percent_inf_marriages defined above

        if len(model_marriage_distances[model_marriage_distances != -1]) != 0:
            # creates histogram of marriage distributions excluding infinite distance unions
            model_max_bin = int(np.max(model_marriage_distances[model_marriage_distances != -1]))
            paj_max_bin = int(np.max(paj_marriage_distances[paj_marriage_distances != -1]))
            target_max_bin = int(np.max(target_marriage_dist[target_marriage_dist != -1]))
            max_bin = max(model_max_bin, paj_max_bin, target_max_bin)

            fig = plt.figure(figsize=(12,9), dpi=300)
            # uncomment these lines if you want the inf distance marriage bar (at -1) to show
            #plt.hist(target_marriage_dist, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='target')
            #plt.hist(model_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='model')
            # uncomment these two lines if you want to only show finite-distance marriage distributions
            if not side_by_side:
                plt.hist(target_marriage_dist, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='target', density=marriage_density)
                plt.hist(model_marriage_distances, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='artificial distances model', density=marriage_density)
                plt.hist(paj_marriage_distances, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='actual distances model', density=marriage_density)
            else:
                plt.hist([target_marriage_dist, model_marriage_distances, paj_marriage_distances],
                         bins=[k for k in range(max_bin + 2)],
                         range=(-1, max_bin+2),
                         label=['target', 'artificial distances model', 'actual distances model'],
                         density=marriage_density)
            plt.legend()
            title = name + '\n'
            title += r"target: {}% $\infty$-distance marriages".format(str(np.round(target_percent_inf_marriages, 3)*100)[:4])
            title += '\n'
            title += r"artificial distances model: {}% $\infty$-distance marriages".format(str(np.round(model_percent_inf_marriages, 3)*100)[:4])
            title += '\n'
            title += r"actual distances model: {}% $\infty$-distance marriages".format(str(np.round(paj_percent_inf_marriages, 3)*100)[:4])
            title += '\n'

            title += fr'$JS(target\Vert artificial) =  {actual_marriage_KL_div:.3e}$'
            title += '\n'
            title += fr'$JS(target\Vert actual) =  {actual_marriage_KL_div:.3e}$'

            plt.title(title, fontsize=12, pad=2)
            if not save_plots:
                plt.show()
            else:
                plt.savefig(os.path.join(output_path,  name + '_finite_marriage_distributions' + '.png'), format='png')
                plt.savefig(os.path.join(marriage_hist_path,  name + '_finite_marriage_distributions' + '.png'), format='png')
            plt.clf()  # clear out the current figure

        # creates histogram of marriage distributions including infinite distance marriages
        model_max_bin = int(np.max(model_marriage_distances))
        paj_max_bin = int(np.max(paj_marriage_distances))
        target_max_bin = int(np.max(target_marriage_dist))
        max_bin = max(model_max_bin, paj_max_bin, target_max_bin)

        fig = plt.figure(figsize=(12,9), dpi=300)
        # uncomment these lines if you want the inf distance marriage bar (at -1) to show
        if not side_by_side:
            plt.hist(target_marriage_dist, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='target', density=marriage_density)
            plt.hist(model_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='artificial distances model', density=marriage_density)
            plt.hist(paj_marriage_distances, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='actual distances model', density=marriage_density)
        else:
            plt.hist([target_marriage_dist, model_marriage_distances, paj_marriage_distances],
                     bins=[k for k in range(-1, max_bin + 2)],
                     range=(-2, max_bin+2),
                     label=['target', 'artificial distances model', 'actual distances model'],
                     density=marriage_density)
        # uncomment these two lines if you want to only show finite-distance marriage distributions
        #plt.hist(target_marriage_dist, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='target', density=marriage_density)
        #plt.hist(model_marriage_distances, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='model', density=marriage_density)
        plt.legend()
        title = name + '\n'
        title += r"target: {}% $\infty$-distance marriages".format(str(np.round(target_percent_inf_marriages, 3)*100)[:4])
        title += '\n'
        title += r"artificial distances model: {}% $\infty$-distance marriages".format(str(np.round(model_percent_inf_marriages, 3)*100)[:4])
        title += '\n'
        title += r"actual distances model: {}% $\infty$-distance marriages".format(str(np.round(paj_percent_inf_marriages, 3)*100)[:4])
        title += '\n'

        title += fr'$JS(target\Vert artificial) =  {actual_marriage_KL_div:.3e}$'
        title += '\n'
        title += fr'$JS(target\Vert actual) =  {actual_marriage_KL_div:.3e}$'

        plt.title(title, pad=2)
        if not save_plots:
            plt.show()
        else:
            plt.savefig(os.path.join(output_path,  name + '_marriage_distributions' + '.png'), format='png')
            plt.savefig(os.path.join(marriage_hist_path,  name + '_marriage_distributions' + '.png'), format='png')
        plt.clf()  # clear out the current figure

        # creates histogram of children per couple distributions
        model_max_bin = int(np.max(model_children_per_couple))
        paj_max_bin = int(np.max(paj_children_per_couple))
        target_max_bin = int(np.max(target_child_dist))
        max_bin = max(model_max_bin, paj_max_bin, target_max_bin)

        fig = plt.figure(figsize=(12,9), dpi=300)
        # uncomment these lines if you want the inf distance marriage bar (at -1) to show
        #plt.hist(target_marriage_dist, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='target')
        #plt.hist(model_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='model')
        # uncomment these two lines if you want to only show finite-distance marriage distributions
        if not side_by_side:
            plt.hist(target_child_dist, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='target', density=child_density)
            plt.hist(model_children_per_couple, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='model', density=child_density)
            plt.hist(paj_children_per_couple, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='actual distances model', density=child_density)
        else:
            plt.hist([target_child_dist, model_children_per_couple, paj_children_per_couple],
                     bins=[k for k in range(max_bin + 2)],
                     range=(-1, max_bin+2),
                     label=['target', 'artificial distances model', 'actual distances model'],
                     density=child_density)
        plt.legend()
        title = name + '\n'
        title += r"target: {} avg. children per couple".format(str(np.round(np.mean(target_child_dist), 3))[:4])
        title += '\n'
        title += r"artificial distances model: {} avg. children per couple".format(str(np.round(np.mean(model_children_per_couple), 3))[:4])
        title += '\n'
        title += r"actual distances model: {}% avg. children per couple".format(str(np.round(np.mean(paj_children_per_couple), 3))[:4])
        title += '\n'

        title += fr'$JS(target\Vert artificial) =  {actual_child_KL_div:.3e}$'
        title += '\n'
        title += fr'$JS(target\Vert actual) =  {actual_child_KL_div:.3e}$'


        plt.title(title, pad=2)
        if not save_plots:
            plt.show()
        else:
            plt.savefig(os.path.join(output_path,  name + '_child_distributions' + '.png'), format='png')
            plt.savefig(os.path.join(child_hist_path,  name + '_child_distributions' + '.png'), format='png')
        plt.clf()  # clear out the current figure
#%%
# target=target_marriage_dist
# data=model_marriage_distances_by_run
# title='finite-distance unions'
# target=target_child_dist
# data = model_children_per_couple_by_run
# title = 'children per household'
# destination=
#%%
def make_box_and_whisker_plot(target, data, name, title, destination, save_plots=True, dpi=300, figsize=(12,9)):


    def get_proportions(single_run_data):
        single_run_data = np.array(single_run_data)
        return {x:np.sum(single_run_data == x)/len(single_run_data) for x in set(single_run_data)}


    # get proportions for each individual instantiation
    # IE a list of proportions of distance 2, of distance 3, etc.
    proportions = []
    for run in data:
        proportions.append(get_proportions(run))

    # now unpack the dictionaries
    all_keys = sorted(list(set.intersection(*map(set, proportions))))
    proportions = [[d[k] for d in proportions] for k in all_keys]

    target = np.array(target)
    # now set up the figure
    fig = plt.figure(dpi=dpi, figsize=figsize)
    plt.bar([k for k in set(target)], [np.sum(target == k)/len(target) for k in set(target)], label='target', alpha=0.85)
    plt.boxplot(proportions, positions=all_keys)
    plt.ylim(0, 1)
    plt.title(name + f" {title} distribution across {len(data)} instantiations")
    plt.legend()
    if not save_plots:
        plt.show()
    else:
        plt.savefig(os.path.join(destination, name + f'{title}_variance.png'), format='png')
    plt.close(fig)


#%%
def visualize_distribution_histograms_many_runs(use_latest=True,
                                                path_to_start_sizes_pkl=os.path.join('data_set_stats', 'review_of_start_sizes_1'),
                                                save_models=True,
                                                save_plots=True,
                                                marriage_density=True,
                                                child_density=True,
                                                side_by_side=True,
                                                max_tries=5,
                                                num_instantiations=1,
                                                names=None,
                                                plot_reduced=True):
    paths = find_files('data_set_stats', 'review_of_start_sizes')
    if use_latest:
        path_to_start_sizes_pkl = paths[-1]
        out_dir = paths[-1]
    else:
        out_dir = path_to_start_sizes_pkl.copy()

    if save_models or save_plots:
        out_dir = os.path.join(out_dir, 'model_specific_results')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if save_plots:
        marriage_hist_path = os.path.join(out_dir, 'marriage_distance_histograms')
        if not os.path.exists(marriage_hist_path):
            os.makedirs(marriage_hist_path)

        child_hist_path = os.path.join(out_dir, 'child_distance_histograms')
        if not os.path.exists(child_hist_path):
            os.makedirs(child_hist_path)

    path_to_start_sizes_pkl = os.path.join(path_to_start_sizes_pkl, 'data_set_start_sizes.pkl')
    # load the start size
    start_sizes = pd.read_pickle(path_to_start_sizes_pkl)
    start_sizes = start_sizes.loc[~start_sizes.avg_start_size.isna(), ['name', 'iters', 'avg_start_size']]
    start_sizes.loc[:, 'avg_start_size'] = start_sizes.loc[:, 'avg_start_size'].astype(int)
    if names is None:
        names = start_sizes.loc[:, 'name'].unique()

    for name in names:
        num_people = start_sizes.loc[start_sizes.name == name, 'avg_start_size'].values[0]
        target_marriage_dist, target_num_marriages, target_prob_inf_marriage, target_prob_finite_marriage, target_child_dist, target_size_goal = get_graph_stats(name)
        model_marriage_distances_by_run = []
        model_children_per_couple_by_run = []
        reduced_marriage_distances_by_run = []
        reduced_children_per_couple_by_run = []
        for k in range(num_instantiations):
            model_dies_out = True
            tries = 0
            model_marriage_distances = []
            model_children_per_couple = []
            reduced_marriage_distances = []
            reduced_children_per_couple = []

            while model_dies_out:
                # try several times to build a model which survives (recaptures the target_size_goal number of nodes)
                this_G, this_G_reduced, this_model_marriage_edges, this_model_marriage_distances, this_model_children_per_couple, model_dies_out, this_output_path = human_family_network(num_people,
                                                                                                                                                                                          target_marriage_dist,
                                                                                                                                                 target_prob_finite_marriage,
                                                                                                                                                 target_prob_inf_marriage,
                                                                                                                                                 target_child_dist,
                                                                                                                                                 name,
                                                                                                                                                 when_to_stop=target_size_goal,
                                                                                                                                                 out_dir=out_dir,
                                                                                                                                                 save=save_models)
                tries += 1
                if tries > max_tries:
                    print(name + ' died out five times with start size {}'.format(num_people))
                    break
            if model_dies_out:
                break

            model_marriage_distances_by_run.append(this_model_marriage_distances)
            model_marriage_distances += this_model_marriage_distances
            model_children_per_couple_by_run.append(this_model_children_per_couple)
            model_children_per_couple += this_model_children_per_couple

            # union edges are the same in reduced, non reduced graph
            # children per union are the same in reduced, non reduced graph

            reduced_biological = this_G_reduced.edge_subgraph([(u,v) for u, v, d in this_G.edges(data=True) if d['Relationship'] != 'Marriage']).copy()
            # may drop immigrant nodes who had no children,  drop their edges
            # from the union list
            reduced_biological_unions = [edge for edge in this_model_marriage_edges if (edge[0] in reduced_biological.nodes() and edge[1] in reduced_biological.nodes())]
            reduced_marriage_dist, reduced_num_inf_marriages, reduced_percent_inf_marriages = get_marriage_distances_kolton(reduced_biological, reduced_biological_unions, plot=False, save=False)
            reduced_marriage_distances_by_run.append(reduced_marriage_dist)
            reduced_marriage_distances += reduced_marriage_dist
            reduced_children_per_couple_by_run.append(this_model_children_per_couple)  # intentional.  Reduced and full models have same union edges, so same children per union edge.
            reduced_children_per_couple += this_model_children_per_couple  # ditto
        if model_dies_out:
            continue
        actual_marriage_KL_div = KL_Divergence(model_marriage_distances, target_marriage_dist)
        actual_child_KL_div = KL_Divergence(model_children_per_couple, target_child_dist)
        reduced_marriage_KL_div = KL_Divergence(reduced_marriage_distances, target_marriage_dist)
        reduced_child_KL_div = actual_child_KL_div  # intentional, reduced and full model have the same set of union edges, so have the same children per union

        model_marriage_distances = np.array(model_marriage_distances)
        target_marriage_dist = np.array(target_marriage_dist)
        reduced_marriage_distances = np.array(reduced_marriage_distances)

        model_percent_inf_marriages = sum(model_marriage_distances == -1) / len(model_marriage_distances)
        target_percent_inf_marriages = sum(target_marriage_dist == -1) / len(target_marriage_dist)
        reduced_percent_inf_marriages = sum(reduced_marriage_distances == -1) / len(reduced_marriage_distances)

        if len(model_marriage_distances[model_marriage_distances != -1]) != 0:
            # creates histogram of marriage distributions excluding infinite distance unions
            model_max_bin = int(np.max(model_marriage_distances[model_marriage_distances != -1]))
            target_max_bin = int(np.max(target_marriage_dist[target_marriage_dist != -1]))
            # reduced_max_bin = int(np.max(reduced_marriage_distances[reduced_marriage_distances != -1]))
            max_bin = max(model_max_bin, target_max_bin,) # reduced_max_bin)

            fig = plt.figure(figsize=(12,9), dpi=300)
            # uncomment these lines if you want the inf distance marriage bar (at -1) to show
            #plt.hist(target_marriage_dist, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='target')
            #plt.hist(model_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='model')
            # uncomment these two lines if you want to only show finite-distance marriage distributions
            if not side_by_side:
                plt.hist(target_marriage_dist, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='target', density=marriage_density)
                plt.hist(model_marriage_distances, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='actual distances model', density=marriage_density)
                plt.hist(reduced_marriage_distances, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='artificial distances model', density=marriage_density)
            else:
                plt.hist([target_marriage_dist, model_marriage_distances, reduced_marriage_distances],
                         bins=[k for k in range(max_bin + 2)],
                         range=(-1, max_bin+2),
                         label=['target', 'actual distances model', 'artificial distances model'],
                         density=marriage_density)
            plt.legend()
            title = name + '\n'

            plt.title(title, fontsize=12, pad=2)

            # add a tabel at the bottom of the axes
            columns = ['target', 'actual distances model', 'artificial distances model']
            rows = ['KL div. distance to union', r'% $\infty$-distance unions']

            the_table = plt.table(cellText=[['-', f'{actual_marriage_KL_div:.3e}', f'{reduced_marriage_KL_div:.3e}'],
                                            #['-', f'{actual_child_KL_div:.3e}', f'{reduced_child_KL_div:.3e}'],
                                            [str(np.round(target_percent_inf_marriages, 3)*100)[:4], str(np.round(model_percent_inf_marriages, 3)*100)[:4], str(np.round(reduced_percent_inf_marriages, 3)*100)[:4]]],
                                  colLabels=columns,
                                  rowLabels=rows,
                                  loc='bottom',
                                  bbox=[0.19, -.15, .81, 0.1])

            plt.subplots_adjust(left=0.2, bottom=0.02)

            if not save_plots:
                plt.show()
            else:
                plt.savefig(os.path.join(marriage_hist_path,  name + '_finite_marriage_distributions' + '.png'), format='png', bbox_inches='tight')
            plt.clf()  # clear out the current figure
            plt.close(fig)

            #make_box_and_whisker_plot(target_marriage_dist, model_marriage_distances, name, title='finite-distance unions', destination=marriage_hist_path)

        # creates histogram of marriage distributions including infinite distance marriages
        model_max_bin = int(np.max(model_marriage_distances))
        target_max_bin = int(np.max(target_marriage_dist))
        # reduced_max_bin = int(np.max(reduced_marriage_distances))
        max_bin = max(model_max_bin, target_max_bin,) # reduced_max_bin)

        fig = plt.figure(figsize=(12,9), dpi=300)
        # uncomment these lines if you want the inf distance marriage bar (at -1) to show
        if not side_by_side:
            plt.hist(target_marriage_dist, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='target', density=marriage_density)
            plt.hist(model_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='actual distances model', density=marriage_density)
            plt.hist(reduced_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='artificial distances model', density=marriage_density)
        else:
            plt.hist([target_marriage_dist, model_marriage_distances, reduced_marriage_distances],
                     bins=[k for k in range(-1, max_bin + 2)],
                     range=(-2, max_bin+2),
                     label=['target', 'actual distances model', 'artificial distances model'],
                     density=marriage_density)
        # uncomment these two lines if you want to only show finite-distance marriage distributions
        #plt.hist(target_marriage_dist, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='target', density=marriage_density)
        #plt.hist(model_marriage_distances, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='model', density=marriage_density)
        plt.legend()
        title = name + '\n'

        plt.title(title, pad=2)

        # add a tabel at the bottom of the axes
        columns = ['target', 'actual distances model', 'artificial distances model']
        rows = ['KL div. distance to union', r'% $\infty$-distance unions']

        the_table = plt.table(cellText=[['-', f'{actual_marriage_KL_div:.3e}', f'{reduced_marriage_KL_div:.3e}'],
                                        #['-', f'{actual_child_KL_div:.3e}', f'{reduced_child_KL_div:.3e}'],
                                        [str(np.round(target_percent_inf_marriages, 3)*100)[:4], str(np.round(model_percent_inf_marriages, 3)*100)[:4], str(np.round(reduced_percent_inf_marriages, 3)*100)[:4]]],
                                colLabels=columns,
                                rowLabels=rows,
                                loc='bottom',
                                bbox=[0.19, -.15, .81, 0.1])

        plt.subplots_adjust(left=0.2, bottom=0.02)

        if not save_plots:
            plt.show()
        else:
            plt.savefig(os.path.join(marriage_hist_path,  name + '_marriage_distributions' + '.png'), format='png', bbox_inches='tight')
        plt.clf()  # clear out the current figure
        plt.close(fig)
        make_box_and_whisker_plot(target_marriage_dist, model_marriage_distances_by_run, name, title='unions', destination=marriage_hist_path)
        make_box_and_whisker_plot(target_marriage_dist, reduced_marriage_distances_by_run, name, title='reduced model unions', destination=marriage_hist_path)


        # creates histogram of children per couple distributions
        model_max_bin = int(np.max(model_children_per_couple))
        target_max_bin = int(np.max(target_child_dist))
        # reduced_max_bin = int(np.max(reduced_children_per_couple))
        max_bin = max(model_max_bin, target_max_bin,) # reduced_max_bin)

        fig = plt.figure(figsize=(12,9), dpi=300)
        # uncomment these lines if you want the inf distance marriage bar (at -1) to show
        #plt.hist(target_marriage_dist, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='target')
        #plt.hist(model_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='model')
        # uncomment these two lines if you want to only show finite-distance marriage distributions
        if not side_by_side:
            plt.hist(target_child_dist, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='target', density=child_density)
            plt.hist(model_children_per_couple, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='actual distances model', density=child_density)
            plt.hist(reduced_children_per_couple, bins=[k for k in range(max_bin + 2)], range=(-1, max_bin+2), alpha=0.65, label='artificial distances model', density=child_density)
        else:
            plt.hist([target_child_dist, model_children_per_couple, reduced_children_per_couple],
                     bins=[k for k in range(max_bin + 2)],
                     range=(-1, max_bin+2),
                     label=['target', 'actual distances model', 'artificial distances model'],
                     density=child_density)
        plt.legend()
        title = name + '\n'


        plt.title(title, pad=2)

        # add a tabel at the bottom of the axes
        columns = ['target', 'actual distances model', 'artificial distances model']
        rows = ['KL div. children per union']

        the_table = plt.table(cellText=[#['-', f'{actual_marriage_KL_div:.3e}', f'{reduced_marriage_KL_div:.3e}'],
                                        ['-', f'{actual_child_KL_div:.3e}', f'{reduced_child_KL_div:.3e}']],
                                        #[str(np.round(target_percent_inf_marriages, 3)*100)[:4], str(np.round(model_percent_inf_marriages, 3)*100)[:4], str(np.round(reduced_percent_inf_marriages, 3)*100)[:4]]],
                                colLabels=columns,
                                rowLabels=rows,
                                loc='bottom',
                                bbox=[0.19, -.15, .81, 0.1])

        plt.subplots_adjust(left=0.2, bottom=0.02)
        if not save_plots:
            plt.show()
        else:
            plt.savefig(os.path.join(child_hist_path,  name + '_child_distributions' + '.png'), format='png', bbox_inches='tight')
        plt.clf()  # clear out the current figure
        plt.close(fig)
        make_box_and_whisker_plot(target_child_dist, model_children_per_couple_by_run, name, title='children per household', destination=child_hist_path)
#%%


if __name__=="__main__":
    evaluate_start_size_searches()
    visualize_distribution_histograms_many_runs(num_instantiations=10, names=None)
