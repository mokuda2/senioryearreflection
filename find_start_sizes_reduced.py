# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:12:00 2023
edited 5/8/2023 at 8:50pm
@author: kolto

runs the bisection search for start size (see Section 3.4) for the Variant
Target Model (Chapter 5).

a list of desired target names (each listed on a new line) should be saved in a
text file at the relative path ./dataset_list/datasets-that-work.txt
"""

from family_model_intergenerational_marriage_reduced import *
from KL_divergence import *
from get_model_parameters import *
import argparse
from distutils import util
import pandas as pd
import os

get_bool = lambda x: bool(util.strtobool(x))  # helper function to parse boolean values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-network_num', '--network_num', type=int, default=0, help="(int) corresponding to the network's index position in returned from get_graphs_and_names()")
    parser.add_argument('-out_directory', '--out_directory', type=str, default='start_size_reduced')
    parser.add_argument('-iters', '--iters', type=int, default=50, help='(int).  number of times to call find_start_size() (IE number of bisessections searchs to hold)')
    parser.add_argument('-max_iters', '--max_iters', type=int, default=100, help='(int).  maximum number of models to create in attempt to find start size that wont die out more than dies_out_threshold times')
    parser.add_argument('-dies_out_threshold', '--dies_out_threshold', type=int, default=5, help='(int. number of times (as a fraction of max_iters) beyond which the model dies out too frequently, below which the model does not die out frequently enough')
    parser.add_argument('-verbose', '--verbose', type=get_bool, default=False, help='(bool).  indicates whether step-by-step printouts of greatest_lower_bound and least_upper_bound should be printed out at each iteration.')
    parser.add_argument('-save_start_sizes', '--save_start_sizes', type=get_bool, default=True, help='(bool), indicates whether each iterations bisection search should be saved (in a single txt file, pickle file) for later use')
    parser.add_argument('-save_individual_start_sizes', '--save_individual_start_sizes', type=get_bool, default=False, help='(bool), indicates whether each bisection searchs list of start sizes should be saved in its own file')
    parser.add_argument('-random_start', '--random_start', type=get_bool, default=True, help='(bool). indicates whether a random initial starting popultation should be used (drawn uniformly between [2, num_people_in_target_network].  If False, the midpoint is used.')
    parser.add_argument('-load_all_networks', '--load_all_networks', type=get_bool, default=True, help='(bool).  indicates whether every available network (saved in ./Original_Sources) should be used.  Default is True.  If False, then a list of which network names to run should be saved to ./instructions')

    # unpack user-specified arguments (set in bash script)
    args = parser.parse_args()

    network_num = args.network_num
    out_directory = args.out_directory
    iters = args.iters
    max_iters = args.max_iters
    dies_out_threshold = args.dies_out_threshold
    verbose = args.verbose
    save_start_sizes = args.save_start_sizes
    save_individual_start_sizes = args.save_individual_start_sizes
    random_start = args.random_start
    load_all_networks = args.load_all_networks

    if load_all_networks:
        # load the list of graph names
        target_names = get_names(save_paths=False)
        target_names.remove('warao')  # you'll want to drop the network that doesnt have a ususal .paj file
    else:
        #target_names = pd.read_pickle(os.path.join('./dataset_list', 'partially_completed_networks.pkl'))
        with open(os.path.join('./dataset_list', 'datasets-that-work.txt'), 'r') as infile:
            target_names = infile.readlines()
        target_names = [k.strip() for k in target_names]


    # access the user-specified graph name
    name = target_names[network_num]
    print('name:', name)
    # now conduct the bisection search (this also saves a png of the search visualization)
    avg_start_size = repeatedly_call_start_size(name,
                                                out_directory=out_directory,
                                                iters=iters,
                                                max_iters=max_iters,
                                                dies_out_threshold=dies_out_threshold,
                                                verbose=verbose,
                                                save_start_sizes=save_start_sizes,
                                                save_individual_start_sizes=save_individual_start_sizes,
                                                random_start=random_start)


    #  Maybe you can overwrite a dictionary: {name: avg_start_size}?  but if you do that you'll need to
    # decide about how to prevent the super computer from editing that file in a sequence that misses changes
    # if not os.path.exists(os.path.join(out_directory, 'master_start_size_dict')):
