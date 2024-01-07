"""
This version of the Variant Target Model results in a graph with (potentially)
many components (returned as G by human_family_network()) and a version with a
single weakly connected component (returned as G_connected by
human_family_network()) (see Chapter 5 in Kolton's thesis))

The main guts of the model are in the human_family_network() method.  There is
an example of how to run the model commented out below human_family_network()
"""
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ast
import itertools
import pickle
import os
import regex as re
from write_model_to_pajek_ore_graph import format_as_pajek
from itertools import combinations, product
import line_profiler
import cython
import marriage_code
import graph_distributions
from line_profiler import profile
import memory_profiler
# from memory_profiler import profile
from pprint import pprint
import unittest
import sys

#%%
def makeOutputDirectory(out_directory, name):
    """
    Make an output directory to keep things cleaner

    Returns a full output path to the new directory
    """
    ver = 1
    output_dir = os.path.join(out_directory, name + '_')
    while os.path.exists(output_dir+str(ver)):
        ver += 1
    output_dir += str(ver)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


#%%
def find_file_version_number(out_directory, filename, extension):
    ver = 1
    output_dir = os.path.join(out_directory, filename + '_')
    while os.path.exists(output_dir+str(ver)+extension):
        ver += 1
    filename = filename + '_' + str(ver)
    return filename



#%%
def get_graph_path(name, path='./Original_Sources/'):
    """
    PARAMETERS:
        name: (str) the name of the kinsources data set (see below for format)
    RETURNS:
        path: (str) path to directory prepended to the full name of specified
              kinsources file
    """
    return path + 'kinsources-'+name+'-oregraph.paj'


#%%
def get_num_people(name):
    """
    gets the number of people (total number of verticies) in the .paj file
    This function assumes that the number of nodes is correctly reported in the
    3rd line (index 2) of the .paj file. For example, contents should begin in
    the following format (defined in the funtion below):
        ['*Network Ore graph Tikopia.puc\n',
         '\n',
         '*vertices 294\n',
         "1 'X (1)' ellipse\n",
         ...]

    PARAMETERS:
        name: (str) the name of the kinsources data set
    RETURNS:
        num_people: (int) total number of people in the graph
    """
    path_to_graph = get_graph_path(name)
    # open and read graph file
    with open(path_to_graph, 'r') as file:
        contents = file.readlines()

    num_people = contents[2]
    num_people_pattern = re.compile("[0-9]+")
    num_people = int(num_people_pattern.findall(num_people)[0])

    return num_people


#%%
def get_graph_stats(name, distance_path='./Kolton_distances/', child_number_path='./ChildrenNumber/'):
    """
    Gets the statistics of a specified kinsources dataset
    PARAMETERS:
        name: (str) the name of the kinsources data set
        distance_path: (str) the filepath to the directory containing the saved
            text files containing the distance to marriage distributions (the output
            of timing_kolton_distance_algorithm.py)
        child_number_path: (str) the filepath to the directory containing the
            saved text files containing the children per couple distributions
    RETURNS:
        marriage_dists: (list of int) one entry per marriage indicating how many
            generations between spouses (reported in the number of parent-child
            edges crossed so that distance between siblings is 2) in the
            specified dataset.  If no common ancestor (IE an infinite distance)
            then the corresponding entry is -1
        num_marriages: (int) total number of marriage edges in the specified
            dataset
        prob_inf_marriage: (float) number of infinite marraiges divided by total
            number of marriages in the specified dataset
        prob_finite_marriage: (float) number of non-infinite marriages divided
            by total number of marriages in the specified dataset
        children_dist: (list of int) one entry per pair of parents, indicating
            how many child edges each parent in the couple share
        num_people: (int) total number of nodes in the graph named.
    """
    with open(distance_path + '{}.txt'.format(name)) as infile:
        marriage_dists, num_inf_marriages, fraction_inf_marriage = [ast.literal_eval(k.strip()) for k in infile.readlines()]

    # number of children data of chosen network
    with open(child_number_path + '{}_children.txt'.format(name)) as f:
        nx_child = f.readline()
    children_dist = ast.literal_eval(nx_child)

    num_marriages = len(marriage_dists)
    num_people = get_num_people(name)
    prob_marriage = num_marriages * 2 / num_people  # *2 since 2 spouses per marriage
    prob_inf_marriage = prob_marriage * fraction_inf_marriage
    prob_finite_marriage = prob_marriage - prob_inf_marriage

    return marriage_dists, num_marriages, prob_inf_marriage, prob_finite_marriage, children_dist, num_people

#%%

def get_child_probabilities(data):

    """
    NO KDE
    given a list (either of distances to marriage for a marriage distribution or
    numbers of children for a children distribution), get_probabilities() produces
    a dictionary of probabilities. NOTE: the resulting "probabilities" should
    not be expected to sum to 1.  If a true probability distribution is desired
    then you should normalize the resulting distribution.  The resulting
    dictionary has entries beyond the data supplied (for example if a supplied
    marriage distribution has a maximum distance of 14 the resulting dictionary
    has entries for distances greater than 14 to allow us to use the
    datastructure without key errors should a larger number be drawn; we add
    1000 entries beyond the maximum.  If ever more than 1000 generations are to
    be run in the model, then this function should be modified).
    PARAMETERS:
        data (list): data taken from an actual family network
        bandwidth (int):  used as an argument in to_KDE(), the std deviation of
            each kernel in the sum (see documentation)
    RETURNS:
        probs (dictionary): keys are the entries of data and successive values,
            too (we lengthen the right tail of the distribution).
    """

    data = np.array(data)


    data = data[data > -1]  # only use non-negative number of children per household
    domain = np.arange(0, max(data) + 1, 1)

    probs = {x: sum(data==x)/len(data) for x in domain}
    # return only those keys with non zero values
    probs = {key:val for key, val in probs.items() if val != 0}
    return probs


##%
# data = marriage_dist

def get_marriage_probabilities(data, num_people, eps=1e-7):

    """
    NO KDE
    given a list (either of distances to marriage for a marriage distribution or
    numbers of children for a children distribution), get_probabilities() produces
    a dictionary of probabilities. NOTE: the resulting "probabilities" should
    not be expected to sum to 1.  If a true probability distribution is desired
    then you should normalize the resulting distribution.  The resulting
    dictionary has entries beyond the data supplied (for example if a supplied
    marriage distribution has a maximum distance of 14 the resulting dictionary
    has entries for distances greater than 14 to allow us to use the
    datastructure without key errors should a larger number be drawn; we add
    1000 entries beyond the maximum.  If ever more than 1000 generations are to
    be run in the model, then this function should be modified).
    PARAMETERS:
        data (list): data taken from an actual family network
        bandwidth (int):  used as an argument in to_KDE(), the std deviation of
            each kernel in the sum (see documentation)
    RETURNS:
        probs (dictionary): keys are the entries of data and successive values,
            too (we lengthen the right tail of the distribution).
    """
    data = np.array(data)

    frac_inf_marriage = sum(data ==  -1) / len(data)  # as a proportion of marriage edges
    prob_inf_marriage = len(data) * 2 / num_people * frac_inf_marriage
    # ??? I feel like marriage distances should always have all distances
    #     possible, even the gross ones (IE we need to count from 2 not the
    #     minimum distance seen in the dataset)
    # domain = np.arange(2, max(data)+1000, 1)
    # domain = np.arange(-1, max(data)+1000, 1)  # ??? shouldn't I go from 0 to inf or from 2 to inf all the time?
    # domain = np.arange(-1, max(data), 1)
    # probs = {x: sum(data==x)/len(data) * prob_marry + eps for x in domain}


    domain = np.arange(2, max(data) + 1, 1)
    # frac_finite_marriage = sum(data != -1) / len(data)
    finite_probs = {x: len(data) * 2 / num_people * sum(data==x)/len(data) for x in domain}
    prob_finite_marriage = sum(finite_probs.values())  # normalize finite probs, then scale to sum to prob_finite_marriage
    # note that we shift the entire distribution up by epsilon, added uniformly everywhere
    # finite_probs = {key: val / denom * prob_finite_marriage + eps for key, val in finite_probs.items()}
    finite_probs = {key: val + eps for key, val in finite_probs.items()}

    prob_single = 1 - prob_inf_marriage - prob_finite_marriage

    probs = {}
    probs[-1] = prob_inf_marriage + eps
    probs[0] = prob_single + eps

    probs = probs | finite_probs

    # return only those keys with non zero values
    probs = {key:val for key, val in probs.items() if val != 0}
    # prevents divide by zero errors when defining dis_probs in add_marriage_edges()
    # eps = min([k for k in probs.values() if k != 0]) / 2
    # probs = {key:val + eps for key, val in probs.items()}

    return probs
#%%
# @profile
def get_difference_in_probabilities(target_probs, current, num_people, name, gen_num, outpath, eps=1e-7, plot=False):
    """
    This method accepts both the target marriage distribution AND the
    current-state model marriage distribution.  It will subtract the
    current-state from the target probabilities, flooring at some positive
    epsilon.  The returned probability distribution will then favor any
    marriages of those distances which have not yet been drawn in proportion
    with the target distribution's frequency for that distance.

    PARAMETERS:
        target_probs (dictionary): keys are marriage distances, values are
            probabilities.  This is the result of
            get_probabilities(marriage_dist) (called finite_marriage_probs
            below). This should already be normalized.
        current (list): list of marriage distances currently represented in the
            graph.
    """
    current_probs = get_marriage_probabilities(current, num_people, eps=eps)
    current_probs = {key:value/sum(current_probs.values()) for key, value in zip(current_probs.keys(), current_probs.values())} # normalize
    # need every key that occurs in target to also occur in current_probs
    current_probs = current_probs | {key : 0 for key in target_probs if key not in current_probs}

    adjusted_probs = {key: target_probs[key] - current_probs[key] if target_probs[key] - current_probs[key] >= 0  else eps for key in target_probs.keys()}

    # normalize
    adjusted_probs = {key:value/sum(adjusted_probs.values()) for key, value in zip(adjusted_probs.keys(), adjusted_probs.values()) if value != 0} # normalize

    adjusted_probs = adjusted_probs | {key:0 for key in target_probs if key not in adjusted_probs}

    if plot:
        graph_current_distributions(target_probs, current_probs, adjusted_probs, gen_num, name, outpath, eps=eps)
    return adjusted_probs

# target_marriage_probs = target_probs
# model_marriage_probs = current_probs
# adjusted_marriage_probs = adjusted_probs
# gen_num = i

#%%
# @profile
def graph_current_distributions(target_marriage_probs, model_marriage_probs, adjusted_marriage_probs, gen_num, name, outpath, save_plots=True, alpha=0.85, eps=1e-7):
    # find max bin (with non-zero occurance)
    model_vals = np.array(list(model_marriage_probs.values()))
    target_vals = np.array(list(target_marriage_probs.values()))
    adjusted_vals = np.array(list(adjusted_marriage_probs.values()))

    target_vals = target_vals[target_vals != 0]
    model_vals = model_vals[model_vals != 0]
    adjusted_vals = adjusted_vals[adjusted_vals != 0]

    # eps = max(min(target_vals), min(model_vals), min(adjusted_vals))
    target_eps = np.min(target_vals)
    model_eps = np.min(model_vals)
    adjusted_eps = np.min(adjusted_vals)

    if eps != 0:
        target_distances = [k for k in target_marriage_probs.keys() if target_marriage_probs[k] > target_eps]
        model_distances = [k for k in model_marriage_probs.keys() if model_marriage_probs[k] > model_eps ]
        adjusted_distances = [k for k in adjusted_marriage_probs.keys() if adjusted_marriage_probs[k] > adjusted_eps]
    else:
        target_distances = [k for k in target_marriage_probs.keys() if target_marriage_probs[k] >= target_eps]
        model_distances = [k for k in model_marriage_probs.keys() if model_marriage_probs[k] >= model_eps ]
        adjusted_distances = [k for k in adjusted_marriage_probs.keys() if adjusted_marriage_probs[k] >= adjusted_eps]


    max_bin = int(max(model_distances + target_distances + adjusted_distances))

    fig = plt.figure(figsize=(12,9), dpi=300)
    # uncomment these lines if you want the inf distance marriage bar (at -1) to show
    #plt.hist(target_marriage_dist, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='target')
    #plt.hist(model_marriage_distances, bins=[k for k in range(-1, max_bin + 2)], range=(-2, max_bin+2), alpha=0.65, label='model')
    # uncomment these two lines if you want to only show finite-distance marriage distributions
    width = 0.3
    plt.bar([k for k in target_marriage_probs.keys() if k <= max_bin], [target_marriage_probs[k] for k in target_marriage_probs.keys() if k <= max_bin], alpha=alpha, label='target', width=width, align='edge')
    plt.bar([k + width for k in model_marriage_probs.keys() if k <= max_bin], [model_marriage_probs[k] for k in model_marriage_probs.keys() if k <= max_bin], alpha=alpha, label='model', width=width, align='edge')
    plt.bar([k + 2 * width for k in adjusted_marriage_probs.keys() if k <= max_bin], [adjusted_marriage_probs[k] for k in adjusted_marriage_probs.keys() if k <= max_bin], alpha=alpha, label='adjusted', width=width, align='edge')

    plt.legend()
    title = name + '\n'
    title += f'generation: {gen_num} \n'

    plt.title(title, fontsize=12, pad=2)
    if not save_plots:
        plt.show()
    else:
        plt.savefig(os.path.join(outpath,  name + f'distributions_generation_{gen_num}' + '.png'), format='png')
    plt.clf()  # clear out the current figure
    plt.close(fig)

#%%
@profile
def add_marriage_edges(people, prev_people, num_people, marriage_probs, prob_marry_immigrant, prob_marry, distance_matrix, indices, original_marriage_dist, tol=0, eps=1e-7):
    """
    Forms both infinite and finite distance marriages in the current generation

    INVERTED: FORMS FINITE MARRIAGES FIRST
    PARAMETERS:
        people: (list) of the current generation (IE those people eligible for
                marriage)
        prev_people: (list) of those in the previous generation who are yet unmarried.
        num_people: (int) the number of nodes/people currently in the graph
        finite_marriage_probs: (dictionary) keys are marriage distances, values
            are probabilities.  Note that this dictionary should only include
            entries for NON-infinite marriage distances, should have
            non-negative values which sum to 1, and should have a long right
            tail (IE lots of entries which map high (beyond what occurs in the
            example dataset) distances to zero(ish) probabilities)
        prob_marry_immigrant: (float) the probability that a given node will marry
                an immigrant (herein a person from outside the genealogical network,
                without common ancestor and therefore at distance infinity from the
                nodes in the list 'people') (formerly 'ncp')
        prob_marry: (float) the probability that a given node will marry another
                node in people
        distance_matrix: ((len(people) x len(people)) numpy array) indexed array of distance
            between nodes in people (siblings are distance 2)
        indices: (dictionary) maps node name (int) to index number in D (int)
    RETURNS:
        unions: (list of tuples of int) marriages formed.  Entries are of two
            types: 1) infinite distance marriages: one spouse is selected
            uniformly at random from the community (people) while the other is
            an immigrant to the community (IE a new node NOT listed in people).
            2) finite distance couples: both spouses are members of the
            community (IE listed in people).  These couples are selected at
            random according to the marriage_probs)
        num_immigrants: (int) the number of NEW people added to the graph.  As
            implemented ALL new people get married to someone in the current
            generation (herein people)
        marriage_distances: (list of int) one entry per marriage created during
            this function call (IE within this generation).  Each entry
            indicates the distance between spouses through their nearest common
            ancestor.  As before, a distance of -1 indicates an infinite
            distance (IE one spouse immigrated into the community)
        wont_marry: (list) of nodes in people who did not marry, will attempt
            to marry someone from the next generation
    """
    print('in add_marriage_edges()')
    finite_marriage_probs = {key: val for key, val in marriage_probs.items() if key > 0}
    if sum(finite_marriage_probs.values()) > 0:
        finite_marriage_probs = {key: val / sum(finite_marriage_probs.values()) for key, val in
                                 finite_marriage_probs.items()}

    desired_finite_distances = [distance for distance, prob in finite_marriage_probs.items() if prob > 0]

    minimum_permissible_distance = min(k for k in original_marriage_dist if k > -1)

    marriage_distances = []
    unions = set()

    people_set = set(people)
    next_person = num_people + 1
    num_inf_couples_to_marry = round(prob_marry_immigrant * len(people) / 2)
    num_finite_couples_to_marry = round(prob_marry * len(people) / 2)

    will_marry = set(np.random.choice(list(people_set), size=len(people_set) // 2, replace=False))
    wont_marry_until_next_time = [node for node in people_set if node not in will_marry]

    people_set |= will_marry | set(prev_people)

    possible_couples = {(man, woman) for man, woman in itertools.combinations(people_set, 2)} - {(man, woman) for
                                                                                                 man, woman in
                                                                                                 itertools.combinations(
                                                                                                     prev_people, 2)}

    possible_finite_couples = {(man, woman): distance_matrix[indices[man]][indices[woman]] for man, woman in possible_couples if
                               distance_matrix[indices[man]][indices[woman]] >= minimum_permissible_distance}
    # TODO: bottleneck?
    preferred_couples = {couple: distance for couple, distance in possible_finite_couples.items() if
                         distance in set(original_marriage_dist) and distance in desired_finite_distances}
    other_couples = {couple: distance for couple, distance in possible_finite_couples.items() if
                     couple not in preferred_couples}

    iter = 0

    while possible_finite_couples and iter < num_finite_couples_to_marry:
        if preferred_couples:
            possible_finite_couples_array = np.array(list(preferred_couples.keys()))

            dis_probs = np.array([finite_marriage_probs.get(d, eps) for d in preferred_couples.values()]).astype(float)
            dis_probs[np.abs(dis_probs) < tol] = 0  # prevent "negative zeros"
            dis_probs /= np.sum(dis_probs)  # normalize

            try:
                couple_index = np.random.choice(len(preferred_couples), p=dis_probs)

            except Exception as e:
                print(f'Error: {e}')
                sys.exit("Exiting program due to exception")

            couple = possible_finite_couples_array[couple_index]
        else:
            possible_finite_couples_array = np.array(list(other_couples.keys()))

            least_bad_distance = min(d for d in other_couples.values() if d >= minimum_permissible_distance)
            dis_probs = np.array([1 if d == least_bad_distance else 0 for d in other_couples.values()]).astype(float)
            dis_probs[np.abs(dis_probs) < tol] = 0  # prevent "negative zeros"
            dis_probs /= np.sum(dis_probs)  # normalize

            couple_index = np.random.choice(len(other_couples), p=dis_probs)
            couple = possible_finite_couples_array[couple_index]

        unions.add(tuple(couple))
        marriage_distances.append(int(distance_matrix[indices[couple[0]], indices[couple[1]]]))

        possible_finite_couples = {pair: distance for pair, distance in possible_finite_couples.items() if
                                   couple[0] not in pair and couple[1] not in pair}
        preferred_couples = {couple: distance for couple, distance in possible_finite_couples.items() if
                             distance in set(original_marriage_dist) and distance in desired_finite_distances}
        other_couples = {couple: distance for couple, distance in possible_finite_couples.items() if
                         couple not in preferred_couples}

        iter += 1

    if iter == 0:
        stay_single_forever = will_marry | set(prev_people)
    else:
        stay_single_forever = set(man for couple in possible_couples for man in couple)

    possible_inf_couples = {(man, woman): distance_matrix[indices[man]][indices[woman]]
                            for man, woman in itertools.combinations(stay_single_forever, 2)
                            if distance_matrix[indices[man]][indices[woman]] == -1}

    iter = 0
    while possible_inf_couples and iter < num_inf_couples_to_marry:
        possible_inf_couples_array = np.array(list(possible_inf_couples.keys()))

        couple_index = np.random.choice(len(possible_inf_couples))  # draw uniformly
        couple = possible_inf_couples_array[couple_index]

        unions.add(tuple(couple))
        marriage_distances.append(int(distance_matrix[indices[couple[0]]][indices[couple[1]]]))
        # print("marriage_distances:", marriage_distances)

        possible_inf_couples = {pair: distance for pair, distance in possible_inf_couples.items()
                                if couple[0] not in pair and couple[1] not in pair}

        iter += 1
        stay_single_forever = set(man for couple in possible_couples for man in couple)

    num_immigrants = num_inf_couples_to_marry - iter
    num_immigrants = min(len(stay_single_forever), num_immigrants)

    immigrants = list(range(next_person, next_person + num_immigrants))
    marry_strangers = np.random.choice(list(stay_single_forever), size=num_immigrants, replace=False)
    stay_single_forever -= set(marry_strangers)

    unions |= {(spouse, immigrant) for spouse, immigrant in zip(marry_strangers, immigrants)}

    marriage_distances.extend([-1] * num_immigrants)

    return unions, num_immigrants, marriage_distances, immigrants, wont_marry_until_next_time, len(stay_single_forever)

def add_children_edges(unions, num_people, child_probs):
    """
    PARAMETERS:
        unions: (list of tuple of int) marriages in the current generation
            (output of add_marriage_edges())
        num_people: (int) current total number of nodes (persons) in the graph
            (IE the sum of the size of every generation)
        child_probs: (dictionary) keys are number of children (int), values are
            the probability (float) that a couple has key children (the output
            of get_probabilities(child_dist))
        indices: (dictionary) mapping the current generations' names (int) to
            index (row/column number) in the current distance matrix D
    RETURNS:
        child_edges: (list of tuple of int) entries are (parent, child) and
            should be added to the graph
        families: (list of list of int) entries are lists of children pertaining
            to the ith couple (follows the order of unions)
        num_people + total_num_children: (int) updated total number of people in
            the community/graph after adding children to the current generation
            of marriages
        num_children_each_couple: (np.array of len(unions) of int) each entry is
            a random draw from the child_probs distribution, how many children
            the ith couple of unions has
        indices: (dictionary) mapping the current generations' names (int) to
            index (row/column number) in the current distance matrix D, updated
            to include the new children (new children are added to D outside of
            this function)
    """
    families = []
    child_edges = []

    num_children_each_couple = np.random.choice(np.array(list(child_probs.keys())), p=np.array(list(child_probs.values())), size=len(unions))
    total_num_children = sum(num_children_each_couple)
    biggest_name = num_people

    for union, num_children in zip(unions, num_children_each_couple):
        if num_children == 0:
            families.append([])
        else:
            children = [biggest_name + 1 + child for child in range(num_children)]
            biggest_name += num_children  # the next 'name' to use, next available index
            father_edges = [(union[0], child) for child in children]
            mother_edges = [(union[1], child) for child in children]
            child_edges.append(father_edges + mother_edges)
            families.append(children)
    # flatten the list of family edges
    child_edges = [edge for family in child_edges for edge in family]
    #max_ind = max(indices.values())
    #indices = indices | {child + num_people: ind for ind, child in zip(range(max_ind+1, max_ind+1+total_num_children), range(total_num_children))}
    return child_edges, families, num_people + total_num_children, num_children_each_couple


#%%

# n = num_people
# people_retained = prev_generation_still_single
def update_distances(distance_matrix, n, unions, families, indices, people_retained):
    """
    Build a distance matrix that keeps track of how far away each node is from
    each other. Need to update distances after new nodes added to graph (i.e.,
    after adding children)
    PARAMETERS:
        distance_matrix (array): "old" matrix of distances, previous generation
            n (int): number of nodes currently in graph
        unions: (list of tuple of int) marriages in the current
            generation (output of add_marriage_edges())
        no_unions: (list of int) list of nodes in the current generation
            which did not marry
        families: (list of list of int) entries are lists of children
            pertaining to the ith couple (follows the order of unions)
            (output of add_children_edges())
        indices (dictionary): maps node name (an int) to index number
            (row/column number) in the current distance matrix D.
        people_retained (list): nodes which are unmarried, but which will attempt to
            marry someone in the next generation of people
    RETURNS:
        D1 (array): "new" (updated) matrix of distances
            for the current generation
        new_indices: (dictionary) mapping the current generations' names (int)
            to index (row/column number) in the current distance matrix D
    """
    print('in update_distances()')
    # initialize new matrix
    # print('indices:', indices)
    num_children = len([child for fam in families for child in fam])
    # print('num_children:', num_children)
    num_people_retained = len(people_retained)
    # print('people_retained:', people_retained)
    # print('num_people_retained:', num_people_retained)

    D1 = np.zeros((num_children + num_people_retained,
                   num_children + num_people_retained))

    new_indices = {person:k for k, person in enumerate(people_retained + [child for fam in families for child in fam])}
    # print('new_indices:', new_indices)

    if num_people_retained > 0:
        # the upper num_retained_people x num_retained_people block of D1 is just a slice from D
        # this builds out the "second quadrant" of the D1 matrix
        D1[new_indices[people_retained[0]]:new_indices[people_retained[-1]]+1,
           new_indices[people_retained[0]]:new_indices[people_retained[-1]]+1] = distance_matrix[[indices[k] for k in people_retained]][:, [indices[k] for k in people_retained]]

        # now build out the distances from people_retained to the new generation (descendants of people_retained's cousins)
        # this builds out the "first and third" quadrants of the D1 matrix
        for rp, fam in product(people_retained, zip(unions, families)):
            # find minimum distance between the retained person and a representative child in the family
            # print('rp:', rp)
            # print('fam:', fam)
            father = fam[0][0]
            # print('father:', father)
            mother = fam[0][1]
            # print('mother:', mother)
            children = fam[1] # all children in family will have same distance from retained person rp
            # print('children:', children)
            if len(children) == 0:
                # ie the union, family pair has no children listed,
                # the end of a line
                continue

            possible_distances = distance_matrix[indices[rp], [indices[father], indices[mother]]]
            # print('possible_distances:', possible_distances)
            if (possible_distances == -1).all():
               d = -1
            else:
                possible_distances = possible_distances[possible_distances > -1]
                d = np.min(possible_distances) + 1 # account for the additional edge between father/mother and child
            # print('d:', d)
            D1[new_indices[rp], [new_indices[ch] for ch in children]] = d
            # print('D1 part 1:', D1)
            D1[[new_indices[ch] for ch in children], new_indices[rp]] = d
            # print('D1 part 2:', D1)

    # compute new distances, between new generation
    # this builds out the "fourth quadrant" of the D1 matrix
    unions = list(unions)
    # print('unions:', unions)
    for u, union in enumerate(unions):
        u_children = families[u]
        # print('u_children:', u_children)

        for other in unions[u:][1:]:
            o_children = families[unions.index(other)]
            # print('o_children:', o_children)

            # find all possible distances from union to other
            d1 = distance_matrix[indices[union[0]]][indices[other[0]]]
            d2 = distance_matrix[indices[union[1]]][indices[other[0]]]
            d3 = distance_matrix[indices[union[0]]][indices[other[1]]]
            d4 = distance_matrix[indices[union[1]]][indices[other[1]]]

            possible_distances = np.array([d1, d2, d3, d4])
            if (possible_distances == -1).all():
                d = -1
            else:
                possible_distances = possible_distances[possible_distances > -1]  # IE where NOT infinite
                # compute distance between children of union and children of other
                d = np.min(possible_distances) + 2
            for uc in u_children:
                for oc in o_children:
                    D1[new_indices[uc]][new_indices[oc]] = d
                    # print('D1 part 3:', D1)
                    D1[new_indices[oc]][new_indices[uc]] = d
                    # print('D1 part 4:', D1)

        for c, ch in enumerate(u_children):
            for sibling in u_children[c:][1:]:
                D1[new_indices[ch]][new_indices[sibling]] = 2
                # print('D1 part 5:', D1)
                D1[new_indices[sibling]][new_indices[ch]] = 2
                # print('D1 part 6:', D1)

    directory_name = "output"
    file_path_000 = makeOutputDirectory(directory_name, name + str('_D1'))
    file_path0000 = os.path.join(file_path_000, 'D1array_' + str(np.random.randint(0, 10000)) + '.txt')
    np.savetxt(file_path0000, D1.astype(int), fmt='%d')
    return D1, new_indices

def initialize_network(num_people, marriage_dist, eps, when_to_stop, children_dist):
    all_marriage_edges = []
    all_marriage_distances = []
    all_temp_marriage_edges = []
    all_temp_marriage_distances = []
    all_children_per_couple = []

    G = nx.DiGraph()

    distance_matrix = np.ones((num_people, num_people)) * -1  # starting off, everyone is infinite distance away
    np.fill_diagonal(distance_matrix, 0)  # everyone is 0 away from themselves, also weirdly done inplace

    indices = {node + 1: k for k, node in enumerate(range(num_people))}  # name:index
    generation_of_people = list(indices.keys())
    # explicitly add our first generation of nodes (otherwise we will fail to
    # add those who do not marry into our graph).  All future generations are
    # connected either through marriage or through parent-child arcs
    G.add_nodes_from(generation_of_people, layer=0, immigrant=True, parents=0)

    # get probabilities of possible finite distances to use in marriage function
    # and normalize it
    marriage_probs = get_marriage_probabilities(marriage_dist, when_to_stop,
                                                eps=eps)  # hand in the ORIGINAL MARRIAGE DISTRIBUTION
    # maybe don't normalize so that you will (approximately) retain the prob_inf + prob_finite + prob_single = 1
    marriage_probs = {key: value / sum(marriage_probs.values()) for key, value in
                      zip(marriage_probs.keys(), marriage_probs.values())}
    lim = len(marriage_probs)

    # ditto for the child distribution
    child_probs = get_child_probabilities(children_dist)
    # ??? make probabilities non-negative (some entries are effectively zero, but negative)
    child_probs = {key: value if value > 0 else 0 for key, value in zip(child_probs.keys(), child_probs.values())}
    child_probs = {key: value / sum(child_probs.values()) for key, value in
                   zip(child_probs.keys(), child_probs.values())}
    return G, indices, distance_matrix, generation_of_people, all_marriage_edges, all_marriage_distances, all_temp_marriage_edges, all_temp_marriage_distances, all_children_per_couple, marriage_probs, lim, child_probs

def save_results(summary_statistics, G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, output_path, name, dies_out):
    df = pd.DataFrame(data=summary_statistics,
                      columns=['num_people (excluding initial setup)', 'num_immigrants', 'num_children',
                               'num_marriages', 'prob_inf_marriage', 'prob_finite_marriage',
                               'prob_inf_marriage(eligible_only)', 'prob_finite_marriage(eligible_only)',
                               'num_connected_components'])
    df.index.name = 'generation'
    df.to_csv(os.path.join(output_path, str(name) + '_summary_statistics.csv'))
    # Gname = "{}/{}_G_withsetup.gpickle".format(output_path, name)   # save graph
    # nx.write_gpickle(G, Gname)
    Gname = "{}/{}_G_withsetup.pkl".format(output_path, name)  # save graph
    with open(Gname, 'wb') as out:
        pickle.dump(G_connected, out)

    # now save a reduced version of the graph, without the setup people
    Gname = "{}/{}_G_no_setup.pkl".format(output_path, name)  # save graph
    with open(Gname, 'wb') as out:
        pickle.dump(G, out)

    Uname = "{}/{}_marriage_edges".format(output_path, name) + '.pkl'  # save unions
    with open(Uname, 'wb') as fup:
        pickle.dump(all_marriage_edges, fup)
    Dname = "{}/{}_marriage_distances".format(output_path, name) + '.pkl'  # save marriage distances
    with open(Dname, 'wb') as myfile:
        pickle.dump(all_marriage_distances, myfile)
    Cname = "{}/{}_children_per_couple".format(output_path, name) + '.pkl'  # save children
    with open(Cname, 'wb') as fcp:
        pickle.dump(all_children_per_couple, fcp)

    paj = format_as_pajek(G, name)
    with open('{}/model-{}-oregraph.paj'.format(output_path, name), 'w') as o:
        o.writelines(paj)
    print(type(G))
    print(type(G_connected))
    print(type(all_marriage_edges))
    print(type(all_marriage_distances))
    print(type(all_children_per_couple))
    print(type(dies_out))
    print(type(None))
    return G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out

#%%
# TODO: what do we actually want this to return?
from pandas.core.ops.mask_ops import libmissing
@profile
def human_family_network_variant(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, children_dist, name, when_to_stop=np.inf, num_gens=np.inf, save=True, out_dir='output', eps=0):
    """
    This is the main driver for implementing the Variant Target Model (see
    Chapter 5 in Kolton's thesis).
    PARAMETERS:
        num_people (int): number of people (nodes) to include in initial
            generation
        marriage_dist: (list of int) one entry per marriage indicating how many
            generations between spouses (reported in the number of parent-child
            edges crossed so that distance between siblings is 2) in the
            specified dataset.  If no common ancestor (IE an infinite distance)
            then the corresponding entry is -1
        prob_finite_marriage (float): probability of marriage being drawn from
            the finite portion of marriage_dist (herein defined and treated as
            finite_marriage_probs)
        prob_inf_marriage (float): probability of marriage to a non-connected
            person.
            NOTE: prob_finite_marriage + prob_inf_marriage + prob_single = 1
        children_dist: (list of int) one entry per pair of parents, indicating
            how many child edges each parent in the couple share
        name: (str) name for prefix of saved files
        when_to_stop: (int) target number of nodes to capture.  If supplied, the
            model will run until this target number of nodes (all together, not
            just the size of the current generation) is surpassed.
        num_gens (int): max number of generations for network to grow beyond the
            initial generation.  Default is np.inf.  If np.inf, then the model
            will run until the number of nodes in the example network is
            surpassed and then stop.
        save (bool): should you save pickled versions of networkx graphs, model
            union edges, and distance to union and child distributions, as well
            as png graphs of distance to union updates each generation as well
            as an xlsx file summarizing model growth?
        out_dir (str): destination directory for save output.  Each run will
            create an out_dir/{name}_{int}/ file structure, counting version
            number automatically
    RETURNS:
    G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, output_path
        G (nx.DiGraph):  modeled network.  This graph can have MANY COMPONENTS.
            Each node has the following attributes: 1) layer: either an int
                indicating the vertice's generation 2) immigrant: bool
                indicating whether the vertex had parents in the graph when the
                vertex was originally added to the graph.  All vertices in the
                initial generation are immigrants, as are those brought in to
                form additional infinite-distance unions. 3) parents: int in
                {0, 1, 2}, counting the number of incoming parent-child type
                edges.  When joining up components into a single component
                (forming G_connected), we only allow immigrant vertices to have
                at most two incoming ancestral lines (i.e. paternal and
                maternal).
            Each edge has the following attributes: 1) Relationship: str, either
                "Marriage" to indicate a union edge or "Parent-Child".  Again,
                Parent-Child edges go from parent to child.
        G_connected (nx.DiGraph): modeled network formatted as above, but
            with auxillary ancestry forming a single connected component out of
            the (many) components in G.  Vertices in G_connected may have their
            'layer' attribute set to 'setup', indicating that the vertex is part
            of an auxiliary ancestral line which was introduced to connect two
            immigrant vertices in different components via a common ancestor.
            Additionally, these now connected immigrant vertices will have their
            'parents' attribute incremented to either 1 or 2.
            G is a subgraph of G_connected.
        all_marriage_edges (list of tuples of int): this lists all union-type
            edges added to G (indicated by node attribtute
            "Relationship"=="Marriage").  NOTE:  The auxiliary ancestry added
            prior to generation 0 contains no union-type edges, so that G and
            G_connected share the same list of union edges.
        all_marriage_distances (list of int): this lists all distances prior to
            union for each union edge in all_marriage_edges (orders match).
            NOTE: these distances are those as observed in G with auxiliary
            ancestry prior to generation 0.  These distances are also the
            distances for the unions in G_connected.
        all_children_per_couple (list of int): this lists the number of children
            per union edge in G and in G_connected.  NOTE: because the auxiliary
            ancestry contains no union edges, the parent-child relationships
            encoded therein do not affect this distribution.  That is, G and
            G_connected have the exact same list of children per union.
        dies_out (bool):  False indicates that the model exited successfully---
            generations 0 to L (inclusive) contain more than when_to_stop nodes
            in total OR L >= num_gens.  True else.
        output_path (str): None if save==False, else out_dir/{name}_{int},
            counting version number automatically.
    """
    if save:
        output_path = makeOutputDirectory(out_dir, name)
    else:
        output_path = ''

    total_num_single = 0
    dies_out = False

    marriage_dist_array = np.array(marriage_dist)
    finite_only_marriage_dist = marriage_dist_array[marriage_dist_array != -1]

    G, indices, distance_matrix, generation_of_people, all_marriage_edges, all_marriage_distances, all_temp_marriage_edges, all_temp_marriage_distances, all_children_per_couple, marriage_probs, lim, child_probs = finitialize_network(num_people, marriage_dist, eps, when_to_stop, children_dist)
    # grow the network until there are the at least as many nodes (not counting
    # those created to impose the distances on generation 0, but counting those
    # in generation 0) as when_to_stop

    prev_generation_still_single = []

    summary_statistics = [] # will hold ordered tuples of integers (# total people in graph, # immigrants, # num_children, # num marriages, prob_inf_marriage, prob_finite_marriage, prob_inf_marriage(eligible_only), prob_finite_marriage(elegible_only))
    i = 1
    tem_num_people = 0

    update_distances_counter = 1
    while (tem_num_people < when_to_stop) & (i < num_gens):
        print('in while loop')

        # create unions between nodes to create next generation
        # unions, no_unions, all_unions, n, m, infdis, indices = add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis, indices)
        if len(generation_of_people) == 0:
            dies_out = True
            break

        # update your current finite marriage probabilities to favor those which are yet underrepresented
        # if i > 1 and len(set(all_marriage_distances)) > 2:
        if (i > round(lim/2)+1) and (i <= round(lim/2*3/2)+1):
            print('going into new_marriage_probs')
            new_marriage_probs = get_difference_in_probabilities(marriage_probs, all_temp_marriage_distances, num_people, name, i, output_path, plot=save, eps=eps)
        elif i > round(lim/2*3/2)+1:
            print('going into new_marriage_probs')
            new_marriage_probs = get_difference_in_probabilities(marriage_probs, all_marriage_distances, num_people - num_setup_people, name, i, output_path, plot=save, eps=eps)
        else:
            # if this is the first generation beyond the initial set up OR
            # if you don't yet have more than one unique distance in your list
            # of marriage edges, then just default to the unaltered finite-
            # distance marriage distribution

            temp_marriage_probs = {key:value/sum(marriage_probs.values()) for key, value in zip(marriage_probs.keys(), marriage_probs.values())}
            v = list(temp_marriage_probs.values())
            k = list(temp_marriage_probs.keys())

            if i == 1:
              new_marriage_probs = {key:value/sum(v[0:i*2]) for key, value in zip(k[0:i*2], v[0:i*2])}
            elif i == 2:
              if i in k[0:i*2]:
                new_marriage_probs = {key:value/sum(v[0:k.index((i-1)*2)+1]) for key, value in zip(k[0:k.index((i-1)*2)+1], v[0:k.index((i-1)*2)+1])}
              else:
                t = 0 
            else:
              if ((i-1)*2) in k[0:i*2]: 
                new_marriage_probs = {key:value/sum(v[0:k.index((i-1)*2)+1]) for key, value in zip(k[0:k.index((i-1)*2)+1], v[0:k.index((i-1)*2)+1])}
              else:
                t = 0

        new_prob_inf_marriage = new_marriage_probs[-1]
        new_prob_finite_marriage = sum(new_marriage_probs.values()) - new_marriage_probs[-1] - new_marriage_probs[0]
        if i == 1:
            print('going into add_marriage_edges()')
            unions, num_immigrants, marriage_distances, immigrants, prev_generation_still_single, stay_single_forever = add_marriage_edges(people=generation_of_people,
                                                                                                                                           prev_people=prev_generation_still_single,
                                                                                                                                           num_people=num_people,
                                                                                                                                           marriage_probs=new_marriage_probs,
                                                                                                                                           prob_marry_immigrant=new_prob_inf_marriage,
                                                                                                                                           prob_marry=new_prob_finite_marriage,
                                                                                                                                           distance_matrix=distance_matrix,
                                                                                                                                           indices=indices,
                                                                                                                                           original_marriage_dist=marriage_dist,
                                                                                                                                           eps=eps)

        else:
            print('going into add_marriage_edges()')
            unions, num_immigrants, marriage_distances, immigrants, prev_generation_still_single, stay_single_forever = add_marriage_edges(
                people=generation_of_people,
                prev_people=prev_generation_still_single,
                num_people=num_people,
                marriage_probs=new_marriage_probs,
                prob_marry_immigrant=new_prob_inf_marriage,
                prob_marry=new_prob_finite_marriage,
                distance_matrix=distance_matrix,
                indices=indices,
                original_marriage_dist=marriage_dist,
                eps=eps)

        total_num_single += stay_single_forever
        G.add_nodes_from(immigrants, layer=i-1, immigrant=True, parents=0)
        G.add_edges_from(unions, Relationship="Marriage")       
        if i > round(lim/2*3/2):              
            all_marriage_edges += list(unions)  
            all_marriage_distances += marriage_distances 
        else:
            num_setup_people = num_people  
            all_temp_marriage_edges += list(unions)
            all_temp_marriage_distances += marriage_distances

        temp_distance_matrix = distance_matrix
        for j in range(num_immigrants): 
            # add non-connected people to distance matrix
            r = np.ones((1, len(indices) + 1 + j)) * -1  # -1 is infinite distance
            r[0, -1] = 0  # distance to self is 0
            c = np.ones((len(indices) + j, 1)) * -1  # -1 is infinite distance
            distance_matrix = np.hstack((distance_matrix, c))
            distance_matrix = np.vstack((distance_matrix, r))

        max_ind = max(indices.values())
        indices = indices | {immigrant + num_people + 1:ind for ind, immigrant in zip(range(max_ind + 1, max_ind+1+num_immigrants), range(num_immigrants))}  # +1 since we begin counting people at 1, 2, ... not at 0, 1, ...
        
        if i > round(lim/2*3/2)+1:               
            stats = [num_people - num_setup_people, num_immigrants]
            print("IF num people: ", num_people,"num set people = ", num_setup_people)
        num_people += num_immigrants
        current_people = num_people

        # add children to each marriage
        child_edges, families, num_people, num_children_per_couple = add_children_edges(unions, num_people, child_probs)
        added_children = num_people - current_people 

        # directory_name = "output"
        # file_path_0 = makeOutputDirectory(directory_name, name + str('_distance_matrix'))
        # file_path00 = os.path.join(file_path_0, 'Darray1_' + str(i) + '.txt')
        # np.savetxt(file_path00, distance_matrix.astype(int), fmt='%d')

        # update distances between nodes
        print('going into update_distances() {}'.format(update_distances_counter))
        distance_matrix, indices = update_distances(distance_matrix, num_people, unions, families, indices, prev_generation_still_single)
        print('out of update_distances()')
        update_distances_counter += 1

        # file_path = os.path.join(file_path_0, 'Darray2_' + str(i) + '.txt')
        # file_path2 = os.path.join(file_path_0, 'tempDarray' + str(i) + '.txt')
        # np.savetxt(file_path, distance_matrix.astype(int), fmt='%d')
        # np.savetxt(file_path2, temp_distance_matrix.astype(int), fmt='%d')

        generation_of_people = [key for key in indices.keys() if key not in prev_generation_still_single]  # only grab the new people
        G.add_nodes_from(generation_of_people, layer=i, immigrant=False, parents=2)
        G.add_edges_from(child_edges, Relationship='Parent-Child')
        all_children_per_couple += list(num_children_per_couple)

        # print("all_temp_marriage_distances= ", len(all_temp_marriage_distances)," increased= ", len(all_temp_marriage_distances)-temp )
        temp = len(all_temp_marriage_distances)
        if i > round(lim/2*3/2)+1:  
            tem_num_people += num_immigrants + added_children 
            # stats.append(len(generation_of_people))
            stats.append(sum(num_children_per_couple))
            stats.append(len(unions))
            stat_prob_marry = len(all_marriage_distances) * 2 / len(G)   
            # print("len(all_marriage_distances)= ",len(all_marriage_distances), " increased= ", len(all_marriage_distances)-temp2)
            temp2 = len(all_marriage_distances)
            if len(all_marriage_distances) > 0:
                stat_frac_inf = sum(np.array(all_marriage_distances) == -1) / len(all_marriage_distances)
            else:
                stat_frac_inf = np.nan
            stats.append(stat_prob_marry * stat_frac_inf)
            stats.append(stat_prob_marry * (1 - stat_frac_inf))

            # now recalculate marriage stats using only the eligible nodes
            # (IE not those preceding gen 0, not prev_gen_still_single, and not
            # leaf nodes---only those nodes that were given the chance to marry)
            stat_prob_marry = len(all_marriage_distances) * 2 / (len(G) - num_setup_people - len(prev_generation_still_single) - sum(num_children_per_couple))
            stats.append(stat_prob_marry * stat_frac_inf)
            stats.append(stat_prob_marry * (1 - stat_frac_inf))
            stats.append(nx.number_connected_components(nx.Graph(G)))

            summary_statistics.append(stats)

        i += 1
        # print("i= ",i)
    # build a single component out of the graph's many components
    print('out of while loop')
    G_connected = nx.DiGraph()  #G.subgraph([node for node, d in G.nodes(data=True) if d['layer'] != 'setup']).copy()

    for component in nx.connected_components(nx.Graph(G)):
        if len(G_connected) == 0:
            G_connected = G.subgraph(component).copy()
            continue
        else:
            component = G.subgraph(component).copy()
            # get lists of candidate nodes (immigrants without two parent lines), by layer
            # in both our largest conglomorate G_connected and in our component of interest
            G_connected_parent_leaf_layers = set(nx.get_node_attributes(G_connected, 'layer').values())
            if 'setup' in G_connected_parent_leaf_layers:
                G_connected_parent_leaf_layers.remove('setup')

            G_connected_parent_leafs = {layer:[node for node, d in G_connected.nodes(data=True) if (d['immigrant']==True and d['layer']==layer and d['parents'] < 2)] for layer in G_connected_parent_leaf_layers}

            component_parent_leaf_layers = set(nx.get_node_attributes(component, 'layer').values())
            # component_parent_leafs = [(node, d) for node, d in component.nodes(data=True) if d['immigrant'] == True]
            component_parent_leafs = {layer:[node for node, d in component.nodes(data=True) if (d['immigrant']==True and d['layer']==layer and d['parents'] < 2)] for layer in component_parent_leaf_layers}

            component_layer_of_choice = None
            G_connected_layer_of_choice = None
            for comp_layer in sorted(component_parent_leafs.keys())[::-1]:
                if comp_layer in G_connected_parent_leafs:
                    component_layer_of_choice = comp_layer
                    G_connected_layer_of_choice = comp_layer
                elif comp_layer+1 in G_connected_parent_leafs:
                    component_layer_of_choice = comp_layer
                    G_connected_layer_of_choice = comp_layer+1
                elif comp_layer-1 in G_connected_parent_leafs:
                    component_layer_of_choice = comp_layer
                    G_connected_layer_of_choice = comp_layer-1
                # this shouldn't ever fail?  Unless you run out of parent nodes in layer 0?

            # grab random a random parent leaf node in G_connected and component at the specified layers
            node1 = np.random.choice(G_connected_parent_leafs[G_connected_layer_of_choice])
            node2 = np.random.choice(component_parent_leafs[component_layer_of_choice])
            distance = np.random.choice(finite_only_marriage_dist)

            nodes_to_add = [k for k in range(num_people+1, num_people+distance)]
            num_people += len(nodes_to_add)
            G_connected = nx.union(G_connected, component)
            G_connected.add_nodes_from(nodes_to_add, layer='setup', immigrant=True, parents=1)
            husband_line = nodes_to_add[:len(nodes_to_add)//2+1][::-1] + [node1]   # path will go FROM first TO last node in list
            wife_line = nodes_to_add[len(nodes_to_add)//2:] + [node2]  # path will go FROM first TO last node in list
            common_ancestor = husband_line[0]

            nx.add_path(G_connected, husband_line, Relationship='Parent-Child')
            nx.add_path(G_connected, wife_line, Relationship='Parent-Child')
            # and update node 1 and node 2 to say how many incoming parent edges they now have, also for their common ancestor node
            nx.set_node_attributes(G_connected, {node1: {'parents': G_connected.nodes[node1]['parents'] + 1,
                                                         'has_setup':True},
                                                 node2: {'parents': G_connected.nodes[node2]['parents'] + 1,
                                                         'has_setup':True},
                                                 common_ancestor: {'parents': 0}})

    if save:
        print('in last save')
        G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out = save_results(summary_statistics, G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, output_path, name, dies_out)
        return G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, None
    else:
        return G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, None

#%%
"""
below is example code to run the model
"""
# torres_strait has negative probability

name = 'torshan'
num_people = 100
eps = 0
marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, output_path = human_family_network_variant(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, child_dist, name, save=True, when_to_stop=size_goal, eps=eps)

#%%

# def find_start_size(name, out_directory='start_size', filename='start_size', max_iters=100, dies_out_threshold=5,  verbose=False, save_start_sizes=True, random_start=True, return_counter=False): # n = number of initial nodes
#     counter = 0
#
#     filename = name + '_' + filename
#     marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
#     greatest_lower_bound = 2
#     least_upper_bound = size_goal
#
#     if random_start:
#         num_people = np.random.randint(greatest_lower_bound, size_goal)
#     else:
#         num_people = size_goal//2
#     dies_out = 0 # counter for the number of times the model dies out
#
#     start_sizes = [num_people]
#     while dies_out != dies_out_threshold: # while the number of times the model dies out is not equal to the threshold of dying:
#
#         for i in range(max_iters):
#             counter += 1
#             G, G_reduced, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies, output_path = human_family_network_variant(num_people,
#                                                                                                                 marriage_dist,
#                                                                                                                 prob_finite_marriage,
#                                                                                                                 prob_inf_marriage,
#                                                                                                                 child_dist,
#                                                                                                                 name,
#                                                                                                                 when_to_stop=size_goal,
#                                                                                                                 save=False)
#             if dies:
#                 dies_out += 1
#             if dies_out > dies_out_threshold:
#                 break
#
#         if greatest_lower_bound >= least_upper_bound - 1:
#             # IE the ideal lies between these two integers
#             # so return the larger
#             num_people = least_upper_bound
#             break
#         elif dies_out == dies_out_threshold:
#             break
#         elif dies_out > dies_out_threshold:  # we want to increase num_people
#             greatest_lower_bound = num_people  # current iteration died out too frequently.  Won't need to search below this point again.
#             num_people = (num_people + least_upper_bound) // 2 # midpoint between num_people and size_goal
#             dies_out = 0
#
#         elif dies_out < dies_out_threshold: # we want to decrease num_people
#             least_upper_bound = num_people  # current iteration died out too infrequently.  Won't need to search above this point again
#             num_people = (greatest_lower_bound + num_people) // 2 # midpoint between 2 and num_people
#             dies_out = 0
#
#         if verbose:
#             print('greatest_lower_bound: ', greatest_lower_bound)
#             print('least_upper_bound: ', least_upper_bound)
#             print('starting population: ', num_people)
#         start_sizes.append(num_people)
#
#
#     if save_start_sizes:
#         if not os.path.exists(out_directory):
#             os.makedirs(out_directory)
#         filename = find_file_version_number(out_directory, filename, extension='.txt')
#         # save a text file (one integer per line)
#         with open(os.path.join(out_directory, filename +'.txt'), 'w') as outfile:
#             outfile.writelines([str(k) + '\n' for k in start_sizes])
#         # save the actual object
#         with open(os.path.join(out_directory, filename + '.pkl'), 'wb') as outfile:
#             pickle.dump(start_sizes, outfile)
#
#     if return_counter:
#         return start_sizes, counter
#     else:
#         return start_sizes
#%%

# def repeatedly_call_start_size(name, out_directory='start_size', iters=5, max_iters=100, dies_out_threshold=5,  verbose=False, save_start_sizes=True, save_individual_start_sizes=False, random_start=True, show_plot=False):
#     #find out directory.  Every iteration in this function call will output to the same file
#     out_dir = makeOutputDirectory(out_directory, name)
#
#     if save_start_sizes:
#         # create the folder, text file to which EACH start_size list will be appended
#         filename = name + '_' + 'start_size'
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
#         if not os.path.exists(os.path.join(out_dir, filename + '.txt')):
#             with open(os.path.join(out_dir, filename + '.txt'), 'w'):
#                 pass
#
#     to_plot = []
#     for i in range(iters):
#         start_sizes = find_start_size(name,
#                                       out_dir,
#                                       max_iters=max_iters,
#                                       dies_out_threshold=dies_out_threshold,
#                                       verbose=verbose,
#                                       save_start_sizes=save_individual_start_sizes,
#                                       random_start=random_start)
#         to_plot.append(start_sizes)
#
#         if save_start_sizes:
#             # save a text file (one line per run of find_start_size())
#             with open(os.path.join(out_dir, filename +'.txt'), 'a') as outfile:
#                 outfile.write(str(start_sizes))
#                 outfile.write('\n')
#
#         if save_start_sizes:
#             # save the unaltered (entries will be of differing lengths) set of start sizes
#             # as a pickle object for later use
#             with open(os.path.join(out_dir, filename + '.pkl'), 'wb') as outfile:
#                 pickle.dump(to_plot, outfile)
#
#     # prep each entry of to_plot.  Not every iteration will have the same num
#     # of entries.  Just repeat the last entry as necessary
#     length = np.max([len(k) for k in to_plot])
#     for start_sizes in to_plot:
#         while len(start_sizes) < length:
#             start_sizes.append(start_sizes[-1])
#
#     fig = plt.figure(figsize=(12,9), dpi=300)
#     for k in range(len(to_plot) -1):
#         plt.plot(to_plot[k], color='k', linewidth=0.5, alpha=0.65)
#
#     # plot the last one with a label
#     plt.plot(to_plot[-1], color='k', linewidth=0.5, alpha=0.65, label='individual run')
#     # make the plot display in text the final average starting value
#     avg_run = np.mean(to_plot, axis=0)
#     plt.text(length - 1.25, avg_run[-1]+3, str(round(avg_run[-1])), fontsize=16)
#     # now plot the average, bolded
#     plt.plot(avg_run, color='k', linewidth=7, alpha=0.8, label='mean')
#     plt.xticks([k for k in range(length)], labels=[k for k in range(1, length+1)], fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.title(name + ' initial population search', fontsize=24)
#     plt.legend()
#     plt.ylabel('starting population', fontsize=16)
#     plt.xlabel('iterations', fontsize=16)
#     plt.savefig(os.path.join(out_dir, name + '_starting_size_graph.png'), format='png')
#     if show_plot:
#         plt.show()
#
#     return avg_run

#%%

# if __name__=='__main__':
#     name = 'arara'
#     num_people = 43
#     marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)

#     # children_dist = child_dist
#     # when_to_stop = size_goal
#     # num_gens = np.inf
#     # out_dir = 'output'
#     # save = True
#     G, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, output_path = human_family_network_variant(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, child_dist, name, save=True, when_to_stop=size_goal)
