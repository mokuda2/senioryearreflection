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
from operator import itemgetter

from setuptools import setup
from Cython.Build import cythonize

import random
import math

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

def get_difference_in_probabilities(target_probs, current, num_people, name, gen_num, outpath, eps=1e-7, plot=False):
    """
    This method accepts both the target marriage distribution AND the
    current-state model marriage distribution.  It will subtract the
    current-state from the target probabilites, flooring at some positive
    epsilon.  The returned probabiltiy distribution will then favor any
    marriages of those distances which have not yet been drawn in proportion
    with the target distribution's frequency for that distance

    PARAMETERS:
        target_probs (dictionary): keys are marriage distances, values are
            probabilities.  This is the result of
            get_probabilities(marriage_dist) (called finite_marriage_probs
            below). This should already be normalized.
        current (list): list of marriage distances currently represented in the
            graph.
    """
    current_probs= get_marriage_probabilities(current, num_people, eps=eps)
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
# people = generation_of_people
# prev_people = prev_generation_still_single
# marriage_probs = new_marriage_probs
# prob_marry_immigrant = new_prob_inf_marriage
# prob_marry = new_prob_finite_marriage
# original_marriage_dist = marriage_dist

def find_mate(source_node, distance, graph, unions, tol, current_generation=[], previous_generation=[]):
    # """
    # By using graph search method, it will find a node in previous or in current generation
    # with the shortest desired distance from source_node.

    # Paramater:
    #     source_node (int): a single node labeled with int
    #     distance (int): desired distance
    #     graph (nx.graph): graph
    #     unions (set of ints): labeled numbers on nodes that are already married
    #     tol (int): the maximum number of iterations
    #     current_generation (set): labeled numbers on nodes that are in current generation
    #     previous_generation (set): labeled numbers on nodes that are in previous generation
    # Return:
    #     [None,False], if finding a mate is impossible or is taking too much time (iterations)
    #     [(source_node, mate_node),Ture], if finding a mate is successful
    #     [(source_node, mate_node), False], if a mate node is found, but the node
    # Caveats:
    #     unions should be set of nodes that are married (not tuple form)
    #     (Actually, if unions is in tuple form, we use for loop to check #"*")
    #     All edge attributes must be either "Marriage" or "Parent-Child"
    # """
    # setting
    graph_T = graph.reverse()

    num_dead_end = 0
    S = [[source_node]]
    path_ = []

    num_up = np.ceil(distance/2) # if distance is odd, it gose up one more than down. if distance is even, it goes up and down same time

    # modified DFS
    while len(path_) < distance + 1:
        #print("S:", S)
        # if finding a node with the distance is impossible,
        if len(S) == 0:
            # we need to choose another source node.
            #print("S is empty")
            return None, False

        # if finding a node with the distance is impossible,
        if num_dead_end == tol:
            # we need to choose another source node.
            #print("Over tol")
            return None, False

        # pop a path from S
        path_ = S.pop()

        # after reaching to the final node
        if len(path_) == distance + 1:  # since num_nodes are more than num_edges by 1
            # check if the last node is not in the current generation or previous generation
            if path_[-1] not in current_generation and path_[-1] not in previous_generation:
                #print("The last node is not either in current generation or in previous generation.")
                #print()
                num_dead_end += 1
                path_ = []
                pass

            else:
                # check if the last node is in unions
                if path_[-1] in unions:                #"*"
                    num_dead_end += 1
                    path_ = []   # reset path_ so that we can stay in the while loop
                    #print(f"The last node {path_[-1]} is already married")
                    #print()
                    pass

                else:
                    # For testing purpose
                    # check if the last node is actually the node with the shortest distance
                    graph_un = graph.to_undirected() # make the graph undirected so that we can find the shortest path's length
                    actual_length = nx.shortest_path_length(graph_un, source=source_node, target=path_[-1])
                    if distance == actual_length:
                        # success to find a mate
                        return (source_node, path_[-1]), True
                    else:
                        # success to find a mate, but not the shortest distance between them
                        return (source_node, path_[-1]), False

                        ## if the union does not have the shortest distance, then find other mate
                        # num_dead_end += 1
                        # path_ = [] # reset path_ so that we can stay in the while loop
                        # print(f"The path {path_} is not the shortest path.")
                        # print(f"The actual length of the shortest path is {actual_length}.")
                        # print()
                        # pass

        else:
            # up
            if len(path_) < num_up+1: # since num_nodes are more than num_edges by 1
                #print("up")
                # c_node is the current node
                c_node = path_[-1]

                parents = set(graph_T[c_node])
                # Print progress
                #for i in parents:
                    #print(i,"::",graph_T[c_node][i]["Relationship"])
                parents_ = [n for n in parents if graph_T[c_node][n]["Relationship"] != "Marriage"]

                # if the c_node has no parent nodes
                if len(parents_) == 0:
                    num_dead_end += 1
                    pass # ignore the c_node and pop another node in S, which will be another parent or previous another parent

                else:
                    # find parents nodes and randomly order them
                    chosen_parents = np.random.choice(parents_,2, replace=False) # remove union nodes and leave only choices of parents nodes

                    # make new path_ and append them in S
                    for walk in chosen_parents:
                        S.append(path_ + [walk])

            # down
            else:
                #print("down")
                # c_node is the current node
                c_node = path_[-1]

                children = set(graph[c_node])
                # Print progress
                #for i in children:
                #    print(i,"::",graph[c_node][i]["Relationship"])
                #print("path_:", path_)

                children_ = [n for n in children if graph[c_node][n]["Relationship"] != "Marriage"] # Remove unions
                children_ = [n for n in children_ if n != source_node] # preventing from matching target node to be the same source_node

                # if the c_node had no children
                if len(children_) == 0:
                    num_dead_end += 1
                    pass

                else:
                    # find children nodes and randomly order them
                    chosen_children = np.random.choice(children_,len(children_), replace=False)

                    # make new path_ and append them in S
                    for walk in chosen_children:
                        S.append(path_ + [walk])


def add_marriage_edges_random(graph, people, prev_people, num_people, marriage_probs, prob_marry_immigrant, prob_marry, D, indices, original_marriage_dist, tol=0, eps=1e-7):
#     Forms both infinite and finite distance marriages in the current generation
#
#     INVERTED: FORMS FINITE MARRAIGES FIRST
#     PARAMETERS:
#         graph: (nx.Graph) the graph of the population this generation
#         people:  (list) of the current generation (IE those people elligible for
#                 marriage)
#         prev_people: (list) of those in the previous generation who are yet unmarried.
#         num_people: (int) the number of nodes/people currently in the graph
#         marriage_probs: (dictionary) keys are marriage distances, values
#             are probabilities.  Note that this dictionary should only include
#             entries for NON-inifite marriage distances, should have
#             non-negative values which sum to 1, and should have a long right
#             tail (IE lots of entries which map high (beyond what occurs in the
#             example dataset) distances to zero(ish) probabilties)
#         prob_marry_immigrant: (float) the probablility that a given node will marry
#                 an immigrant (herein a person from outside the genealogical network,
#                 without comon ancestor and therefore at distance infinity from the
#                 nodes in the list 'people') (formerly 'ncp')
#         prob_marry: (float) the probability that a given node will marry another
#                 node in people
#         D: ((len(people) x len(people)) numpy array) indexed array of distance
#             between nodes in  people (siblings are distance 2)
#         indices: (dictionary) maps node name (int) to index number in D (int)
#     RETURNS:
#         unions: (list of tuples of int) marriages formed.  Entries are of two
#             types: 1) infinite distance marriages: one spouse is selected
#             uniformly at random from the community (people) while the other is
#             an immigrant to the community (IE a new node NOT listed in people).
#             2) finite distance couples: both spouses are members of the
#             community (IE listed in people).  These couples are selected at
#             random according to the marriage_probs)
#         num_immigrants: (int) the number of NEW people added to the graph.  As
#             implemented ALL new people get married to someone in the current
#             generation (herein people)
#         marriage_distances: (list of int) one entry per marriage created during
#             this function call (IE within this generation).  Each entry
#             indicates the distance between spouses through their nearest common
#             ancestor.  As before, a distnace of -1 indicates an infinite
#             distance (IE one spouse immigrated into the community)
#         didnt_marry: (list) of nodes in people who did not marry, will attempt
#             to marry someone from the next generation


    # Construct and normalize dictionary of finite marriage distance probabilities, only allowing marriages above
    # the minimum permissible distance in the original dataset
    minimum_permissible_distance = min(k for k in original_marriage_dist if k > -1)
    finite_probs = {key: val for key, val in marriage_probs.items() if key >= minimum_permissible_distance}

    if sum(finite_probs.values()) > 0:
        finite_probs = {key: val / sum(finite_probs.values()) for key, val in finite_probs.items()}

    marriage_distances = []
    unions = set()

    # Define the starting number for labeling the immigrants this generation
    next_immigrant = num_people + 1

    # Set goals for numbers of couples to marry this generation
    num_inf_couples_to_marry = round(prob_marry_immigrant * len(people) / 2)
    num_finite_couples_to_marry = round(prob_marry * len(people) / 2)

    gen_size = len(people)
    prev_gen_size = len(prev_people)

    can_marry_prev = set(prev_people)
    can_marry = set(people)

    # DEBUG PRINT
    print("\ngen_size", gen_size)
    print("prev_gen_size", prev_gen_size)
    print("marriage_probs", marriage_probs)
    print("finite_probs", finite_probs)
    print("prob_marry", prob_marry)
    print("prob_marry_immigrant", prob_marry_immigrant)
    print("finite couples goal", num_finite_couples_to_marry)
    print("infinite couples goal", num_inf_couples_to_marry)

    people_ignored = set()
    # Dictionary of distances mapping to lists of booleans representing whether forming that distance marriage was successful
    accuracy = {val:list() for val in range(-1,50)}

    num_unions = 0
    while num_unions < num_finite_couples_to_marry and can_marry:
        # Randomly select a desired distance and a candidate we would like to marry off. Attempt to find a mate tol times.
        desired_dist = np.random.choice(list(finite_probs.keys()), p=list(finite_probs.values()))
        candidate = np.random.choice(list(can_marry))
        couple, success = find_mate(candidate, desired_dist, graph, people_ignored, 15, can_marry, can_marry_prev)

        if couple is not None:
            # Update the accuracy dictionary.
            # If the random walk produced the actual geodesic distance between
            # the man and woman, add a success (True); otherwise, add a failure (False)
            if success:
                accuracy[desired_dist].append(True)
            else:
                accuracy[desired_dist].append(False)

            # Update the people_ignored set
            people_ignored.update(list(couple))
            # Update the unions set
            unions.add(couple)
            # Update the marriage_distances list
            marriage_distances.append(desired_dist)

            # Update the set of people in the current and previous generation who can marry
            can_marry.difference_update(set(couple))
            can_marry_prev.difference_update(set(couple))

            num_unions += 1

    for dist in accuracy.keys():
        if len(accuracy[dist]) > 0:
            print(f"Distance {dist}: {100*sum(accuracy[dist])/len(accuracy[dist])}% accuracy in finding shortest-distance paths")

    # Get the components of the graph (less efficient, but a solution for now) and
    # map each node in the current or previous generations (that is still unmarried) to its component number
    components = nx.connected_components(nx.Graph(graph))
    min_index = min(can_marry_prev) if can_marry_prev else 0
    eligible_people = {[x for x in component if x in range(min_index,next_immigrant) and x not in people_ignored][0]:n for n,component in enumerate(components)} # this is messy

    num_unions = 0
    while num_unions < num_inf_couples_to_marry and can_marry:
        # Uniformly select someone in the current generation who is unmarried
        candidate = np.random.choice(list(eligible_people.keys()))
        selected_component = eligible_people[candidate]

        # Try 5 times to find a partner in a separate component from our candidate
        success = False
        attempts = 0
        while attempts < 5:
            candidate2 = np.random.choice(list(eligible_people.keys()))
            attempts += 1
            if eligible_people[candidate2] != selected_component:
                # Add the new couple to the ignored set and unions set
                # and remove them from the can_marry and can_marry_prev sets (as applicable)
                people_ignored.update({candidate, candidate2})
                unions.add((candidate, candidate2))

                # Update the marriage_distances list
                marriage_distances.append(-1)

                can_marry.difference_update({candidate, candidate2})
                can_marry_prev.difference_update({candidate, candidate2})
                success = True
                num_unions += 1
                break

        # If we attempted 5 times and failed, blacklist this candidate (so we eventually break out of the while loop)
        if not success:
            eligible_people.pop(candidate)

    # DEBUG PRINT
    print("num_unions", num_unions)

    # Get a set of people who are still single by taking the set union of those still single in this generation
    # with those still single in the last generation (we have been updating these sets along the way)
    still_single = can_marry.union(can_marry_prev)

    # Bring in as many immigrants as needed to reach num_inf_couples_to_marry goal
    num_immigrants_goal = num_inf_couples_to_marry - num_unions
    num_immigrants = min(len(still_single), num_immigrants_goal)

    # Form immigrant couples by uniformly choosing num_immigrants candidates to marry an immigrant
    married_immigrant = np.random.choice(list(still_single), size=num_immigrants, replace=False)
    immigrants = list(range(next_immigrant, next_immigrant + num_immigrants))
    immigrant_couples = set(zip(married_immigrant, immigrants))

    # Update various data structures
    prev_gen_still_single = can_marry_prev.difference(set(married_immigrant))
    num_never_marry = len(prev_gen_still_single)
    unions.update(immigrant_couples)
    marriage_distances.extend([-1] * num_immigrants)
    didnt_marry = list(can_marry.difference(married_immigrant))

    # DEBUG PRINT
    print("didnt marry from current gen", len(didnt_marry))

    return unions, num_immigrants, marriage_distances, immigrants, didnt_marry, num_never_marry


def add_marriage_edges_modified(people, prev_people, num_people, marriage_probs, prob_marry_immigrant, prob_marry, D, indices, original_marriage_dist, tol=0, eps=1e-7):
#     Forms both infinite and finite distance marriages in the current generation
#
#     INVERTED: FORMS FINITE MARRAIGES FIRST
#     PARAMETERS:
#         people:  (list) of the current generation (IE those people elligible for
#                 marriage)
#         prev_people: (list) of those in the previous generation who are yet unmarried.
#         num_people: (int) the number of nodes/people currently in the graph
#         marriage_probs: (dictionary) keys are marriage distances, values
#             are probabilities.  Note that this dictionary should only include
#             entries for NON-inifite marriage distances, should have
#             non-negative values which sum to 1, and should have a long right
#             tail (IE lots of entries which map high (beyond what occurs in the
#             example dataset) distances to zero(ish) probabilties)
#         prob_marry_immigrant: (float) the probablility that a given node will marry
#                 an immigrant (herein a person from outside the genealogical network,
#                 without comon ancestor and therefore at distance infinity from the
#                 nodes in the list 'people') (formerly 'ncp')
#         prob_marry: (float) the probability that a given node will marry another
#                 node in people
#         D: ((len(people) x len(people)) numpy array) indexed array of distance
#             between nodes in  people (siblings are distance 2)
#         indices: (dictionary) maps node name (int) to index number in D (int)
#     RETURNS:
#         unions: (list of tuples of int) marriages formed.  Entries are of two
#             types: 1) infinite distance marriages: one spouse is selected
#             uniformly at random from the community (people) while the other is
#             an immigrant to the community (IE a new node NOT listed in people).
#             2) finite distance couples: both spouses are members of the
#             community (IE listed in people).  These couples are selected at
#             random according to the marriage_probs)
#         num_immigrants: (int) the number of NEW people added to the graph.  As
#             implemented ALL new people get married to someone in the current
#             generation (herein people)
#         marriage_distances: (list of int) one entry per marriage created during
#             this function call (IE within this generation).  Each entry
#             indicates the distance between spouses through their nearest common
#             ancestor.  As before, a distnace of -1 indicates an infinite
#             distance (IE one spouse immigrated into the community)
#         didnt_marry: (list) of nodes in people who did not marry, will attempt
#             to marry someone from the next generation

    def generate_couples(curr_gen, prev_gen, inf_only=False):
        # Create possible combinations of couples (either two from current gen or one from this generation and one from previous)
        possible_mixed_couples = list(itertools.product(curr_gen,prev_gen))
        possible_couples = list(itertools.combinations(curr_gen, 2))
        possible_couples.extend(possible_mixed_couples)

        # Create a dictionary of marriage distances mapping to the list of associated couples.
        # Assume that all possible marriage distances are found in marriage_probs.keys().
        # Populate the preferred_dist_dict with distances with nonzero-probability distances.
        # Populate the other_dist_dict with couples with zero-probability distances.
        # Add the rest to inf_dist_list.
        if inf_only:
            inf_dist_list = list()
        else:
            preferred_dist_dict = {distance:list() for distance in finite_probs.keys() if finite_probs[distance] > eps}
            other_dist_dict = {distance:list() for distance in finite_probs.keys() if finite_probs[distance] <= eps}

        if inf_only:
            inf_dist_list = [(man, woman) for (man, woman) in possible_couples if D[indices[man]][indices[woman]] == -1]
            return inf_dist_list

        else:
            # Populate the preferred_dist_dict and other_dist_dict dictionaries
            for (man, woman) in possible_couples:
                dist = D[indices[man]][indices[woman]]
                if dist in preferred_dist_dict.keys():
                    preferred_dist_dict[dist].append((man, woman))
                elif dist in other_dist_dict:
                    other_dist_dict[dist].append((man, woman))

        if inf_only:
            return inf_dist_list
        else:
            return preferred_dist_dict, other_dist_dict

    # Construct and normalize dictionary of finite marriage distance probabilities, only allowing marriages above
    # the minimum permissible distance in the original dataset
    minimum_permissible_distance = min(k for k in original_marriage_dist if k > -1)
    finite_probs = {key: val for key, val in marriage_probs.items() if key >= minimum_permissible_distance}

    if sum(finite_probs.values()) > 0:
        finite_probs = {key: val / sum(finite_probs.values()) for key, val in finite_probs.items()}

    marriage_distances = []
    unions = set()

    # Define the starting number for labeling the immigrants this generation
    next_immigrant = num_people + 1

    # Set goals for numbers of couples to marry this generation
    num_inf_couples_to_marry = round(prob_marry_immigrant * len(people) / 2)
    num_finite_couples_to_marry = round(prob_marry * len(people) / 2)

    # Dictate which half of this generation may marry and which may not
    gen_size = len(people)
    random_order = np.random.permutation(people)
    # Limit the people who can marry (if desired); algorithm specifies gen_size//2
    num_can_marry = gen_size//2
    can_marry = set(random_order[:num_can_marry])
    didnt_marry = list(random_order[num_can_marry:])
    prev_people = set(prev_people)

    # Compute the possible couples and the dictionaries mapping marriage distances to a set of respective couples
    preferred_dist_dict, other_dist_dict = generate_couples(can_marry, prev_people)
    inf_dist_list = generate_couples(can_marry, prev_people, inf_only = True)

    # DEBUG PRINT
    print("\ngen_size", len(people))
    print("prev_gen_size", len(prev_people))
    print("marriage_probs", marriage_probs)
    print("finite_probs", finite_probs)
    print("prob_marry", prob_marry)
    print("prob_marry_immigrant", prob_marry_immigrant)
    print("finite couples goal", num_finite_couples_to_marry)
    print("infinite couples goal", num_inf_couples_to_marry)
    print("preferred_dist_dict", [(f"Distance {str(key)} couples: {len(preferred_dist_dict[key])}") for key in preferred_dist_dict.keys()])
    print("other_dist_dict", [(f"Distance {str(key)} couples: {len(other_dist_dict[key])}") for key in other_dist_dict.keys()])
    print("inf_dist_list", f"Distance -1 couples: {len(inf_dist_list)}")

    # Compute the probability distribution we will draw from for preferred-distance unions.
    if preferred_dist_dict:
        preferred_probs = [marriage_probs[key] for key in preferred_dist_dict.keys()]
        preferred_probs /= np.sum(preferred_probs)

    people_ignored = set()
    num_unions = 0

    ctr_update = 0
    ctr_exec = 0
    counter = 0

    while num_unions < num_finite_couples_to_marry and can_marry:
        # The conditional block below determines when we re-calculate the dictionaries above.
        # One extreme: every time, the algorithm will select two people from the current generation to get married.
        # So it will form about len(can_marry)//2 unions.
        # The other extreme: every time, the algorithm will select two people from different generations to get married.
        # So it will form about len(can_marry) unions.
        ctr_exec += 1
        if not preferred_dist_dict or len(people_ignored) >= len(can_marry) or counter == 10:
            ctr_update += 1
            # Recalculate possible couples to form unions (only choose those that are not already in unions)
            can_marry = can_marry.difference(people_ignored)
            prev_people = prev_people.difference(people_ignored)
            print("in finite loop, can_marry", len(can_marry), "prev_people", len(prev_people))
            preferred_dist_dict, other_dist_dict = generate_couples(can_marry, prev_people)

            # Reduce the number of unions needed to form before this conditional executes again
            people_ignored.clear()

            # Remove marriage distances from preferred_dist_dict and other_dist_dict that cannot be realized
            dict_keys = list(preferred_dist_dict.keys()).copy()
            for key in dict_keys:
                if not preferred_dist_dict[key]:
                    preferred_dist_dict.pop(key)

            dict2_keys = list(other_dist_dict.keys()).copy()
            for key in dict2_keys:
                if not other_dist_dict[key]:
                    other_dist_dict.pop(key)

            # Compute the probability distribution we will draw from for preferred-distance unions. Normalize this probability if
            # we popped keys in the above step.
            if preferred_dist_dict:
                preferred_probs = [marriage_probs[key] for key in preferred_dist_dict.keys()]
                preferred_probs /= np.sum(preferred_probs)

        # Most often we will enter this conditional
        if preferred_dist_dict or other_dist_dict:
            # If there are still preferred-distance couples remaining, execute this code:
            if preferred_dist_dict and not all([not l for l in preferred_dist_dict.values()]):

                # Choose a marriage distance with nonzero probability according to the right distribution.
                # Then uniformly draw from the group of couples with that marriage distance to select a couple.
                counter = 0
                while True:
                    counter += 1
                    if counter == 10:
                        break
                    chosen_dist = np.random.choice(list(preferred_dist_dict.keys()), p=preferred_probs)
                    chosen_bin = preferred_dist_dict[chosen_dist]
                    # Redo if the chosen bin is empty
                    if len(chosen_bin) > 0:
                        chosen_couple_index = np.random.choice(len(chosen_bin))
                        chosen_couple = chosen_bin[chosen_couple_index]
                        # Redo the draw if the couple we chose included someone already in a union
                        if (chosen_couple[0] not in people_ignored and chosen_couple[1] not in people_ignored): break

            # If there are still other-distance couples remaining, execute this code
            elif other_dist_dict and not all([not l for l in other_dist_dict.values()]):
                # Uniformly draw from the list of couples with marriage distance closest to the support of finite_probs (that is also nonempty)
                # Every time this runs, we will be updating the preferred_dist_dict in the first conditional
                chosen_dist = min(d for d in other_dist_dict.keys() if other_dist_dict[d])
                chosen_bin = other_dist_dict[chosen_dist]
                while True:
                    chosen_couple_index = np.random.choice(len(chosen_bin))
                    chosen_couple = chosen_bin[chosen_couple_index]
                    if (chosen_couple[0] not in people_ignored and chosen_couple[1] not in people_ignored): break

            else:
                print("Shouldn't reach this point...")
                break

            # Don't let anyone in the current couple get married again in the future
            people_ignored.update(list(chosen_couple))

            # Add couple to the set of unions. Update the list of marriage distances.
            unions.add(tuple(chosen_couple))
            num_unions += 1
            marriage_distances.append(chosen_dist)

        # Exit the while loop prematurely if we ran out of finite marriage distance canditates
        else:
            break

    # Recalculate possible couples to form infinite unions
    can_marry = can_marry.difference(people_ignored)
    prev_people = prev_people.difference(people_ignored)
    inf_dist_list = generate_couples(can_marry, prev_people, inf_only = True)

    num_unions = 0
    counter = 0
    while inf_dist_list and num_unions < num_inf_couples_to_marry:
        ctr_exec += 1
        # Occasionally update the possible_inf_couples list
        if len(people_ignored) >= len(can_marry) or counter == 10:
            ctr_update += 1

            can_marry = can_marry.difference(people_ignored)
            prev_people = prev_people.difference(people_ignored)
            print("in inf loop, can_marry", len(can_marry), "prev_people", len(prev_people))
            inf_dist_list = generate_couples(can_marry, prev_people, inf_only = True)

            people_ignored.clear()
            # If we are out of candidates, exit the loop.
            if not inf_dist_list:
                break
        # Uniformly choose a couple that will be married at infinite distance.
        # If our choice is invalid, re-select.
        counter = 0
        while True:
            counter += 1
            if counter == 10:
                break
            chosen_couple_index = np.random.choice(len(inf_dist_list))
            chosen_couple = inf_dist_list[chosen_couple_index]
            if (chosen_couple[0] not in people_ignored and chosen_couple[1] not in people_ignored): break

        people_ignored.update(list(chosen_couple))

        # Add the chosen couple to the set of unions and append distance -1 to the marriage_distances list
        unions.add(tuple(chosen_couple))
        if counter != 10:
            num_unions += 1
        marriage_distances.append(-1)

    if ctr_exec != 0:
        print(f"EXECUTED LOOP {100*ctr_update/ctr_exec}% of the time (times executed = {ctr_update})")

    print("num_unions", num_unions)

    # Get a set of people who are still single by taking the set union of those still single in this generation
    # with those still single in the last generation (we have been updating these sets along the way)
    can_marry = can_marry.difference(people_ignored)
    prev_people = prev_people.difference(people_ignored)
    still_single = can_marry.union(prev_people)

    # Bring in as many immigrants as needed to reach num_inf_couples_to_marry goal
    num_immigrants_goal = num_inf_couples_to_marry - num_unions
    num_immigrants = min(len(still_single), num_immigrants_goal)

    # Form immigrant couples by uniformly choosing num_immigrants candidates to marry an immigrant
    married_immigrant = np.random.choice(list(still_single), size=num_immigrants, replace=False)
    immigrants = list(range(next_immigrant, next_immigrant + num_immigrants))
    immigrant_couples = set(zip(married_immigrant, immigrants))

    # Update various data structures
    prev_gen_still_single = prev_people.difference(set(married_immigrant))
    unions.union(immigrant_couples)
    marriage_distances.extend([-1] * num_immigrants)
    didnt_marry.extend(list(can_marry.difference(married_immigrant)))
    print("didnt marry from current gen", len(didnt_marry))
    num_never_marry = len(prev_gen_still_single)

    return unions, num_immigrants, marriage_distances, immigrants, didnt_marry, num_never_marry

#%%
#@profile
def add_marriage_edges(people, prev_people, num_people, marriage_probs, prob_marry_immigrant, prob_marry, D, indices, original_marriage_dist, tol=0, eps=1e-7):
    """
    Forms both infinite and finite distance marriages in the current generation

    INVERTED: FORMS FINITE MARRAIGES FIRST
    PARAMETERS:
        people:  (list) of the current generation (IE those people elligible for
                marriage)
        prev_people: (list) of those in the previous generation who are yet unmarried.
        num_people: (int) the number of nodes/people currently in the graph
        finite_marriage_probs: (dictionary) keys are marriage distances, values
            are probabilities.  Note that this dictionary should only include
            entries for NON-inifite marriage distances, should have
            non-negaitive values which sum to 1, and should have a long right
            tail (IE lots of entries which map high (beyond what occurs in the
            example dataset) distances to zero(ish) probabilties)
        prob_marry_immigrant: (float) the probablility that a given node will marry
                a immigrant (herein a person from outside the genealogical network,
                without comon ancestor and therefore at distance infinity from the
                nodes in the list 'people') (formerly 'ncp')
        prob_marry: (float) the probability that a given node will marry another
                node in people
        D: ((len(people) x len(people)) numpy array) indexed array of distance
            between nodes in  people (siblings are distance 2)
        indices: (dictionary) maps node name (int) to index number in D (int)
    RETURNS:
        unions: (list of tuples of int) marriages formed.  Entries are of two
            types: 1) infite distance marraiges: one spouse is selected
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
            ancestor.  As before, a distnace of -1 indicates an infinite
            distance (IE one spouse immigrated into the community)
        wont_marry: (list) of nodes in people who did not marry, will attempt
            to marry someone from the next generation
    """

    print("prev_people", len(prev_people))
    print("people", len(people))

    finite_marriage_probs = {key: val for key, val in marriage_probs.items() if key > 0}

    if sum(finite_marriage_probs.values()) > 0:
        finite_marriage_probs = {key:val /sum(finite_marriage_probs.values()) for key, val in finite_marriage_probs.items()} # normalization
    desired_finite_distances = [distance for distance,prob in finite_marriage_probs.items() if prob > 0] # prob이 0 이상이라는 뜻은, 확률 조절후에 부족한 distance들을 넣은 리스트
    # if len(desired_finite_distances) == 0:
    #     print("NO DESIRED FINITE DISTANCES")
    #     print('finite_marriage_probs', finite_marriage_probs)

    minimum_permissible_distance = min([k for k in original_marriage_dist if k > -1])

    marriage_distances = []

    unions = set()




    people_set = set(people)  # for fast removal later

    # find the next 'name' to add to your set of people
    next_person = num_people + 1

    # number of non-connected people to add
    num_inf_couples_to_marry = round(prob_marry_immigrant * len(people)/2) #확률x초기인구값  # NOT SURE HERE NOW THAT WE MIX ADDING NEW AND NOT ADDING NEW IMMIGRANTS

    num_finite_couples_to_marry = round(prob_marry * len(people) / 2)

    # num_immigrants = np.random.binomial(len(people_set), prob_marry_immigrant)
    # num_finite_couples_to_marry = np.random.binomial(len(people_set) // 2, prob_marry)

    # divide the current generation into two camps, those who will marry among this and the previous generation
    # AND those who will marry next generation
    #결혼할 사람들을 뽑고 (절반정도만 뽑음.)
    will_marry = set(np.random.choice(list(people_set), size=len(people_set)//2, replace=False))

    wont_marry_until_next_time = [node for node in people_set if node not in will_marry]  # won't attempt to form marriages until next generation

    # add in singles from the previous generation 이전 세대의 사람들 절반과 현재세대의 절반
    people_set = will_marry | set(prev_people)
    # get number of people to marry
    # num_finite_couples_to_marry = round(len(people_set)*prob_marry/2)  # this line grabs a fraction of those who either stay single or marry at a finite difference but it doesn't account that part of that gen already married strangers
    # # num_finite_couples_to_marry = round((len(people_set) + len(marry_strangers)) * prob_marry / 2)
    # get all possible pairs of the still single nodes
    # rejecting possible parrings which have a common ancestor more recently
    # than allowed by finite_marriage_probs (IE this is where we account that siblings
    # don't marry in most cultures (but still can such as in the tikopia_1930
    # family network))
    possible_couples = [(man, woman) for man, woman in itertools.combinations(people_set, 2)]

    last_gen_couples = [(man, woman) for man, woman in itertools.combinations(prev_people, 2)]

    # we want combinations of couples from those in will_marry and prev_people (singles from last generation)
    # but not pairings where boths spouses are in the previous generation
    possible_couples = set(possible_couples) - set(last_gen_couples)  #전체(이전 세대와 현제 세대 사이에 결합할 수 있는 모든 결혼 가능성을 만들고 = set(possible_couples))
                                                                      #이전 세데끼리 결혼할 수 있는 경우의 수들= set(last_gen_couples)을 전체에서 제거하므로써
                                                                      #오직 이전 세대 한명과 현재 세대 한명이 결혼하거나 현재 세대끼리 결혼하는 경우의 수만 남겨둘 수 있음.

    # possible_couples = {(man, woman): D[indices[man]][indices[woman]]
    #                     for man, woman in itertools.combinations(people_set, 2)
    #                     if D[indices[man]][indices[woman]] >= min(finite_marriage_probs)}

    # 전체 경우의 수에서 미니멈 distance를 넘는 애들만 결혼 가능하게 할때 가능한 애들
    possible_finite_couples = {(man, woman): D[indices[man]][indices[woman]]
                        for man, woman in possible_couples
                        if D[indices[man]][indices[woman]] >= minimum_permissible_distance}


    #확률 조정이 끝난 후에 0이상인 애들은 부족한 애들이고, 부족한 애들 체워 넣을 수 있는 distance들의 리스트를 가진 애들
    preferred_couples = {couple:distance for couple, distance in possible_finite_couples.items() if (distance in set(original_marriage_dist) and distance in desired_finite_distances)}

    #얘네들은 그렇지 않지만, 차선책으로 사용할 수 있는 애들.
    other_couples = {couple:distance for couple, distance in possible_finite_couples.items() if couple not in preferred_couples}


    iter = 0

    # finite 커플들 연결 가능한 애들 먼저 함.
    while possible_finite_couples and iter < math.ceil(num_finite_couples_to_marry/2):
        # find the probabilities of all possible distances
        # must update after each marriage
        # change to a data structure suited to random draws:

        if preferred_couples:

            possible_finite_couples_array = np.array(list(preferred_couples.keys()))

            # dis_probs = np.array([finite_marriage_probs[d] for d in possible_couples.values()])

            dis_probs = np.array([finite_marriage_probs[d] if d in finite_marriage_probs else eps for d in preferred_couples.values()])
            print("dis_probs: ", type(dis_probs))
            print("What is d?")

            temp1 = dis_probs.copy()
            print("temp1: ", type(temp1))

            dis_probs[np.abs(dis_probs) < tol] = 0  # prevent "negative zeros"

            temp2 = dis_probs.copy()
            print("temp2: ", type(temp2))

            dis_probs = dis_probs / np.sum(dis_probs)  # normalize

            # choose couple based on relative probability of distances
            try:
                couple_index = np.random.choice(np.arange(len(preferred_couples)), p=dis_probs)
                print("couple_index: ", type(couple_index))

            except:
                print('marriage_probs', marriage_probs)
                print('desired distances', desired_finite_distances)
                print('finite_marriage_probs', finite_marriage_probs)
                print('preferred_couples', preferred_couples)
                print('temp1', temp1)
                print('temp2', temp2)
                print('dis_probs', dis_probs)
            couple = possible_finite_couples_array[couple_index]
        else:
            # print('len(preferred_couples): ', len(preferred_couples))
            # choose randomly from our candidate distances which are next closest
            # to our original distribution's support
            possible_finite_couples_array = np.array(list(other_couples.keys()))

            # find our least bad distance:
            least_bad_distance = min([d for d in other_couples.values() if d >= minimum_permissible_distance])
            print("least_bad_distance: ", type(least_bad_distance))

            # print('least_bad_distance: ', least_bad_distance, '\n')
            dis_probs = np.array([1 if d == least_bad_distance else 0 for d in other_couples.values()])
            print("dis_probs: ", type(dis_probs))

            dis_probs[np.abs(dis_probs) < tol] = 0  # prevent "negative zeros"
            dis_probs = dis_probs / np.sum(dis_probs)  # normalize   Should be equal where not zero

            # choose couple based on relative probability of distances
            couple_index = np.random.choice(np.arange(len(other_couples)), p=dis_probs)
            print("couple_index: ", type(couple_index))

            couple = possible_finite_couples_array[couple_index]
            print("couple: ", type(couple))


        unions.add(tuple(couple))
        # and save the distance of that couple
        marriage_distances += [int(D[indices[couple[0]], indices[couple[1]]])]


        # remove all possible pairings which included either of the now-married couple
        possible_finite_couples = {pair:possible_finite_couples[pair]
                                for pair in possible_finite_couples
                                    if ((pair[0] != couple[0])
                                    and (pair[1] != couple[0])
                                    and (pair[0] != couple[1])
                                    and (pair[1] != couple[1]))}
        print("What are pair and possible?")

        preferred_couples = {couple:distance for couple, distance in possible_finite_couples.items() if (distance in set(original_marriage_dist) and distance in desired_finite_distances)}
        other_couples = {couple:distance for couple, distance in possible_finite_couples.items() if couple not in preferred_couples}

        iter += 1  # number of finite union edges added
        stay_single_forever = set([node[0] for node in possible_couples] + [node[1] for node in possible_couples])
        print("stay_single_forever: ", type(stay_single_forever))

    if iter == 0:
        # IE you never entered the while loop above
        stay_single_forever = will_marry | set(prev_people)   #유한거리 결혼을 못했다면, 일단 싱글에 넣어놓고, 이 다음에 싱글애들을 대상으로 무한 거리 결혼을 결정하는듯.

    # 여기에서 가능한 결혼 옵션을 모두 만듬. Cartesian Product으로 다 만듬. 거의 10만개의 가능성이 나옴.
    possible_inf_couples = [(man, woman) for man, woman in itertools.combinations(stay_single_forever, 2)]
    print("possible_inf_couples: ", type(possible_inf_couples))

    # 전체 가능성 옵션들 중에서, 거리가 -1인 얘들만 결혼에 참여시킴.
    possible_inf_couples = {(man, woman): D[indices[man]][indices[woman]]
                            for man, woman in possible_inf_couples
                            if D[indices[man]][indices[woman]] == -1}

    iter = 0
    # -1 결혼을 만드는 작업.
    while possible_inf_couples and iter < math.ceil(num_inf_couples_to_marry/2):
        possible_inf_couples_array = np.array(list(possible_inf_couples.keys()))
        #print("possible_inf_couples_array: ", type(possible_inf_couples_array))


        couple_index = np.random.choice(np.arange(len(possible_inf_couples)))  # draw uniformly
        couple = possible_inf_couples_array[couple_index]

        unions.add(tuple(couple))
        marriage_distances += [int(D[indices[couple[0]]][indices[couple[1]]])]

        # remove all possible pairings which include either spouse in couple
        possible_inf_couples = {pair:possible_inf_couples[pair]
                                for pair in possible_inf_couples
                                    if ((pair[0] != couple[0])
                                    and (pair[1] != couple[0])
                                    and (pair[0] != couple[1])
                                    and (pair[1] != couple[1]))}

        # iter += 2  # nodes in graph that married immigrants
        iter += 1  #if we divide by 2 when defining num_inf_couples_to_marry

        stay_single_forever = set([node[0] for node in possible_inf_couples] + [node[1] for node in possible_inf_couples])

    # find how many infinite-distance pairings are possible in the people already introduced
    # and how many new immigrants you will need to add to the graph to
    # marry off the immigrants at random to nodes in the current generation
    num_immigrants = num_inf_couples_to_marry - iter

    # num_immigrants = max(num_immigrants, 0)  # if we already added enough inf couples possible than needed, don't introduce any more possibilities
    num_immigrants = min(len(stay_single_forever), num_immigrants)  # dont want to add people who wont get married this time

    immigrants = [k for k in range(next_person, next_person + num_immigrants)]

    marry_strangers = np.random.choice(list(stay_single_forever), size=num_immigrants, replace=False)

    stay_single_forever -= set(marry_strangers)

    unions = unions | {(spouse, immigrant) for spouse, immigrant in zip(marry_strangers, immigrants)}
    #print("What are spouse and immigrant?")

    # remove the married people from the pool #2ndManifesto
    # people_set = people_set - set(marry_strangers)
    # and record the (infinite) distances of each marriage
    marriage_distances += [-1 for k in range(num_immigrants)]

    # now that you've married off some fraction of the generation to new nodes (ie to immigrants)

    return unions, num_immigrants, marriage_distances, immigrants, wont_marry_until_next_time, len(stay_single_forever)


#%%
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
def update_distances(D, n, unions, families, indices, people_retained):
    """
    Build a distance matrix that keeps track of how far away each node is from
    each other. Need to update distances after new nodes added to graph (i.e.,
    after adding children)
    PARAMETERS:
        D (array): "old" matrix of distances, previous generation
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
    # initialize new matrix
    num_children = len([child for fam in families for child in fam])
    num_people_retained = len(people_retained)

    D1 = np.zeros((num_children + num_people_retained,
                   num_children + num_people_retained))

    new_indices = {person:k for k, person in enumerate(people_retained + [child for fam in families for child in fam])}

    # check_indices(new_indices)
    if num_people_retained > 0:
        # the upper num_retained_people x num_retained_people block of D1 is just a slice from D
        # this builds out the "second quadrant" of the D1 matrix
        D1[new_indices[people_retained[0]]:new_indices[people_retained[-1]]+1,
           new_indices[people_retained[0]]:new_indices[people_retained[-1]]+1] = D[[indices[k] for k in people_retained]][:, [indices[k] for k in people_retained]]

        # now build out the distances from people_retained to the new generation (descendants of people_retained's cousins)
        # this builds out the "first and third" quadrants of the D1 matrix
        for rp, fam in product(people_retained, zip(unions, families)):
            # find minimum distance between the retained person and a representative child in the family
            father = fam[0][0]
            mother = fam[0][1]
            children = fam[1]  # all children in family will have same distance from retained person rp
            if len(children) == 0:
                # ie the union, family pair has no children listed,
                # the end of a line
                continue

            possible_distances = D[indices[rp], [indices[father], indices[mother]]]
            if (possible_distances == -1).all():
               d = -1
            else:
                possible_distances = possible_distances[possible_distances > -1]
                d = np.min(possible_distances) + 1  # account for the additional edge between father/mother and ch
            D1[new_indices[rp], [new_indices[ch] for ch in children]] = d
            D1[[new_indices[ch] for ch in children], new_indices[rp]] = d


    # compute new distances, between new generation
    # this builds out the "fourth quadrant" of the D1 matrix
    unions = list(unions)
    for u, union in enumerate(unions):
        u_children = families[u]

        for other in unions[u:][1:]:
            o_children = families[unions.index(other)]

            # find all possible distances from union to other
            d1 = D[indices[union[0]]][indices[other[0]]]
            d2 = D[indices[union[1]]][indices[other[0]]]
            d3 = D[indices[union[0]]][indices[other[1]]]
            d4 = D[indices[union[1]]][indices[other[1]]]

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
                    D1[new_indices[oc]][new_indices[uc]] = d

        for c, ch in enumerate(u_children):
            for sibling in u_children[c:][1:]:
                D1[new_indices[ch]][new_indices[sibling]] = 2
                D1[new_indices[sibling]][new_indices[ch]] = 2

    return D1, new_indices


#%%
# TODO: what do we actually want this to return?
from pandas.core.ops.mask_ops import libmissing

def human_family_network_variant(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, children_dist, name, when_to_stop=np.inf, num_gens=np.inf, save=True, out_dir='output', eps=0):
    """
    This is the main driver for implementing the Variant Target Model (see
    Chapter 5 in Kolton's thesis).
    PARAMETERS:
        num_people (int): number of people (nodes) to include in initial
            generation
        marriage_dists: (list of int) one entry per marriage indicating how many
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

    num_original_people = num_people
    total_num_single = 0
    dies_out = False

    all_marriage_edges = []
    all_marriage_distances = []
    all_temp_marriage_edges = []
    all_temp_marriage_distances = []
    all_children_per_couple = []
    temp = 0
    temp2 = 0

    # 타겟 데이터에서 결혼 거리 가져옴.
    marriage_dist_array = np.array(marriage_dist)
    # 타겟 데이터에서 유한 거리 데이터만 가져옴.
    finite_only_marriage_dist = marriage_dist_array[marriage_dist_array != -1]

    G = nx.DiGraph()

    D = np.ones((num_people, num_people)) * -1  # starting off, everyone is infinite distance away
    np.fill_diagonal(D, 0)  # everyone is 0 away from themselves, also weirdly done inplace

    indices = {node + 1:k for k, node in enumerate(range(num_people))}  # name:index
    generation_of_people = list(indices.keys())

    # Define the initial graph components
    # components = {k:list(k) for k in generation_of_people}

    # explicitly add our first generation of nodes (otherwise we will fail to
    # add those who do not marry into our graph).  All future generations are
    # connected either through marriage or through parent-child arcs
    G.add_nodes_from(generation_of_people, layer=0, immigrant=True, parents=0)


    # get probabilities of possible finite distances to use in marriage function
    # and normalize it
    marriage_probs = get_marriage_probabilities(marriage_dist, when_to_stop, eps=eps)  # hand in the ORIGINAL MARRIAGE DISTRIBUTION
    # maybe don't normalize so that you will (approximately) retain the prob_inf + prob_finite + prob_single = 1
    marriage_probs = {key:value/sum(marriage_probs.values()) for key, value in zip(marriage_probs.keys(), marriage_probs.values())}
    lim = len(marriage_probs)




    # ditto for the child distribution
    child_probs = get_child_probabilities(children_dist)
    # ??? make probabilities non-negative (some entries are effectively zero, but negative)
    child_probs = {key:value if value > 0 else 0 for key, value in zip(child_probs.keys(), child_probs.values()) }
    child_probs = {key:value/sum(child_probs.values()) for key, value in zip(child_probs.keys(), child_probs.values())}

    # grow the network until there are the at least as many nodes (not counting
    # those created to impose the distances on generation 0, but counting those
    # in generation 0) as when_to_stop
#    num_setup_people = num_people - num_original_people

    prev_generation_still_single = []
    current_prob_inf = np.inf

    summary_statistics = []  # will hold ordered tuples of integers (# total people in graph,  # immigrants, # num_children, # num marriages, prob_inf_marriage, prob_finite_marriage, prob_inf_marriage(eligible_only), prob_finite_marriage(elegible_only))
    i = 1
    tem_num_people = 0
#    while (num_people - num_setup_people < when_to_stop) & (i < num_gens):
    while (tem_num_people < when_to_stop) & (i < num_gens):

        # create unions between nodes to create next generation
        # unions, no_unions, all_unions, n, m, infdis, indices = add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis, indices)
        if len(generation_of_people) == 0:
            dies_out = True
            break

        # update your current finite marriage probabilities to favor those which are yet underrepresented
        # if i > 1 and len(set(all_marriage_distances)) > 2:
        if (i > round(lim/2)+1) and (i <= round(lim/2*3/2)+1):
            new_marriage_probs = get_difference_in_probabilities(marriage_probs, all_temp_marriage_distances, num_people, name, i, output_path, plot=save, eps=eps)  #all_temp_marriage_distances 이게 아니라 all_marriage_distances 이거였음
        elif i > round(lim/2*3/2)+1:
            new_marriage_probs = get_difference_in_probabilities(marriage_probs, all_marriage_distances, num_people - num_setup_people, name, i, output_path, plot=save, eps=eps)
        else:
            # if this is the first generation beyond the initial set up OR
            # if you don't yet have more than one unique distance in your list
            # of marriage edges, then just default to the unaltered finite-
            # distance marriage distribution



            temp_marriage_probs =  {key:value/sum(marriage_probs.values()) for key, value in zip(marriage_probs.keys(), marriage_probs.values())}
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

            print(new_marriage_probs)
            print("Sum: ", sum(new_marriage_probs.values()))



        new_prob_inf_marriage = new_marriage_probs[-1]
        new_prob_finite_marriage = sum(new_marriage_probs.values()) - new_marriage_probs[-1] - new_marriage_probs[0]
        unions, num_immigrants, marriage_distances, immigrants, prev_generation_still_single, stay_single_forever = add_marriage_edges_random(graph=G,
                                                                                                                                              people=generation_of_people,
                                                                                                                                              prev_people=prev_generation_still_single,
                                                                                                                                              num_people=num_people,
                                                                                                                                              marriage_probs=new_marriage_probs,
                                                                                                                                              prob_marry_immigrant=new_prob_inf_marriage,
                                                                                                                                              prob_marry=new_prob_finite_marriage,
                                                                                                                                              D=D,
                                                                                                                                              indices=indices,
                                                                                                                                              original_marriage_dist=marriage_dist,
                                                                                                                                              eps=eps)
        # unions는 add_marriage_edge에서 만들어진 새로운 결혼 nodes들
        # D는 add_marriage_edge에서 직접 연결시키지 않고, 이미 있는 그래프에서 nodes들을 불러와 가능성을 찾는 역할을 함.
        # 아 D는 그래프가 아니라 행렬이였구나!
        total_num_single += stay_single_forever
        G.add_nodes_from(immigrants, layer=i-1, immigrant=True, parents=0)
        G.add_edges_from(unions, Relationship="Marriage")        #결혼한 노드들을 연결시킴.
        if i > round(lim/2*3/2):
            all_marriage_edges += list(unions)
            all_marriage_distances += marriage_distances
        else :
            num_setup_people = num_people
            all_temp_marriage_edges += list(unions)
            all_temp_marriage_distances += marriage_distances


        # D에 이민온 사람들을 행렬 D에 추가하는 방법
        for j in range(num_immigrants):
            # add non-connected people to distance matrix
            r = np.ones((1, len(indices) + 1 + j)) * -1  # -1 is infinite distance
            r[0, -1] = 0  # distance to self is 0
            c = np.ones((len(indices) + j, 1)) * -1  # -1 is infinite distance
            D = np.hstack((D, c))
            D = np.vstack((D, r))

        max_ind = max(indices.values())
        indices = indices | {immigrant + num_people + 1:ind for ind, immigrant in zip(range(max_ind + 1, max_ind+1+num_immigrants), range(num_immigrants))}  # +1 since we begin counting people at 1, 2, ... not at 0, 1, ...

        if i > round(lim/2*3/2)+1:
            stats = [num_people - num_setup_people, num_immigrants]
            #print("IF num people: ", num_people,"num set people = ", num_setup_people)
        num_people += num_immigrants
        current_people = num_people
        #print("num_people: ", num_people)

        # add children to each marriage
        child_edges, families, num_people, num_children_per_couple = add_children_edges(unions, num_people, child_probs)
        added_children = num_people - current_people

        # update distances between nodes
        D, indices = update_distances(D, num_people, unions, families, indices, prev_generation_still_single)

        generation_of_people = [key for key in indices.keys() if key not in prev_generation_still_single]  # only grab the new people
        G.add_nodes_from(generation_of_people, layer=i, immigrant=False, parents=2)
        G.add_edges_from(child_edges, Relationship='Parent-Child')
        all_children_per_couple += list(num_children_per_couple)


        #print("all_temp_marriage_distances= ", len(all_temp_marriage_distances)," increased= ", len(all_temp_marriage_distances)-temp )
        temp = len(all_temp_marriage_distances)
        if i > round(lim/2*3/2)+1:
            tem_num_people += num_immigrants + added_children
            # stats.append(len(generation_of_people))
            stats.append(sum(num_children_per_couple))
            stats.append(len(unions))
            stat_prob_marry = len(all_marriage_distances) * 2 / len(G)
            print("len(all_marriage_distances)= ",len(all_marriage_distances), " increased= ", len(all_marriage_distances)-temp2)
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
        #print("i= ",i)
    # build a single component out of the graph's many components
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
                G_connected_parent_leaf_layers.remove('setup')  #'setup'이라고 표시된 leaf nodes들을 없애라는 코드...

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

        df = pd.DataFrame(data=summary_statistics, columns=['num_people (excluding initial setup)', 'num_immigrants', 'num_children', 'num_marriages', 'prob_inf_marriage', 'prob_finite_marriage', 'prob_inf_marriage(eligible_only)', 'prob_finite_marriage(eligible_only)', 'num_connected_components'])
        df.index.name='generation'
        df.to_csv(os.path.join(output_path, str(name)+'_summary_statistics.csv'))
        # Gname = "{}/{}_G_withsetup.gpickle".format(output_path, name)   # save graph
        #nx.write_gpickle(G, Gname)
        Gname = "{}/{}_G_withsetup.pkl".format(output_path, name)   # save graph
        with open(Gname, 'wb') as out:
            pickle.dump(G_connected, out)
        # now save a reduced version of the graph, without the setup people

        Gname = "{}/{}_G_no_setup.pkl".format(output_path, name)   # save graph
        with open(Gname, 'wb') as out:
            pickle.dump(G, out)

        Uname = "{}/{}_marriage_edges".format(output_path, name) + '.pkl'   # save unions
        with open(Uname, 'wb') as fup:
            pickle.dump(all_marriage_edges, fup)
        Dname = "{}/{}_marriage_distances".format(output_path, name) +'.pkl' # save marriage distances
        with open(Dname, 'wb') as myfile:
            pickle.dump(all_marriage_distances, myfile)
        Cname = "{}/{}_children_per_couple".format(output_path, name) + '.pkl'  # save children
        with open(Cname, 'wb') as fcp:
            pickle.dump(all_children_per_couple, fcp)

        paj = format_as_pajek(G, name)
        with open('{}/model-{}-oregraph.paj'.format(output_path, name), 'w') as o:
            o.writelines(paj)

        return G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, output_path

    else:
        return G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, None

#%%
"""
below is example code to run the model
"""

name = 'torshan'
num_people = 1000
eps = 0
marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
G, G_connected, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies_out, output_path = human_family_network_variant(num_people, marriage_dist, prob_finite_marriage, prob_inf_marriage, child_dist, name, save=True, when_to_stop=size_goal, eps=eps)

def find_start_size(name, out_directory='start_size', filename='start_size', max_iters=100, dies_out_threshold=5,  verbose=False, save_start_sizes=True, random_start=True, return_counter=False): # n = number of initial nodes
    counter = 0

    filename = name + '_' + filename
    marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)
    greatest_lower_bound = 2
    least_upper_bound = size_goal

    if random_start:
        num_people = np.random.randint(greatest_lower_bound, size_goal)
    else:
        num_people = size_goal//2
    dies_out = 0 # counter for the number of times the model dies out

    start_sizes = [num_people]
    while dies_out != dies_out_threshold: # while the number of times the model dies out is not equal to the threshold of dying:

        for i in range(max_iters):
            counter += 1
            G, G_reduced, all_marriage_edges, all_marriage_distances, all_children_per_couple, dies, output_path = human_family_network_variant(num_people,
                                                                                                                marriage_dist,
                                                                                                                prob_finite_marriage,
                                                                                                                prob_inf_marriage,
                                                                                                                child_dist,
                                                                                                                name,
                                                                                                                when_to_stop=size_goal,
                                                                                                                save=False)
            if dies:
                dies_out += 1
            if dies_out > dies_out_threshold:
                break

        if greatest_lower_bound >= least_upper_bound - 1:
            # IE the ideal lies between these two integers
            # so return the larger
            num_people = least_upper_bound
            break
        elif dies_out == dies_out_threshold:
            break
        elif dies_out > dies_out_threshold:  # we want to increase num_people
            greatest_lower_bound = num_people  # current iteration died out too frequently.  Won't need to search below this point again.
            num_people = (num_people + least_upper_bound) // 2 # midpoint between num_people and size_goal
            dies_out = 0

        elif dies_out < dies_out_threshold: # we want to decrease num_people
            least_upper_bound = num_people  # current iteration died out too infrequently.  Won't need to search above this point again
            num_people = (greatest_lower_bound + num_people) // 2 # midpoint between 2 and num_people
            dies_out = 0

        if verbose:
            print('greatest_lower_bound: ', greatest_lower_bound)
            print('least_upper_bound: ', least_upper_bound)
            print('starting population: ', num_people)
        start_sizes.append(num_people)


    if save_start_sizes:
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        filename = find_file_version_number(out_directory, filename, extension='.txt')
        # save a text file (one integer per line)
        with open(os.path.join(out_directory, filename +'.txt'), 'w') as outfile:
            outfile.writelines([str(k) + '\n' for k in start_sizes])
        # save the actual object
        with open(os.path.join(out_directory, filename + '.pkl'), 'wb') as outfile:
            pickle.dump(start_sizes, outfile)

    if return_counter:
        return start_sizes, counter
    else:
        return start_sizes
#%%


def repeatedly_call_start_size(name, out_directory='start_size', iters=5, max_iters=100, dies_out_threshold=5,  verbose=False, save_start_sizes=True, save_individual_start_sizes=False, random_start=True, show_plot=False):
    #find out directory.  Every iteration in this function call will output to the same file
    out_dir = makeOutputDirectory(out_directory, name)

    if save_start_sizes:
        # create the folder, text file to which EACH start_size list will be appended
        filename = name + '_' + 'start_size'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(os.path.join(out_dir, filename + '.txt')):
            with open(os.path.join(out_dir, filename + '.txt'), 'w'):
                pass

    to_plot = []
    for i in range(iters):
        start_sizes = find_start_size(name,
                                      out_dir,
                                      max_iters=max_iters,
                                      dies_out_threshold=dies_out_threshold,
                                      verbose=verbose,
                                      save_start_sizes=save_individual_start_sizes,
                                      random_start=random_start)
        to_plot.append(start_sizes)

        if save_start_sizes:
            # save a text file (one line per run of find_start_size())
            with open(os.path.join(out_dir, filename +'.txt'), 'a') as outfile:
                outfile.write(str(start_sizes))
                outfile.write('\n')

        if save_start_sizes:
            # save the unaltered (entries will be of differing lengths) set of start sizes
            # as a pickle object for later use
            with open(os.path.join(out_dir, filename + '.pkl'), 'wb') as outfile:
                pickle.dump(to_plot, outfile)

    # prep each entry of to_plot.  Not every iteration will have the same num
    # of entries.  Just repeat the last entry as necessary
    length = np.max([len(k) for k in to_plot])
    for start_sizes in to_plot:
        while len(start_sizes) < length:
            start_sizes.append(start_sizes[-1])

    fig = plt.figure(figsize=(12,9), dpi=300)
    for k in range(len(to_plot) -1):
        plt.plot(to_plot[k], color='k', linewidth=0.5, alpha=0.65)

    # plot the last one with a label
    plt.plot(to_plot[-1], color='k', linewidth=0.5, alpha=0.65, label='individual run')
    # make the plot display in text the final average starting value
    avg_run = np.mean(to_plot, axis=0)
    plt.text(length - 1.25, avg_run[-1]+3, str(round(avg_run[-1])), fontsize=16)
    # now plot the average, bolded
    plt.plot(avg_run, color='k', linewidth=7, alpha=0.8, label='mean')
    plt.xticks([k for k in range(length)], labels=[k for k in range(1, length+1)], fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(name + ' initial population search', fontsize=24)
    plt.legend()
    plt.ylabel('starting population', fontsize=16)
    plt.xlabel('iterations', fontsize=16)
    plt.savefig(os.path.join(out_dir, name + '_starting_size_graph.png'), format='png')
    if show_plot:
        plt.show()

    return avg_run



def plot_gen(g0, p_union, C, gen_num, max_imm):
    """Generate plots for the expected network populations each generation,
    given the following parameters:
    (int) g0: size of the initial generation
    (float) p_union: probability that a person will get married
    (float) C: the expected number of children for each couple
    (int) gen_num: number of generations to compute
    (int) max_imm: the greatest number of immigrants that should be added to form infinite-distance unions each cycle (generated randomly)
    """

    # define the probability of a person staying single
    p_0 = 1 - p_union
    pop = []
    pop2 = []
    # generate list of numbers of immigrants to arrive each generation, up to the gen_numth generation
    imm = [random.randint(0,max_imm) for i in range(gen_num)]

    # the first generation size is a special case
    g1 = (1/4)*g0*p_union*C

    pop.append(g0)
    pop.append(g1)
    pop2.append(g0)
    pop2.append(g1)

    i = 2
    while i < gen_num:
        # recursively identify the number of people in each successive generation
        # using the formula we came up with
        next_gen = (1/4)*(pop[i-1]+(1+p_0)*pop[i-2])*p_union*C
        pop.append(next_gen)

        # do the same in the case that we want to consider immigrants coming in
        next_gen_imm = (1/4)*(pop2[i-1]+(1+p_0)*pop2[i-2]+2*imm[i-2])*p_union*C
        pop2.append(next_gen_imm)
        i += 1

    pop = np.array(pop)
    pop2 = np.array(pop2)
    x = range(gen_num)

    # graph the data sets in the left subplot
    plt.subplot(121)
    plt.plot(x,pop,label="no immigrants")
    plt.plot(x,pop2,label="random immigrants each generation")

    # fit the data to y = Ae^(tx)
    # so log y = log(A) + t*log(x)
    t, log_A = np.polyfit(x, np.log(pop), 1)
    A = np.exp(log_A)

    # y = Ae^(tx) = A*r^x
    print(f"r value without immigrants: {np.exp(t)}")

    # Calculate y = A*e^(tx) for each x in {1,2,...,gen_num}
    exp_points = [A*np.exp(t*s) for s in x]
    plt.plot(x,exp_points,label="best fit curve without immigrants")

    # compute the same fit for the data with immigrants
    t, log_A = np.polyfit(x, np.log(pop2), 1)
    A = np.exp(log_A)

    # y = Ae^(tx) = A*r^x
    print(f"r value with immigrants: {np.exp(t)}")
    exp_points2 = [A*np.exp(t*s) for s in x]
    plt.plot(x,exp_points2,label="best fit curve with immigrants")

    # Graph settings
    plt.xlabel("Generation number")
    plt.ylabel("Population count")
    plt.legend()

    # Also graph the total size of the network for each generation
    # (at generation n, sum up the first n elements of the population lists)
    plt.subplot(122)
    pop_total = [sum(pop[:i+1]) for i in range(len(pop))]
    plt.plot(x,pop_total,label="size of network without immigrants")

    pop_total = [sum(pop2[:i+1]) for i in range(len(pop2))]
    plt.plot(x,pop_total,label="size of network with immigrants")

    plt.xlabel("Generation number")
    plt.ylabel("Population count")
    plt.legend()

    plt.show()


def find_r(name, g0, num_gen, max_imm):
    """ Generates a generation plot for a given network name using an arbitrary number of immigrants for each generation
     and another plot without immigrants.
     Paramaters:
     name: the name of a network
     g0: the size of initial generation 0
     num_gen: the nubmer of generations we want to plot
     max_imm: the maximum number of immigrants for each generation
    """

    # Gathering data from the network
    marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, size_goal = get_graph_stats(name)

    # Find the expected number of children in the network
    chil_dic = get_child_probabilities(child_dist)
    expected_children = sum([key * chil_dic[key] for key in chil_dic.keys()])

    # Find the probability to get married in the network
    punion = prob_inf_marriage + prob_finite_marriage

    # Generate 'plot_gen' function
    print(f"{name} netwrok has following,")
    plot_gen(g0, punion, expected_children, num_gen, max_imm)



#%%

if __name__=='__main__':

    #name = "tikopia_1930"
    #name = "arawete"
    name = "kelkummer"

    find_r(name,20000,25,5)