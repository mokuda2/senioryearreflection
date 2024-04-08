"""
This version of the Variant Target Model results in a graph with (potentially)
many components (returned as G by human_family_network()) and a version with a
single weakly connected component (returned as G_connected by
human_family_network()) (see Chapter 5 in Kolton's thesis))

The main guts of the model are in the make_model() method.  There is
an example of how to run the model is on the bottom

This version new an updated to implement a class structure (in hopes to make updating code more efficient)
"""

# Missing functions from the original
    # find_file_version_number
    # find_start_size
    # update_distance
    # repeatedly_call_start_size
    # plot_gen
    # find_r

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
import time

import regex as re
import itertools
from itertools import combinations, product
from operator import itemgetter

import ast
import pickle
import os
from pandas.core.ops.mask_ops import libmissing
from write_model_to_pajek_ore_graph import format_as_pajek
from setuptools import setup
from Cython.Build import cythonize


class HumanFamily:
    """
    PARAMETERS:
        name: (str) the name of the kinsources data set (see below for format)
    METHODS (To call on the object):
        HumanFamily(name): Initializes the model, recording data from the model "name"
        make_model(init_gen_size): Makes the model, given an initial size for the first generation
            if save=True, the distributions of distance to unions are stored in graphs in ouptut folder
            if method='BFS', the model will use the BFS method to form marriage unions
        build_single_component(): Builds a single component graph from the model
            if save=True, the distributions of distance to unions are stored in graphs in ouptut folder
    ATTRIBUTES:
        Initialized when the object is created:
            save (bool): should you save pickled versions of networkx graphs, model
                union edges, and distance to union and child distributions, as well
                as png graphs of distance to union updates each generation as well
                as an xlsx file summarizing model growth? should you print debug info?
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
            num_people_orig_graph: (int) total number of nodes in the real life data set.
        Created When the make_model() function is run
            num_people_first_generation (int): the number of people put in the first generation
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
            out_dir (str): destination directory for save output.  Each run will
                create an out_dir/{name}_{int}/ file structure, counting version
                number automatically
            dies_out (bool):  False indicates that the model exited successfully---
                generations 0 to L (inclusive) contain more than when_to_stop nodes
                in total OR L >= num_gens.  True else.
            summary_statistics (tuple): Ordered tuples of integers 
                (# total people in graph,  # immigrants, # num_children, # num marriages, 
                self.prob_inf_marriage, self.prob_finite_marriage, self.prob_inf_marriage(eligible_only), 
                self.prob_finite_marriage(elegible_only))
            method (str): Used to tell make_model which method it should use when forming marraige unions
                'NetworkX' - Use Networkx BFS to find a mate of specified distance
                BFS - Do a BFS up the tree to determine which nodes to avoid while traversing done
            fixed (boolean): Used to determine if we use a fixed probability distribution or not
            generation_ranges: (list) Stores the largest person number in each generation 
                index: generation number
                value: total_num_people in generations 0-index
                Note: The initial generation of infinite distance people is gen 0
            num_people_setup (int): The number of people chosen to be in the first generation
            num_people (int): The number of people in the entire model
            tol (float)  # Stopping criteria 
                (the algorithm will stop when the difference between the erros is less than tol)
            eps (int)   # Allowed error
    """
    # Initialization function is same as get_graph_stats
    def __init__(self, name, save=True, distance_path='./Kolton_distances/', child_number_path='./ChildrenNumber/'):
        """
        Gets the statistics of a specified kinsources dataset
        PARAMETERS:
            name: (str) the name of the kinsources data set
            distance_path: (str) the filepath to the directory containing the saved
                text files containing the distance to marriage distributions (the output
                of timing_kolton_distance_algorithm.py)
            child_number_path: (str) the filepath to the directory containing the
                saved text files containing the children per couple distributions
        ATTRIBUTES (That it initializes):
            name: (str) the name of the kinsources data set
            save (bool): should you save pickled versions of networkx graphs, model
                union edges, and distance to union and child distributions, as well
                as png graphs of distance to union updates each generation as well
                as an xlsx file summarizing model growth? should you print debug info?
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
                NOTE: prob_finite_marriage + prob_inf_marriage + prob_single = 1
            children_dist: (list of int) one entry per pair of parents, indicating
                how many child edges each parent in the couple share
            num_people_orig_graph: (int) total number of nodes in the real life data set.
        """
        if save: print(f"Gathering Data From {name}.txt")
        with open(distance_path + '{}.txt'.format(name)) as infile:
            self.marriage_dists, num_inf_marriages, fraction_inf_marriage = [ast.literal_eval(k.strip()) for k in infile.readlines()]

        # number of children data of chosen network
        with open(child_number_path + '{}_children.txt'.format(name)) as f:
            nx_child = f.readline()

        self.name = name
        self.num_people_orig_graph = self.get_num_people()   # Get the number of people in the original graph
        if save: print(f"This Data Set Has {self.num_people_orig_graph} people")

        self.save = save
        self.children_dist = ast.literal_eval(nx_child)
        self.num_marriages = len(self.marriage_dists)
        prob_marriage = self.num_marriages * 2 / self.num_people_orig_graph  # *2 since 2 spouses per marriage
        self.prob_inf_marriage = prob_marriage * fraction_inf_marriage
        self.prob_finite_marriage = prob_marriage - self.prob_inf_marriage

    def reset(self):
        """
        Defaults back to status of the object before the make_model() function was called
        Used if you want to run a model on the same data set over and over again without reinitalizing
        """
        self.out_dir = None
        self.fixed = None
        self.eps = None
        self.num_people_setup = None
        self.method = None
        self.generation_ranges = None
        self.G = None
        self.summary_statistics = None
        self.all_marriage_edges = None
        self.all_marriage_distances = None
        self.all_children_per_couple = None
        self.dies_out = None
        self.num_people = None
        self.error = None


    # Work with directories (next 2 functions)
    def get_graph_path(self, path='./Original_Sources/'):
        """
        PARAMETERS:
            name: (str) the name of the kinsources data set (see below for format)
        RETURNS:
            path: (str) path to directory prepended to the full name of specified
                kinsources file
        """
        return path + 'kinsources-' + self.name + '-oregraph.paj'

    def makeOutputDirectory(self):
        """
        Make an output directory to keep things cleaner

        Returns a full output path to the new directory
        """
        ver = 1
        output_dir = os.path.join(self.out_dir, self.name + '_')
        while os.path.exists(output_dir + str(ver)):
            ver += 1
        output_dir += str(ver)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir


    # Get graph stats
    def get_num_people(self):
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
        path_to_graph = self.get_graph_path()
        # open and read graph file
        with open(path_to_graph, 'r') as file:
            contents = file.readlines()

        num_people = contents[2]
        num_people_pattern = re.compile("[0-9]+")
        num_people = int(num_people_pattern.findall(num_people)[0])

        return num_people

    # Get distributions (next 3 functions)
    def get_marriage_probabilities(self, data, num_people):
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
        # finite_probs = {key: val / denom * prob_finite_marriage + self.eps for key, val in finite_probs.items()}
        finite_probs = {key: val + self.eps for key, val in finite_probs.items()}
        prob_single = 1 - prob_inf_marriage - prob_finite_marriage

        probs = {}
        probs[-1] = prob_inf_marriage + self.eps
        probs[0] = prob_single + self.eps
        probs = probs | finite_probs

        # return only those keys with non zero values
        probs = {key:val for key, val in probs.items() if val != 0}
        return probs
    
    def get_child_probabilities(self, data):
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
        RETURNS:
            probs (dictionary): keys are the entries of data and successive values,
                too (we lengthen the right tail of the distribution).
        """
        data = np.array(data)
        data = data[data > -1]  # only use non-negative number of children per household
        domain = np.arange(0, max(data) + 1, 1)

        probs = {x: sum(data==x)/len(data) for x in domain}
        probs = {key:val for key, val in probs.items() if val != 0} # return only those keys with non zero values

        return probs
    
    def get_difference_in_probabilities(self, target_probs, current, num_people, gen_num, plot=False, fixed=False):
        """
        This method accepts both the target marriage distribution AND the
        current-state model marriage distribution.  It will subtract the
        current-state from the target probabilites, flooring at some positive
        epsilon.  The returned probabiltiy distribution will then favor any
        marriages of those distances which have not yet been drawn in proportion
        with the target distribution's frequency for that distance

        This method also returns the total square error between the two distributions

        PARAMETERS:
            target_probs (dictionary): keys are marriage distances, values are
                probabilities.  This is the result of
                get_probabilities(marriage_dist) (called finite_marriage_probs
                below). This should already be normalized.
            current (list): list of marriage distances currently represented in the
                graph.
            num_people (int): the current number of people in the model
            gen_num (int): the generation number
            plot (boolean): if true, it will plot the distant to union distributions
            fixed (boolean): if it is True, the function returns fixed distribution
                if it is False, the function returns balanced distribution.
        Returns:
            adjusted_probs (dict): probability distribution
            total_error (int): the total error between the distributions
        """
        current_probs= self.get_marriage_probabilities(current, num_people)
        current_probs = {key:value/sum(current_probs.values()) for key, value in zip(current_probs.keys(), current_probs.values())} # normalize
        # need every key that occurs in target to also occur in current_probs
        current_probs = current_probs | {key : 0 for key in target_probs if key not in current_probs}

        if fixed :
            # Fix the probability distribution of marriage distance to the target distribution
            adjusted_probs = target_probs
        else:
            # Allow the current probability distribution of marriage distance be changed every generation
            adjusted_probs = {key: target_probs[key] - current_probs[key] if target_probs[key] - current_probs[key] >= 0  else self.eps for key in target_probs.keys()}
            # normalize
            adjusted_probs = {key:value/sum(adjusted_probs.values()) for key, value in zip(adjusted_probs.keys(), adjusted_probs.values()) if value != 0} # normalize
            adjusted_probs = adjusted_probs | {key:0 for key in target_probs if key not in adjusted_probs}

        if plot:
            self.graph_current_distributions(target_probs, current_probs, adjusted_probs, gen_num)

        # Error calculation between target distribution and current distribution
        total_error = []
        for k, v in current_probs.items():
            total_error.append((abs(target_probs[k] - v) * 100) ** 2)

        return adjusted_probs, sum(total_error)
    

    # Form marriage unions (next 4 functions)
    def find_people_to_avoid(self, source_node, distance, graph_T):
        """
        Uses a BFS to find people to avoid while traversing down the tree in the find_mate() function
        This reduces the likelihood of forming unions with true distances that are smaller than desired distance
        For example, it the model goes up through one great grandparent, it will not be able to go back fown through
        any of that grandparent's children 

        Parameter:
            source_node (int): a single node labeled with int
            distance (int): desired distance
            graph_T (nx.graph): reversed graph
        Return:
            people_to_avoid (set): set of nodes too avoid while traversing the list
        """
        S = [[source_node]]
        path_ = []
        num_up = np.ceil(distance/2) # if distance is odd, it goes up one more than down. if distance is even, it goes up and down same time

        # BFS to find nodes it is not allowed to visit when going down
        people_to_avoid = set()   # set of people to avoid while traversing back down (to eliminiate incorrect distances)
        Q = [[source_node]]
        path_ = Q.pop(0)   # Get first element
        people_to_avoid.add(path_[-1])
        while len(path_) < num_up + 1:   # Only need to check people in number up (excluding the redudat final generation)
            if len(S) == 0:   # If empty the search is done
                break

            c_node = path_[-1]

            parents = set(graph_T[c_node])
            parents_ = [n for n in parents if graph_T[c_node][n]["Relationship"] != "Marriage"]
            people_to_avoid.update(parents_)

            if len(parents_) != 0:   # if the c_node has at least one parent nodes
                for walk in parents_:     # make new path_ and append them in S
                    Q.append(path_ + [walk])

            # print("Q", Q)     # for Debugging
            if len(Q) == 0: break
            path_ = Q.pop(0)   # Get first element

        return people_to_avoid
    

    def find_mate_networkX(self, source_node, distance, graph_un, current_generation=set(), previous_generation=set()):
        """
        Uses networkX to find a set of all nodes at a specified distance from the source
        Checks to see if any of those nodes are in a valid generation to marry

        Paramater:
            source_node (int): a single node labeled with int
            distance (int): desired distance
            graph_un (nx.graph): undirected graph with marriage edges thrown out
            current_generation (set): labeled numbers on nodes that are in current generation
            previous_generation (set): labeled numbers on nodes that are in previous generation
        Return:
            None, None: if finding a mate is impossible or is taking too much time (iterations)
            (source_node, mate_node), actual_length: if finding a mate is successful
        """
        candidates = nx.descendants_at_distance(graph_un, source_node, distance)   # get list of nodes at specified distance
        valid_gens = previous_generation.union(current_generation)  # combine list of valid nodes to marry
        if (not candidates): return None, None     # no one is at the specified distance
        # print(f"Candidates of distance {distance} from {source_node}", candidates)  # For debugging

        for candidate in candidates:
            # print("Candidate:", candidate)  # For debugging
            if candidate in valid_gens:   # If the person is valid, return the inion
                actual_length = nx.shortest_path_length(graph_un, source=source_node, target=candidate)   # For debugging (TODO: delete this redunant distance calculation)
                return (source_node, candidate), actual_length
        return None, None # No valid mate
    
    def find_mate(self, source_node, distance, graph_un, unions, tol, current_generation=[], previous_generation=[]):
        """
        By using graph search method, it will find a node in previous or in current generation
        with the shortest desired distance from source_node.

        Paramater:
            source_node (int): a single node labeled with int
            distance (int): desired distance
            graph_un (nx.graph): undirected graph with marriage edges thrown out
            unions (set of ints): labeled numbers on nodes that are already married
            tol (int): the maximum number of iterations
            current_generation (set): labeled numbers on nodes that are in current generation
            previous_generation (set): labeled numbers on nodes that are in previous generation
        Return:
            None, None: if finding a mate is impossible or is taking too much time (iterations)
            (source_node, mate_node), actual_length: if finding a mate is successful
        Caveats:
            unions should be set of nodes that are married (not tuple form)
            (Actually, if unions is in tuple form, we use for loop to check #"*")
            All edge attributes must be either "Marriage" or "Parent-Child"
        """
        # setting
        graph_T = self.G.reverse(copy=False)  # reverse graph to avoid traversal        
        num_dead_end = 0
        S = [[source_node]]
        path_ = []
        num_up = np.ceil(distance/2) # if distance is odd, it goes up one more than down. if distance is even, it goes up and down same time
        people_ignored = []

        # Nodes to avoid while traversing down depend on the method we are using
        if(self.method == "BFS"): people_to_avoid = self.find_people_to_avoid(source_node, distance, graph_T)  # find people to avoid when traversing down using BFS
        else:   # If no BFS, at least avoid siblings and parents
            parent = [par for par in graph_T[source_node] if graph_T[source_node][par]["Relationship"] != "Marriage"][0] 
            siblings = [child for child in self.G[parent] if self.G[parent][child]["Relationship"] != "Marriage"]

        # modified DFS
        while len(path_) < distance + 1:
            #print("S:", S)
            if len(S) == 0:      # if finding a node with the distance is impossible, we need to choose another source node.
                #print("S is empty")
                return None, None
            if num_dead_end == tol: 
                #print("Over tol")
                return None, None

            path_ = S.pop()   # pop a path from S

            # after reaching to the final node
            if len(path_) == distance + 1:  # since num_nodes are more than num_edges by 1
                # check if the last node is not in the current generation or previous generation
                if path_[-1] not in current_generation and path_[-1] not in previous_generation:
                    #print("The last node is not either in current generation or in previous generation.")
                    #print()
                    num_dead_end += 1
                    path_ = []
                else:
                    if path_[-1] in unions:    # check if the last node is in unions
                        num_dead_end += 1
                        path_ = []   # reset path_ so that we can stay in the while loop
                        #print(f"The last node {path_[-1]} is already married")
                        #print()
                    else:
                        # For testing purposes
                        # Check if the last node is actually the node with the shortest distance through Parent-Child edges.
                        actual_length = nx.shortest_path_length(graph_un, source=source_node, target=path_[-1])
                        return (source_node, path_[-1]), actual_length
            else:
                # up
                if len(path_) < num_up+1: # since num_nodes are more than num_edges by 1
                    #print("up")
                    c_node = path_[-1]    # c_node is the current node

                    parents = set(graph_T[c_node])
                    # Print progress
                    #for i in parents:
                        #print(i,"::",graph_T[c_node][i]["Relationship"])
                    parents_ = [n for n in parents if graph_T[c_node][n]["Relationship"] != "Marriage"]

                    # if the c_node has no parent nodes
                    if len(parents_) == 0:
                        num_dead_end += 1    # ignore the c_node and pop another node in S, which will be another parent or previous another parent
                    else:
                        # find parents nodes and randomly order them
                        chosen_parents = np.random.choice(parents_,2, replace=False) # remove union nodes and leave only choices of parents nodes
                        people_ignored.extend(list(chosen_parents))

                        for walk in chosen_parents:    # make new path_ and append them in S
                            S.append(path_ + [walk])
                # down
                else:
                    #print("down")
                    c_node = path_[-1]   # c_node is the current node

                    children = set(self.G[c_node])
                    # Print progress
                    #for i in children:
                    #    print(i,"::",graph[c_node][i]["Relationship"])
                    #print("path_:", path_)

                    children_ = [n for n in children if self.G[c_node][n]["Relationship"] != "Marriage"] # Remove unions

                    # preventing redundant paths
                    if(self.method == "BFS"): children_ = [n for n in children_ if n not in people_to_avoid]   # If using BFS method
                    else:   # If not, at least avoid the parents and siblings
                        children_ = [n for n in children_ if n not in path_]   
                        if distance > 2:
                            children_ = [n for n in children_ if n not in siblings]
                        children_ = [n for n in children_ if (n not in path_ and n not in people_ignored)] # preventing redundant paths

                    if len(children_) == 0:    # if the c_node had no children
                        num_dead_end += 1
                    else:
                        # find children nodes and randomly order them
                        chosen_children = np.random.choice(children_,len(children_), replace=False)

                        # make new path_ and append them in S
                        for walk in chosen_children:
                            S.append(path_ + [walk])

    def add_marriage_edges(self, people, prev_people, num_people, marriage_probs, prob_marry_immigrant, prob_marry, i, indices):
        """
        Forms both infinite and finite distance marriages in the current generation
    
        INVERTED: FORMS FINITE MARRAIGES FIRST
        PARAMETERS:
            people:  (list) of the current generation (IE those people elligible for
                    marriage)
            prev_people: (list) of those in the previous generation who are yet unmarried.
            num_people: (int) the number of nodes/people currently in the graph
            marriage_probs: (dictionary) keys are marriage distances, values
                are probabilities.  Note that this dictionary should only include
                entries for NON-inifite marriage distances, should have
                non-negative values which sum to 1, and should have a long right
                tail (IE lots of entries which map high (beyond what occurs in the
                example dataset) distances to zero(ish) probabilties)
            prob_marry_immigrant: (float) the probablility that a given node will marry
                    an immigrant (herein a person from outside the genealogical network,
                    without comon ancestor and therefore at distance infinity from the
                    nodes in the list 'people') (formerly 'ncp')
            prob_marry: (float) the probability that a given node will marry another
                    node in people
            i (int): generation number
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
                ancestor.  As before, a distnace of -1 indicates an infinite
                distance (IE one spouse immigrated into the community)
            immigrants: (list) list of immigrants added 
            didnt_marry: (list) of nodes in people who did not marry, will attempt
                to marry someone from the next generation
            num_never_marry (int): number of nodes in the previous generation 
                that still did not marrt
            accuracy (dict) Dictionary of distances mapping to lists of booleans 
                representing whether forming that distance marriage was both
                accurate and successful
            successes (dict) Dictionary of distances mapping to lists of booleans 
                representing whether forming that distance marriage was successful
        """
        # Construct and normalize dictionary of finite marriage distance probabilities, 
        # only allowing marriages above the minimum permissible distance in the original dataset
        try:
            minimum_permissible_distance = min(k for k in self.marriage_dists if k > -1)
        except:
            minimum_permissible_distance = 0  # All distances were infinite
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
        if self.save:
            print("\nCurrent generation number:", i)
            print("Current generation size:", gen_size)
            print("Previous generation size:", prev_gen_size)
            #print("marriage_probs", marriage_probs)
            #print("finite_probs", finite_probs)
            print("Probability of finite distance marriage:", prob_marry)
            print("Probability of infinite distance marriage:", prob_marry_immigrant)
            print("Finite couples goal:", num_finite_couples_to_marry)
            print("Infinite couples goal:", num_inf_couples_to_marry)

        people_ignored = set()
        # Dictionary of distances mapping to lists of booleans representing whether forming that distance marriage was successful
        accuracy = {val:list() for val in range(-1,50)}
        successes = {val:list() for val in range(-1,50)}

        # Make an undirected copy of the graph; filter out marriage edges; find the shortest Parent-Child path
        graph_un = self.G.to_undirected()
        edges_to_remove = [e for e in graph_un.edges(data=True) if e[2]['Relationship'] == 'Marriage']
        graph_un.remove_edges_from(edges_to_remove)

        num_unions = 0
        limit_tol = 5 # option 1: high limit_tol and remove thespecific distance // # option 2: low limit_tol and keep all distances
        num_failures = 0

        while num_unions < num_finite_couples_to_marry and can_marry:
            if num_failures > limit_tol*gen_size:
                break
                # # Modify the list of finite distnaces
                # modified_list = list(finite_probs.keys())
                # ind_num = modified_list.index(desired_dist)
                # modified_list.remove(desired_dist)

                # # Modify the list of probability distribution of finite distance
                # modified_prob = list(finite_probs.values())
                # ind_val = modified_prob[ind_num]
                # modified_prob.remove(ind_val)
                # # Normalize their probability
                # modified_prob = [i/sum(modified_prob) for i in modified_prob]

                # # Choose a distance based on new list and new probability
                # desired_dist = np.random.choice(modified_list, p = modified_prob)

            # Randomly select a desired distance and a candidate we would like to marry off. Attempt to find a mate tol times.
            desired_dist = np.random.choice(list(finite_probs.keys()), p=list(finite_probs.values()))

            candidate = np.random.choice(list(can_marry))
            # Search for a mate and set the distance variable to reflect the actual distance between the two people
            
            # Determine which find_mate function to use
            if(self.method == "NetworkX"): couple, true_dist = self.find_mate_networkX(candidate, desired_dist, graph_un, can_marry, can_marry_prev)
            else: couple, true_dist = self.find_mate(candidate, desired_dist, graph_un, people_ignored, 100, can_marry, can_marry_prev)

            if couple is not None:
                # Update the accuracy dictionary.
                # If the random walk produced the actual geodesic distance between
                # the man and woman, add a success (True); otherwise, add a failure (False)
                if true_dist == desired_dist:
                    accuracy[desired_dist].append(True)
                else:
                    accuracy[desired_dist].append(False)

                successes[desired_dist].append(True)
                # Update the people_ignored set
                people_ignored.update(list(couple))
                # Update the unions set
                unions.add(couple)

                # Update the marriage_distances list
                marriage_distances.append(true_dist)

                # Update the set of people in the current and previous generation who can marry
                can_marry.difference_update(set(couple))
                can_marry_prev.difference_update(set(couple))

                num_unions += 1
            else:
                num_failures += 1
                successes[desired_dist].append(False)

        for dist in accuracy.keys():
            if len(accuracy[dist]) > 0:
                if self.save: print(f"Distance {dist}: {100*sum(accuracy[dist])/len(accuracy[dist])}% accuracy ({sum(accuracy[dist])}/{len(accuracy[dist])}) in finding shortest-distance paths")
        for dist in successes.keys():
            if len(successes[dist]) > 0:
                if self.save: print(f"Distance {dist}: {100*sum(successes[dist])/len(successes[dist])}% successes ({sum(successes[dist])}/{len(successes[dist])}) in finding a mate")

        # DEBUG PRINT
        if self.save:
            print(f"{num_failures} failures to find a mate")
            print(f"{num_unions} finite unions have been formed")

        # Get the components of the graph and map each node in the current
        # or previous generations (that is still unmarried) to its component number
        # NOTE: this is messy/inefficient, should be improved
        # TODO: Make it more efficient for first few generations

        # if (i <= 10):  # Unions can be formed imediately since all components are not single
        #     eligible_people = can_marry.union(can_marry_prev)   # See valid people to marry

        #     num_unions = 0
        #     while num_unions < num_inf_couples_to_marry and eligible_people:
        #         candidate = eligible_people.pop()
        #         candidate2 = eligible_people.pop()
        #         works = False
                
        #         while candidate == candidate2:     # Keep lookinf for valid candidate 
        #             candidate2 = np.random.choice(eligible_people)
        #         # Add the new couple to the ignored set and unions set
        #         # and remove them from the can_marry and can_marry_prev sets (as applicable)
        #         people_ignored.update({candidate, candidate2})
        #         unions.add((candidate, candidate2))

        #         # Update the marriage_distances list
        #         marriage_distances.append(-1)

        #         can_marry.difference_update({candidate, candidate2})
        #         can_marry_prev.difference_update({candidate, candidate2})
        #         # Remove the couple from eligible_people (for future marriages)

        #         success = True
        #         num_unions += 1
        components = list(nx.connected_components(nx.Graph(self.G)))
        if self.save:
            print(f"Number of components: {sum(1 for _ in components)}")
            print(f"Graph size: {self.G.number_of_nodes()}")
        eligible_people = {x:n for n,component in enumerate(components) for x in component if x in can_marry or x in can_marry_prev}

        num_unions = 0
        while num_unions < num_inf_couples_to_marry and eligible_people:
            # Uniformly select someone in the current generation who is unmarried
            candidate = np.random.choice(list(eligible_people.keys()))
            selected_component = eligible_people[candidate]

            # Try 5 times to find a partner in a separate component from our candidate
            success = False
            attempts = 0
            while attempts < 5:
                candidate2 = np.random.choice(list(eligible_people.keys()))
                attempts += 1
                target_component = eligible_people[candidate2]
                if target_component != selected_component:
                    # Add the new couple to the ignored set and unions set
                    # and remove them from the can_marry and can_marry_prev sets (as applicable)
                    people_ignored.update({candidate, candidate2})
                    unions.add((candidate, candidate2))

                    # Update the marriage_distances list
                    marriage_distances.append(-1)

                    can_marry.difference_update({candidate, candidate2})
                    can_marry_prev.difference_update({candidate, candidate2})
                    # Remove the couple from eligible_people (for future marriages)
                    eligible_people.pop(candidate)
                    eligible_people.pop(candidate2)

                    success = True
                    num_unions += 1
                    break

            # If we attempted 5 times and failed, blacklist this candidate (so we eventually break out of the while loop)
            if not success:
                eligible_people.pop(candidate)

        # DEBUG PRINT
        if self.save: print(f"{num_unions} infinite unions have been formed")

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
        didnt_marry = list(can_marry.difference(set(married_immigrant)))
        # print("didnt marry from current gen", len(didnt_marry))   # DEBUG PRINT

        return unions, num_immigrants, marriage_distances, immigrants, didnt_marry, num_never_marry, accuracy, successes
    

    # Form child unions
    def add_children_edges(self, unions, num_people, child_probs):
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
        return child_edges, families, num_people + total_num_children, num_children_each_couple
    

    # Plot things (next 3 functions)
    @staticmethod
    def plot_find_mate_stats(cum_successes, cum_accuracy):
        """
        Create plots of successes (how often the find_mate function finds a mate) and accuracy
        (how often the two candidates' geodesic distance matches the searched distance)
        against the searched distance.
        """
        accuracy_data = {val:list() for val in range(-1,50)}
        successes_data = {val:list() for val in range(-1,50)}

        # Aggregate statistics for successes (filtered by distance)
        # and similar for accuracies
        for successes in cum_successes:
            for dist in successes:
                successes_data[dist].extend(successes[dist])

        for accuracy in cum_accuracy:
            for dist in accuracy:
                accuracy_data[dist].extend(accuracy[dist])

        # Compute totals as well as percentages for each of the quantities.
        # Relate accuracy percentage to the total number of trials, rather than the total number of successful trials
        # (which is why we divide by len(successes_data[dist]))
        distances = range(-1,50)
        success_totals = [sum(successes_data[dist]) for dist in successes_data]
        accuracy_totals = [sum(accuracy_data[dist]) for dist in accuracy_data]
        success_perc = [sum(successes_data[dist])/len(successes_data[dist]) if len(successes_data[dist]) > 0 else 0 for dist in successes_data]
        accuracy_perc = [sum(accuracy_data[dist])/len(successes_data[dist]) if len(successes_data[dist]) > 0 else 0 for dist in accuracy_data]

        plt.subplot(121)
        plt.plot(distances, success_totals, "o-", label="Number of successes")
        plt.plot(distances, accuracy_totals, "o-", label="Number of accurate successes")
        plt.xlim((0,15))
        plt.xlabel("Desired Marriage Distance")
        plt.legend()

        plt.subplot(122)
        plt.plot(distances, success_perc, "o-", label="Percentage of successes")
        plt.plot(distances, accuracy_perc, "o-", label="Percentage of accurate successes")
        plt.xlim((0,15))
        plt.xlabel("Desired Marriage Distance")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_time_to_form_marraiges(time_to_form_marraiges):
        """
        Create plot of the time it takes each generation to form all of its marraiges

        PARAMETERS:
            time_to_form_marraiges (list): The time it takes to form the marraiges in each generation
        """
        plt.plot(np.arange(len(time_to_form_marraiges)), time_to_form_marraiges)
        plt.xlabel("Generation Number")
        plt.ylabel("Time (Seconds)")
        plt.title(f"The Time it takes to form marraiges at each generation")
        plt.show()

    @staticmethod
    def plot_error_per_generation(errors):
        """
        Create plot of the time it takes each generation to form all of its marraiges

        PARAMETERS:
            time_to_form_marraiges (list): The time it takes to form the marraiges in each generation
        """
        plt.plot(np.arange(len(errors))[2:], errors[2:])  # Plot errors (excluding first few generations)
        plt.xlabel("Generation Number")
        plt.ylabel("Total Squared Error")
        plt.title(f"The error in the model after each generation")
        plt.show()

    def graph_current_distributions(self, target_marriage_probs, model_marriage_probs, adjusted_marriage_probs, gen_num, save_plots=True, alpha=0.85):
        """
        Plots the distribution of distance to unions in a given generation
        """
        name = self.name
        eps = self.eps
        outpath = self.output_path

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

        plt.legend(fontsize="20")
        # title = name + '\n'                       이거랑
        title = f'generation: {gen_num} \n'
        ticks = [x for x in range(-1, max_bin + 1)]  # 이거랑
        marks = [r'$\infty$'] + ticks[1:]  # 이거랑
        plt.xticks(ticks, labels=marks)  # 이거랑 다름
        plt.title(title, fontsize=30, pad=2)
        plt.ylabel("Ratio", fontsize=20)
        plt.xlabel("Distance", fontsize=20)

        if not save_plots:
            plt.show()
        else:
            plt.savefig(os.path.join(outpath,  name + f'distributions_generation_{gen_num}' + '.png'), format='png')
        plt.clf()  # clear out the current figure
        plt.close(fig)


    # Actually make the model
    # Same as human_family_network_variant()
    def make_model(self, num_people, when_to_stop=None, num_gens=np.inf, out_dir='output', eps=0, fixed=False, method="BFS", tol=1e-1):
        """
        This is the main driver for implementing the Variant Target Model (see
        Chapter 5 in Kolton's thesis).
        PARAMETERS:
            num_people (int): number of people (nodes) to include in initial
                generation
            when_to_stop: (int) target number of nodes to capture.  If supplied, the
                model will run until this target number of nodes (all together, not
                just the size of the current generation) is surpassed.
            num_gens (int): max number of generations for network to grow beyond the
                initial generation.  Default is np.inf.  If np.inf, then the model
                will run until the number of nodes in the example network is
                surpassed and then stop.
            out_dir (str): destination directory for save output.  Each run will
                create an out_dir/{name}_{int}/ file structure, counting version
                number automatically
            eps (int): Allowed error
            method (str): Used to tell make_model which method it should use when forming marraige unions
                'BFS' - Do a BFS up the tree to determine which nodes to avoid while traversing done
                'NetworkX' - Use Networkx BFS to find a mate of specified distance
                anything else - Run original model, only avoiding parrents and siblings
            fixed (boolean): Used to determine if we use a fixed probability distribution or not
        ATTRIBUTES (That it creates):
            num_people_first_generation (int): the number of people put in the first generation
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
            out_dir (str): destination directory for save output.  Each run will
                create an out_dir/{name}_{int}/ file structure, counting version
                number automatically
            dies_out (bool):  False indicates that the model exited successfully---
                generations 0 to L (inclusive) contain more than when_to_stop nodes
                in total OR L >= num_gens.  True else.
            summary_statistics (tuple): Ordered tuples of integers 
                (# total people in graph,  # immigrants, # num_children, # num marriages, 
                self.prob_inf_marriage, self.prob_finite_marriage, self.prob_inf_marriage(eligible_only), 
                self.prob_finite_marriage(elegible_only))
            method (str): Used to tell make_model which method it should use when forming marraige unions
                'BFS' - Do a BFS up the tree to determine which nodes to avoid while traversing done
                anything else - Run original model, only avoiding parrents and siblings
            fixed (boolean): Used to determine if we use a fixed probability distribution or not
            generation_ranges: (list) Stores the largest person number in each generation 
                index: generation number
                value: total_num_people in generations 0-index
                Note: The initial generation of infinite distance people is gen 0
            num_people_setup (int): The number of people chosen to be in the first generation
            num_people (int): The number of people in the entire model
            tol (float)  # Stopping criteria 
                (the algorithm will stop when the difference between the erros is less than tol)
            eps (int)   # Allowed error
            error (float)   # The total sum squared error between the distributions
        """
        self.reset()  # Reset model to, making the instance what it was when first created
        # Set new attributes
        self.out_dir = out_dir
        self.fixed = fixed
        self.eps = eps
        self.tol = tol
        self.num_people_setup = num_people
        self.method = method
        self.generation_ranges = [self.num_people_setup]   # Stores the largest person number in each generation [index: generation number, value: total_num_people]
        self.G = nx.DiGraph()
        if self.save: self.output_path = self.makeOutputDirectory()
        else: self.output_path = ''

        # Initialize variables
        if(when_to_stop is None): when_to_stop = self.num_people_orig_graph   # If nothing was passed in, use the total number of people in the initial model
        total_num_single = 0
        dies_out = False
        all_marriage_edges = []
        all_marriage_distances = []
        all_temp_marriage_edges = []
        all_temp_marriage_distances = []
        all_children_per_couple = []
        prev_generation_still_single = []
        formula_target = [self.num_people_setup]
        summary_statistics = []  # will hold ordered tuples of integers (# total people in graph,  # immigrants, # num_children, # num marriages, self.prob_inf_marriage, self.prob_finite_marriage, self.prob_inf_marriage(eligible_only), self.prob_finite_marriage(elegible_only))
        i = 1   # Generation number
        tem_num_people = 0
        temp = 0
        # Distance matrix stuff (no longer in use)
            # D = np.ones((num_people, num_people)) * -1  # starting off, everyone is infinite distance away
            # np.fill_diagonal(D, 0)  # everyone is 0 away from themselves, also weirdly done inplace

        # Collect data for plotting later about the accuracy and timing of our algorithm
        find_mate_accuracy = []
        find_mate_successes = []
        time_to_form_marraiges = []           # Used to plot information about the time it takes to form all the unions

        indices = {node + 1:k for k, node in enumerate(range(num_people))}  # name:index
        generation_of_people = list(indices.keys())

        # explicitly add our first generation of nodes (otherwise we will fail to add those who do not marry into our graph).  
        # All future generations are connected either through marriage or through parent-child arcs
        self.G.add_nodes_from(generation_of_people, layer=0, immigrant=True, parents=0)

        # get probabilities of possible finite distances to use in marriage function and normalize it
        # maybe don't normalize so that you will (approximately) retain the prob_inf + prob_finite + prob_single = 1
        marriage_probs = self.get_marriage_probabilities(self.marriage_dists, when_to_stop)  # hand in the ORIGINAL MARRIAGE DISTRIBUTION
        marriage_probs = {key:value/sum(marriage_probs.values()) for key, value in zip(marriage_probs.keys(), marriage_probs.values())}
        lim = len(marriage_probs)

        # ditto for the child distribution
        # ??? make probabilities non-negative (some entries are effectively zero, but negative)
        child_probs = self.get_child_probabilities(self.children_dist)
        child_probs = {key:value if value > 0 else 0 for key, value in zip(child_probs.keys(), child_probs.values()) }
        child_probs = {key:value/sum(child_probs.values()) for key, value in zip(child_probs.keys(), child_probs.values())}

        # setting for restricting the first a few of the length of probability distribution of marriage distance
        temp_marriage_probs = {key: value / sum(marriage_probs.values()) for key, value in
                            zip(marriage_probs.keys(), marriage_probs.values())}
        v = list(temp_marriage_probs.values())
        k = list(temp_marriage_probs.keys())

        total_error = 1000000       # the initial total_error is just needed to be greater than 50
        total_error_next = 10000    # the initial total_error_next is just needed to be greater than total_error
        errors = []   # to store the errors that we will use in a graph

        # grow the network until:
            # the total error is less than 50 or
            # the difference between total_error and next total_error is less than 1e-1 or
            # the number of generations is more than 30 or
            # the model dies out
        while total_error_next > 50 and abs(total_error - total_error_next) > self.tol and i < 30:
            # You can fix the probability distribution of marriage distance by fixed = True
            total_error = total_error_next   # Reset the total_error

            if len(generation_of_people) == 0:
                dies_out = False
                break
            # update your current finite marriage probabilities to favor those which are yet underrepresented
            if (i > round(lim/2)+1) and (i <= round(lim/2*3/2)+1):
                new_marriage_probs, total_error_next = self.get_difference_in_probabilities(marriage_probs, all_temp_marriage_distances, num_people, i, plot=self.save, fixed=self.fixed)  #all_temp_marriage_distances 이게 아니라 all_marriage_distances 이거였음
            elif i > round(lim/2*3/2)+1:
                new_marriage_probs, total_error_next = self.get_difference_in_probabilities(marriage_probs, all_marriage_distances, num_people - self.num_setup_people, i, plot=self.save, fixed=self.fixed)
            else:
                # if this is the first generation beyond the initial set up OR
                # if you don't yet have more than one unique distance in your list of marriage edges,
                # then just default to the unaltered finite distance marriage distribution
                if i == 1:
                    new_marriage_probs = {key:value/sum(v[0:i*2]) for key, value in zip(k[0:i*2], v[0:i*2])}
                    total_error_next = total_error_next+1 # To avoid terminating while loop in early stage
                elif i == 2:
                    if i in k[0:i*2]:
                        marriage_probs_stricted = {key:value/sum(v[0:k.index((i-1)*2)+1]) for key, value in zip(k[0:k.index((i-1)*2)+1], v[0:k.index((i-1)*2)+1])}
                        new_marriage_probs, total_error_next = self.get_difference_in_probabilities(marriage_probs_stricted, all_temp_marriage_distances,
                                                                            num_people, i, plot=self.save,fixed=self.fixed)
                    else:  total_error_next = total_error_next+1   # To avoid terminating while loop in early stage
                else:
                    if ((i-1)*2) in k[0:i*2]:
                        marriage_probs_stricted = {key:value/sum(v[0:k.index((i-1)*2)+1]) for key, value in zip(k[0:k.index((i-1)*2)+1], v[0:k.index((i-1)*2)+1])}
                        new_marriage_probs, total_error_next = self.get_difference_in_probabilities(marriage_probs_stricted, all_temp_marriage_distances,
                                                                            num_people, i, plot=self.save, fixed=self.fixed)
                    else:  total_error_next = total_error_next+1

            new_prob_inf_marriage = new_marriage_probs[-1]
            new_prob_finite_marriage = sum(new_marriage_probs.values()) - new_marriage_probs[-1] - new_marriage_probs[0]

            # Time how long it takes to make the unions
            start = time.time()
            unions, num_immigrants, marriage_distances, immigrants, \
                prev_generation_still_single, stay_single_forever, \
                    accuracy, successes = self.add_marriage_edges(people=generation_of_people, prev_people=prev_generation_still_single, num_people=num_people, 
                                                                  marriage_probs=new_marriage_probs, prob_marry_immigrant=new_prob_inf_marriage, 
                                                                  prob_marry=new_prob_finite_marriage, i=i, indices=indices)
            end = time.time()
            # Gather statistics starting at the 5th generation (once there are lots of possible paths available for each distance)
            time_to_form_marraiges.append(end - start)

            if i > 5:
                find_mate_accuracy.append(accuracy)
                find_mate_successes.append(successes)

            total_num_single += stay_single_forever
            self.G.add_nodes_from(immigrants, layer=i-1, immigrant=True, parents=0)
            self.G.add_edges_from(unions, Relationship="Marriage")        #adding marriage edges
            if i > round(lim/2*3/2):
                all_marriage_edges += list(unions)
                all_marriage_distances += marriage_distances
            else:
                self.num_setup_people = num_people
                all_temp_marriage_edges += list(unions)
                all_temp_marriage_distances += marriage_distances

            max_ind = max(indices.values())
            indices = indices | {immigrant + num_people + 1:ind for ind, immigrant in zip(range(max_ind + 1, max_ind+1+num_immigrants), range(num_immigrants))}  # +1 since we begin counting people at 1, 2, ... not at 0, 1, ...

            if i > round(lim/2*3/2)+1:
                stats = [num_people - self.num_setup_people, num_immigrants]
            num_people += num_immigrants
            current_people = num_people

            # add children to each marriage
            child_edges, families, num_people, num_children_per_couple = self.add_children_edges(unions, num_people, child_probs)
            self.generation_ranges.append(num_people)   # Store the largest person number in each generation
            formula_target.append(self.num_people_setup * (self.prob_finite_marriage * np.mean(self.children_dist)) ** i + num_immigrants)
            if self.save: print("Generation Ranges:", self.generation_ranges)  # For debugging
            added_children = num_people - current_people


            ########################################################
            # new indices except for updating D matrix
            ind_children = [child for fam in families for child in fam] # list of individuals of children

            indices = {person:k for k, person in enumerate(prev_generation_still_single + ind_children)}
            ########################################################

            generation_of_people = [key for key in indices.keys() if key not in prev_generation_still_single]  # only grab the new people
            self.G.add_nodes_from(generation_of_people, layer=i, immigrant=False, parents=2)
            self.G.add_edges_from(child_edges, Relationship='Parent-Child')
            all_children_per_couple += list(num_children_per_couple)


            if i > round(lim/2*3/2)+1:
                tem_num_people += num_immigrants + added_children
                stats.append(sum(num_children_per_couple))
                stats.append(len(unions))
                stat_prob_marry = len(all_marriage_distances) * 2 / len(self.G)
                if self.save: print("len(all_marriage_distances)= ", len(all_marriage_distances), " increased= ", len(all_marriage_distances)-temp)
                temp = len(all_marriage_distances)
                if len(all_marriage_distances) > 0:
                    stat_frac_inf = sum(np.array(all_marriage_distances) == -1) / len(all_marriage_distances)
                else:
                    stat_frac_inf = np.nan
                stats.append(stat_prob_marry * stat_frac_inf)
                stats.append(stat_prob_marry * (1 - stat_frac_inf))

                # now recalculate marriage stats using only the eligible nodes
                # (IE not those preceding gen 0, not prev_gen_still_single, and not
                # leaf nodes---only those nodes that were given the chance to marry)
                stat_prob_marry = len(all_marriage_distances) * 2 / (len(self.G) - self.num_setup_people - len(prev_generation_still_single) - sum(num_children_per_couple))
                stats.append(stat_prob_marry * stat_frac_inf)
                stats.append(stat_prob_marry * (1 - stat_frac_inf))
                stats.append(nx.number_connected_components(nx.Graph(self.G)))
                summary_statistics.append(stats)

            errors.append(total_error_next)
            if self.save and i > 2: print(f"Total error: {total_error_next} is within {abs(total_error_next - total_error)} of previous error")
            i += 1

        plt.plot(self.generation_ranges, label='Model')
        plt.plot(formula_target, label='Target')
        plt.xlabel('Generation')
        plt.ylabel('Number of People')
        plt.legend()
        plt.savefig('./output/{}_{}.png'.format(self.name, self.num_people_setup))
        plt.show()

        ###########################
        if(self.save):
            pass
            # HumanFamily.plot_time_to_form_marraiges(time_to_form_marraiges)
            # HumanFamily.plot_find_mate_stats(find_mate_successes, find_mate_accuracy)
            # HumanFamily.plot_error_per_generation(errors)
        ##########################

        # Store data
        self.summary_statistics = summary_statistics
        self.all_marriage_edges = all_marriage_edges
        self.all_marriage_distances = all_marriage_distances
        self.all_children_per_couple = all_children_per_couple
        self.dies_out = dies_out
        self.num_people = num_people
        self.error = total_error_next

        # Print Out Useful Data
        if self.save:
            print()
            print(f"{self.name} had {self.num_people_orig_graph} people")
            print(f"The model made {self.num_people} people")
            print("##It is Done##")  # :)

        # Did our model success to make a structure that has total error less than 50?
        if total_error_next < 50:
            success = True
        else:
            success = False
        return success, num_immigrants
     

    def build_single_component(self):
        """Build a single component out of the graph's many components and stores data about it in the output directory
        Note: This use to be the last part of human_family_network_variant()
        Since it takes a long time and is not always needed, it is now a separate function that can be called only after make_model

        Returns:
            G_connected (nx.DiGraph): modeled network formatted as above, but
                    with auxillary ancestry forming a single connected component out of
                    the (many) components in G.  Vertices in G_connected may have their
                    'layer' attribute set to 'setup', indicating that the vertex is part
                    of an auxiliary ancestral line which was introduced to connect two
                    immigrant vertices in different components via a common ancestor.
                    Additionally, these now connected immigrant vertices will have their
                    'parents' attribute incremented to either 1 or 2.
                    G is a subgraph of G_connected.
        """ 
        # Get the data of union distance
        marriage_dist_array = np.array(self.marriage_dists)
        # Get the data of finite union distnace
        finite_only_marriage_dist = marriage_dist_array[marriage_dist_array != -1]
        G_connected = nx.DiGraph()  #G.subgraph([node for node, d in G.nodes(data=True) if d['layer'] != 'setup']).copy()

        for component in nx.connected_components(nx.Graph(self.G)):
            if len(G_connected) == 0:
                G_connected = self.G.subgraph(component).copy()
                continue
            else:
                component = self.G.subgraph(component).copy()
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

                nodes_to_add = [k for k in range(self.num_people + 1, self.num_people + distance)]
                self.num_people += len(nodes_to_add)
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
        if self.save:
            df = pd.DataFrame(data=self.summary_statistics, columns=['num_people (excluding initial setup)', 'num_immigrants', 'num_children', 'num_marriages', 'prob_inf_marriage', 'prob_finite_marriage', 'prob_inf_marriage(eligible_only)', 'prob_finite_marriage(eligible_only)', 'num_connected_components'])
            df.index.name='generation'
            df.to_csv(os.path.join(self.output_path, str(self.name)+'_summary_statistics.csv'))
            # Gname = "{}/{}_G_withsetup.gpickle".format(output_path, name)   # save graph
            #nx.write_gpickle(G, Gname)
            Gname = "{}/{}_G_withsetup.pkl".format(self.output_path, self.name)   # save graph
            with open(Gname, 'wb') as out:
                pickle.dump(G_connected, out)
            # now save a reduced version of the graph, without the setup people

            Gname = "{}/{}_G_no_setup.pkl".format(self.output_path, self.name)   # save graph
            with open(Gname, 'wb') as out:
                pickle.dump(self.G, out)

            Uname = "{}/{}_marriage_edges".format(self.output_path, self.name) + '.pkl'   # save unions
            with open(Uname, 'wb') as fup:
                pickle.dump(self.all_marriage_edges, fup)
            Dname = "{}/{}_marriage_distances".format(self.output_path, self.name) +'.pkl' # save marriage distances
            with open(Dname, 'wb') as myfile:
                pickle.dump(self.all_marriage_distances, myfile)
            Cname = "{}/{}_children_per_couple".format(self.output_path, self.name) + '.pkl'  # save children
            with open(Cname, 'wb') as fcp:
                pickle.dump(self.all_children_per_couple, fcp)

            paj = format_as_pajek(self.G, name)
            with open('{}/model-{}-oregraph.paj'.format(self.output_path, self.name), 'w') as o:
                o.writelines(paj)

        return G_connected     # All other things that this use to return can be accessed as attributes

    def find_file_version_number(self, out_directory, filename, extension):
        ver = 1
        output_dir = os.path.join(out_directory, filename + '_')
        while os.path.exists(output_dir + str(ver) + extension):
            ver += 1
        filename = filename + '_' + str(ver)
        return filename

    def find_start_size(self, name, marriage_dists, prob_inf_marriage, prob_finite_marriage, children_dist, num_people_orig_graph,
                        out_directory='start_size', filename='start_size', max_iters=100, dies_out_threshold=5,
                        verbose=False, save_start_sizes=True, random_start=True,
                        return_counter=False):  # n = number of initial nodes
        counter = 0
        print("name:", name)
        print("filename:", filename)

        filename = name + '_' + filename
        greatest_lower_bound = 2
        least_upper_bound = num_people_orig_graph

        if random_start:
            num_people = np.random.randint(greatest_lower_bound, num_people_orig_graph)
        else:
            num_people = num_people_orig_graph // 2
        dies_out_counter = 0  # counter for the number of times the model dies out

        start_sizes = [num_people]
        while dies_out_counter != dies_out_threshold:  # while the number of times the model dies out is not equal to the threshold of dying:

            for i in range(max_iters):
                counter += 1
                dies = self.make_model(num_people)
                if dies:
                    dies_out_counter += 1
                if dies_out_counter > dies_out_threshold:
                    break

            if greatest_lower_bound >= least_upper_bound - 1:
                # IE the ideal lies between these two integers
                # so return the larger
                num_people = least_upper_bound
                break
            elif dies_out_counter == dies_out_threshold:
                break
            elif dies_out_counter > dies_out_threshold:  # we want to increase num_people
                greatest_lower_bound = num_people  # current iteration died out too frequently.  Won't need to search below this point again.
                num_people = (num_people + least_upper_bound) // 2  # midpoint between num_people and size_goal
                dies_out_counter = 0

            elif dies_out_counter < dies_out_threshold:  # we want to decrease num_people
                least_upper_bound = num_people  # current iteration died out too infrequently.  Won't need to search above this point again
                num_people = (greatest_lower_bound + num_people) // 2  # midpoint between 2 and num_people
                dies_out_counter = 0

            if verbose:
                print('greatest_lower_bound: ', greatest_lower_bound)
                print('least_upper_bound: ', least_upper_bound)
                print('starting population: ', num_people)
            start_sizes.append(num_people)

        if save_start_sizes:
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            filename = self.find_file_version_number(out_directory, filename, extension='.txt')
            # save a text file (one integer per line)
            with open(os.path.join(out_directory, filename + '.txt'), 'w') as outfile:
                outfile.writelines([str(k) + '\n' for k in start_sizes])
            # save the actual object
            with open(os.path.join(out_directory, filename + '.pkl'), 'wb') as outfile:
                pickle.dump(start_sizes, outfile)

        if return_counter:
            return start_sizes, counter
        else:
            return start_sizes

# Get a list of all the data sets we can work with
data_sets = [x[:-4] for x in os.listdir("Kolton_distances")]  # get from folder, excluding .txt part (last 4 characters)

# name = np.random.choice(data_sets)   # Get a random data set
# name = "tikopia_1930"
# name = "arawete"
# name = "kelkummer"
# name = "dogon_boni"
name = "tikopia_1930"

# Example on how to run the model
starting_size = 200
family = HumanFamily(name)   # Gathers data from the model and saves it into the object
family.make_model(starting_size, fixed=False, method="NetworkX")   # Actually runs the model and creates the graph (pass in initial starting size)

# family.build_single_component()    # Builds a single component and returns it (not necessary for most of our analysis)

# TODO: Plot number of component s to generation
# TODO: C thing
