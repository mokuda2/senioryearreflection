import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from graph_attributes import *
# from scipy.stats import linregress
# import pandas as pd
# import pyarrow.feather as feather
# import ast
# import random
import os
import shlex
import regex as re
import glob
import pickle 
from family_model import get_graph_stats
#%%

def content_list(graph_name):
    """
    returns list of rows from pajek file, used in other functions
    """
    # open and read graph file
    with open(graph_name, 'r') as file:
        contents = file.readlines()

    # clean up graph file
    for i in range(len(contents)):
        contents[i] = contents[i].rstrip() # remove trailing whitespace
    contents = np.array(contents)
    noempties = [item != '' for item in contents] # get rid of elements with empty strings
    return list(contents[noempties]) # turn it back into a list and return


#%%
def separate_parts(graph_name,edge_type):
    """Count the number of edges in a pajek graph file.
           Parameters:
               graph_name (str): full path to .paj file
               edge_type (str):
           Returns:
               nodes (list of lists): rows are people, columns are attributes
               marriages (list): list of tuples representing marriage edges
               pc (list): list of tuples representing parent child edges
    """
    # read in and format contents of pajek file
    contents = content_list(graph_name)
    # get the correct edge-type
    if edge_type == 'M':
        start_str = '*edges' # represent marriages
        alter_str = '*arcs'
    elif edge_type == 'PC':
        start_str = '*arcs'  # represent parent-child relationships
        alter_str = '*edges'
    elif edge_type == 'V':
        start_ind = 2
    elif edge_type == 'A':
        return separate_parts(graph_name,'V'), separate_parts(graph_name, 'M'), separate_parts(graph_name,'PC')
    # return contents list (for verification purposes)
    if edge_type == 'c':
        return contents
    def get_tuples(e_list):
        # turn edge_lists into list of tuples
        for i in range(len(e_list)):
            t,f,l = e_list[i].split()
            e_list[i] = (int(t),int(f))
        return e_list
    # get vertex list and attribute dictionary
    if edge_type == 'V':
        start_ind = 2
        try:
            e_ind = contents.index('*edges')
        except:
            e_ind = 0
        try:
            a_ind = contents.index('*arcs')
        except:
            a_ind = 0
        alter_ind = min(a_ind,e_ind)
        # get names and genders
        name_dict = {}
        gender_dict = {}
        for node in contents[start_ind:alter_ind]:
            try:
                num,name,gen = shlex.split(node)
                num = int(num)
            except:
                print('Vertex Error after' + str(num))
            if gen == 'triangle':
                gen = 'M'
            elif gen == 'ellipse' or gen == 'circle':
                gen = 'F'
            elif gen == 'diamond' or gen == 'square':
                gen = 'U'
            else:
                print('Error in gender:', node)
            name_dict.update({num:name}) # add name to dict
            gender_dict.update({num:gen}) # add gender to dict

        return [alter_ind-start_ind,name_dict,gender_dict]
    # count edges
    else:
        try:
            start_ind = contents.index(start_str)
        except:
            return 0
        try:
            alter_ind = contents.index(alter_str)
        except:
            alter_ind = 0
        if start_ind < alter_ind:
            return get_tuples(contents[start_ind+1:alter_ind])
        else:
            return get_tuples(contents[start_ind+1:])
#%%


def graph_with_attributes(graph_name, directed = False,pc_only = False):
    """ Input lists, return a nx.Graph with correct node and edge attributes.
    """
    # get graph data
    nodes,marr_edges,pc_edges = separate_parts(graph_name,'A')
    # create graph and add nodes
    if directed == False:
        g = nx.Graph()
        g.add_nodes_from(np.arange(nodes[0])+1)
        if pc_only == False:
            try:
                g.add_edges_from(marr_edges)
            except:
                print('Error: marriage edges from',graph_name)
        try:
            g.add_edges_from(pc_edges)
        except:
                print('Error: parent-child edges from',graph_name)
    else:
        g = nx.DiGraph()
        #g.add_nodes_from(np.arange(nodes[0])+1)
        g.add_nodes_from(nodes[1].keys())
        g.add_edges_from(pc_edges)
    # assign names and gender attributes
    nx.set_node_attributes(g,nodes[1],'Name')
    nx.set_node_attributes(g,nodes[2],'Gender')
    # assign edge types: marriage vs parent child
    edge_type_dict = {}
    if directed == False:
        for edge in marr_edges:
            edge_type_dict.update({edge:'Marriage'})
    for edge in pc_edges:
        edge_type_dict.update({edge:'Parent-Child'})
    nx.set_edge_attributes(g,edge_type_dict,'Relationship')
    # return graph with attributes
    return g
#%%


def get_names(path = './Original_Sources', save_paths=True):
    """Create list of nx graphs from original sources"""
    name_pattern = re.compile("(?<=-).*(?=-)")
    name_list = os.listdir(path)
    n = len(name_list)
    for i in range(n):
        if save_paths:
            name_list[i] = path + '/' + name_list[i]
        else:
            name_list[i] = name_pattern.findall(name_list[i])[0]
    return name_list 
#%%


def get_graphs_and_names(path = 'Original_Sources', directed=False, sort=False):
    """Create list of nx graphs from original sources"""
    # name_list = os.listdir(path)
    # n = len(name_list)
    # for i in range(n):
    #     name_list[i] = path + '/' + name_list[i]
    name_list = glob.glob(os.path.join(path, '*.paj'))  # searches directory at path for pajek files
    name_list += glob.glob(os.path.join(path, '**', '*.paj'))  # searches its subdirs for pajek files 
    # get graphs and names
    graph_list = []
    graph_names = []
    for g in name_list:
        try:
            graph_list.append(graph_with_attributes(g,directed))
            graph_names.append(g)
        except:
            print('Graph failed:',g)
    if sort == True:
        nodes = np.array([g.number_of_nodes() for g in graph_list])
        order = np.argsort(nodes)
        g_list = [graph_list[n] for n in order]
        g_names = [graph_names[n] for n in order]
        return g_list,g_names
    else:
        return graph_list, graph_names
#%%


def get_marriage_distances_kolton(G, marriage_edges, name='', plot=True, save=True):
    """
    finds the initial marriage distance distrubtion (IE distance to nearest
    common ancestor) for a given genealogical network.  Assumes that siblings
    are distance 2 apart.

    PARAMETERS:
    G: (networkx digraph):  should comprise ALL nodes in the given community and
                            all parent-child edges (directed FROM parent TO
                            child) but should NOT contain ANY marriage edges (IE
                            those between spouses)
    marraige_edges (list of tuples):  list of marriages in the network (man,
                            wife)
    name: (str) the unique name of the graph which will be concatenated with
            the path to the .paj file (e.g. 'tikopia_1930')
    plot: (bool) indicating whether or not to generate a histogram of
            marriage distances when calling get_marraige_distances_kolton()
    save: (bool) indicating whether the histogram should be saved as a .pdf
            file or not.  If False, the histogram will be displayed at run
            time, but not saved for later use.

    RETURNS:
    distances: (list) of len(marriage_edges); each entry is the distance from
                            one spouse, through the nearest common ancestor, to
                            the other spouse in the corresponding entry of
                            marraige_edges.  Infinite distance marriages (when
                            no common ancestor exists within the network) are
                            assigned a distance of -1.
    num_inf_marriages: (int) number of infinite distance (distance of -1)
                            marriages in the genealogical network.
    percent_inf_marraiges: (float) proportion of number of infinite distance
                            (distance of -1) marriages to the total number of
                            marriages in the genealogical network.
    """
    distances = []
    num_reports = 10
    when_to_print = int(len(marriage_edges) / num_reports)
    for i, couple in enumerate(marriage_edges):
        # get parents
        paternal_gen = {couple[0]}
        maternal_gen = {couple[1]}
        paternal_tree = paternal_gen.copy()
        maternal_tree = maternal_gen.copy()
        intersection = paternal_tree.intersection(maternal_tree)
        dist = 0
        paternal_distances = {couple[0]:dist}
        maternal_distances = {couple[1]:dist}
        while len(intersection) == 0:

            dist += 1
            paternal_gen = set([parent[0] for ancestor in paternal_gen for parent in G.in_edges(ancestor)])
            maternal_gen = set([parent[0] for ancestor in maternal_gen for parent in G.in_edges(ancestor)])
            paternal_distances = paternal_distances | {ancestor: dist for ancestor in paternal_gen if ancestor not in paternal_distances}  # if clause prevents overwriting distances if we encounter the same ancestor at multiple distances 
            maternal_distances = maternal_distances | {ancestor: dist for ancestor in maternal_gen if ancestor not in maternal_distances}
            paternal_tree = paternal_tree.union(paternal_gen)
            maternal_tree = maternal_tree.union(maternal_gen)
            intersection = paternal_tree.intersection(maternal_tree)
            if len(paternal_gen) == 0 and len(maternal_gen) == 0 and len(intersection) == 0:
                # IE if both trees are exhausted, no new parents
                dist = -1  # IE infinite distance marraige
                break
        if dist == -1:
            distances.append(dist)
        else:
            min_dist = min([paternal_distances[common_ancestor] + maternal_distances[common_ancestor] for common_ancestor in intersection])
            distances.append(min_dist)
        # if i % when_to_print == 0:
            # print(i / len(marriage_edges)*100, '% finished')
    distances_array = np.array(distances)
    num_inf_marriages = sum(distances_array == -1)
    percent_inf_marraiges = num_inf_marriages/len(marriage_edges)
    if plot and len(distances_array[distances_array != -1]) != 0:
        max_bin = int(np.max(distances_array[distances_array != -1]))
        plt.hist(distances, bins=[k for k in range(max_bin + 2)], range=(0, max_bin+2))
        title = name + '\n'
        plt.title(title + "You have {} inf-distance marriages ({}%)".format(num_inf_marriages, round(percent_inf_marraiges, 3)*100))
        if not save:
            plt.show()
        else:
            plt.savefig('./Kolton_distances_histograms/' + name + '.pdf', format='pdf')
        plt.clf()  # clear out the current figure
    if len(distances_array[distances_array != -1]) == 0:
        print("OJO! All distances were infinite!")
    return distances, num_inf_marriages, percent_inf_marraiges
#%%


def build_target_marriage_hist_kolton(name, graphs, graph_names, plot=True, save=True, path='Original_Sources'):
    """ helper function.  Returns the same things as
    get_marraige_distances_kolton() and saves histogram of the marraige
    distances for a given graph name.
    PARAMETERS:
        name: (str) the unique name of the graph which will be concatenated with
                the path to the .paj file (e.g. 'tikopia_1930')
        plot: (bool) indicating whether or not to generate a histogram of
                marriage distances when calling get_marraige_distances_kolton()
        save: (bool) indicating whether the histogram should be saved as a .pdf
                file or not.  If False, the histogram will be displayed at run
                time, but not saved for later use.
    RETURNS:
    distances: (list) of len(marriage_edges); each entry is the distance from
                            one spouse, through the nearest common ancestor, to
                            the other spouse in the corresponding entry of
                            marraige_edges.  Infinite distance marriages (when
                            no common ancestor exists within the network) are
                            assigned a distance of -1.
    num_inf_marriages: (int) number of infinite distance (distance of -1)
                            marriages in the genealogical network.
    percent_inf_marraiges: (float) proportion of number of infinite distance
                            (distance of -1) marriages to the total number of
                            marriages in the genealogical network.
    """
    # # load all original sources
    # graphs, graph_names = get_graphs_and_names(directed=True)   # graphs is a list of all Kinsource graphs
    #                                                # graph_names is a list of Kinsource graph file names
    # get number of chosen graph
    g_num = graph_names.index(path + '\kinsources-'+name+'-oregraph.paj')

    # g_num = graph_names.index(name)

    # genealogical network
    G = graphs[g_num]

    # get all parts of graph
    vertex_names, marriage_edges, child_edges = separate_parts(graph_names[g_num],'A')
    distances, num_inf_marriages, percent_inf_marraiges = get_marriage_distances_kolton(G, marriage_edges, name=name, plot=plot, save=save)
    return distances, num_inf_marriages, percent_inf_marraiges



def find_children(g_num, graph_names):
    """
    Finds number of children per married couple in the graph
    NOTE: this does not find children with only a single parent listed
    """
    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')
    # list of marriage edge tuples for the given graph
    marriages = stuff[1]
    # list of parent child edge tuples for the given graph
    children = stuff[2]
    count = []
    # go through all marriage pairs
    for i, m in enumerate(marriages):
        # get parent nodes
        p1, p2 = m
        p1_list = set()
        p2_list = set()
        # find children of each parent node & count children node
        for child in children:
            if child[0] == p1:
                p1_list.add(child[1])
            elif child[0] == p2:
                p2_list.add(child[1])
        #intersection of p lists
        c_list = p1_list.intersection(p2_list)
        count.append(len(c_list))

    return count


def save_marriage_distance_txt_files(distance_path='./Kolton_distances/', child_number_path='./ChildrenNumber/', plot=False):
    if not os.path.exists(distance_path):
        os.makedirs(distance_path)
        
    graphs, graph_names = get_graphs_and_names(directed=True)
    name_pattern = re.compile("(?<=-).*(?=-)")

    count = 0  # number of graphs included in master histogram
    master_distances = []  # for aggregate histogram
    for name in graph_names:
        print(name)
        if name == "Original_Sources\kinsources-warao-oregraph.paj":
            continue
        name = name_pattern.findall(name)[0]
        distances, num_inf_marriages, percent_inf_marraiges = build_target_marriage_hist_kolton(name, graphs, graph_names, plot=plot)
        master_distances += distances
        count += 1
        try:
            with open(os.path.join(distance_path, name + '.txt'), 'w') as outfile:
                outfile.write(str(distances))
                outfile.write('\n')
                outfile.write(str(num_inf_marriages))
                outfile.write('\n')
                outfile.write(str(percent_inf_marraiges))
                outfile.write('\n')
        except FileExistsError as e:
            print(os.path.join(distance_path, name + '.txt') + " file exists.  Skipping. ")

        g_num = graph_names.index('Original_Sources\kinsources-'+name+'-oregraph.paj')
        vertex_names, marriage_edges, child_edges = separate_parts(graph_names[g_num], 'A')

        # now save list of children per couple to a text file
        children_per_couple = find_children(g_num, graph_names)
        child_file = os.path.join(child_number_path, name+'_children.txt')
        with open(child_file, 'w') as outfile:
            outfile.write(str(children_per_couple))

    print("count: ", count )
    try:
        with open(os.path.join(distance_path, "master_distances_104"+'.txt'), 'w') as outfile:
            outfile.write(str(master_distances))
            outfile.write('\n')
    except FileExistsError as e:
        print("\'./Kolton_distances/master_distances_104.txt\' file exists.  Skipping. ")


#%%
path_to_paj = './output/arara_4/model-arara-oregraph.paj'
name = 'arara_1'

# path_to_paj = graph_names[11]
def find_model_marriage_child_distributions_from_paj(path_to_paj, plot=False, save=False):
    # genealogical network
    G = graph_with_attributes(path_to_paj, directed=True)

    # get all parts of graph
    vertex_names, marriage_edges, child_edges = separate_parts(path_to_paj,'A')
    # marraige info
    distances, num_inf_marriages, percent_inf_marraiges = get_marriage_distances_kolton(G, marriage_edges, name='', plot=plot, save=save)
    # child info 
    children_per_couple = find_children(0, [path_to_paj])  # hate that formatting, but I think that it will work 
    
    return distances, num_inf_marriages, percent_inf_marraiges, children_per_couple 

#%% 
def compile_historgrams_of_target_model_and_paj_distributions(path_to_model, name):
    # calculate distances/children per couple from the pajek file
    # NOTE: this uses ONLY the ACTUAL network structure and ignores the artificially-imposed distances from our model's initial generation setup 
    paj_distances, paj_num_inf_marriages, paj_percent_inf_marriages, paj_children_per_couple = find_model_marriage_child_distributions_from_paj(os.path.join(path_to_model, 'model-'+name+'oregraph.paj'))
    
    # load the distances, children per couple that were saved during construction 
    # NOTE: these distances are calculated acounting for the artificially-imposed distances from our model's intial generation setup 
    #       (IE those distances which are imposed in our initial generation's distance matrix D, but for which there is not a corresponding 
    #        structure in the actual graph object)
    with open(os.path.join(path_to_model, name+'_marriage_distances.pkl'), 'rb') as infile: 
        artificial_distances =  pickle.load(infile)
    with open(os.path.join(path_to_model, name+'_children_per_couple.pkl'), 'rb') as infile:
        artificial_children_per_couple = pickle.load(infile) 
    
    # load the target data 
    target_distances, target_num_marriages, target_prob_inf_marriage, target_prob_finite_marriage, target_child_dist, target_size_goal = get_graph_stats(name)
    
    