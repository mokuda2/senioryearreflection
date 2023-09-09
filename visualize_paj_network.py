# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:54:52 2022

@author: kolto

takes a .paj file (see Section 3.1)---either a real-world genealogical network
or the result of the Target Model or Variant Target Model---and saves
visualizations of the network.

Can also work with nx.DiGraph objects from human_family_network() in
family_model_intergenerational_marriage.py or in
family_model_intergenerational_marriage_reduced.py.

There are commented out examples of how to run this code below.  
"""
#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import shlex
import os
import pickle
from get_model_parameters import separate_parts
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
#%%

def content_list(graph_name):
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
def draw_paj_graph(graph_name,
                   out_directory='distances_histograms',
                   file_name='tikopia_1930',
                   title="",
                   dpi=300,
                   figsize=(12,9),
                   layout=nx.kamada_kawai_layout,
                   layout_args=[],
                   edge_width=1,
                   node_size=1
                   ):
    """Takes in a PAJEK formatted graph file and draws the graph in an easy to interpret form.
    Marriage edges are red. Parent child edges are blue. Males are triangles, Females are circles."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # get graph data
    nodes,marr_edges,pc_edges = separate_parts(graph_name,'A')
    # create graph and add nodes
    g = nx.Graph()
    g.add_nodes_from(np.arange(1, nodes[0]+1))
    g.add_edges_from(marr_edges)
    g.add_edges_from(pc_edges)
    # get genders
    male = list(np.array(list(nodes[2].keys()))[np.array(list(nodes[2].values()))=='M'])
    female = list(np.array(list(nodes[2].keys()))[np.array(list(nodes[2].values()))=='F'])
    unknown = list(np.array(list(nodes[2].keys()))[np.array(list(nodes[2].values()))=='U'])
    # get position of nodes and edges
    pos = layout(g, *layout_args)
    # draw graph
    nx.draw_networkx_edges(g,pos,marr_edges,edge_color='r', label='marriage', width=edge_width)
    nx.draw_networkx_edges(g,pos,pc_edges,edge_color='b', label='parent-child', width=edge_width)
    nx.draw_networkx_nodes(g,pos,male,node_shape = 'v',node_color='k',node_size=node_size, label='male')
    nx.draw_networkx_nodes(g,pos,female,node_shape='o',node_color='k',node_size=node_size, label='female')
    nx.draw_networkx_nodes(g,pos,unknown,node_shape='x',node_color='g',node_size=node_size)
    plt.title(title, fontsize=24)
    plt.legend(fontsize=16)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    plt.savefig(os.path.join(out_directory, file_name + '_family_network.png'), format='png')
    plt.show()
    print('Saved to: {}'.format(os.path.join(out_directory, file_name + '_family_network.png')))


#%%
def draw_nx_graph(graph_name,
                   out_directory='distances_histograms',
                   file_name='tikopia_1930',
                   title="",
                   dpi=300,
                   figsize=(12,9),
                   layout=nx.multipartite_layout,
                   layout_args=['layer', 'horizontal'],
                   marriage_edge_width=1,
                   marriage_edge_connection_style='arc3',
                   marriage_edge_alpha=0.35,
                   parent_child_edge_width=1,
                   parent_child_edge_alpha=0.35,
                   node_size=1,
                   draw_generation_by_generation=False,
                   rows=False,
                   cols=False,
                   new_gen_scale=1.33,
                   ):
    """Takes in a nx formatted graph file and draws the graph in an easy to interpret form.
    Marriage edges are red. Parent child edges are blue. Males are triangles, Females are circles."""
    # fig = plt.figure(figsize=figsize, dpi=dpi)

    with open(graph_name, 'rb') as infile:
        g = pickle.load(infile)
    edge_attributes = nx.get_edge_attributes(g, 'Relationship')
    marr_edges = [edge for edge in edge_attributes.keys() if edge_attributes[edge]=='Marriage']
    pc_edges = [edge for edge in edge_attributes.keys() if edge_attributes[edge]=='Parent-Child']
    # get position of nodes and edges

    pos = layout(g, *layout_args)

    # draw graph
    if not draw_generation_by_generation:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        nx.draw_networkx_edges(g,
                                pos,
                                pc_edges,
                                ax=ax,
                                edge_color='b',

                                width=parent_child_edge_width,
                                arrowstyle='-',
                                alpha=parent_child_edge_alpha,
                                label='parent-child',)
        nx.draw_networkx_edges(g,
                               pos,
                               marr_edges,
                               ax=ax,
                               edge_color='r',
                               label='marriage',
                               width=marriage_edge_width,
                               arrowstyle='-',
                               connectionstyle=marriage_edge_connection_style,
                               alpha=marriage_edge_alpha)
        nx.draw_networkx_nodes(g,pos,g.nodes, ax=ax, node_shape = 's',node_color='k',node_size=node_size, label='person')
        plt.title(title, fontsize=24)
        ax.legend(fontsize=16)
        if layout == nx.multipartite_layout:
            unique_ys = set([k[1] for k in pos.values()])
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(min(unique_ys) - 0.1 * np.abs(min(unique_ys)), max(unique_ys) + 0.1 * np.abs(max(unique_ys)))
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        plt.savefig(os.path.join(out_directory, file_name + '_family_network.png'), format='png')
        plt.show()
        print('Saved to: {}'.format(os.path.join(out_directory, file_name + '_family_network.png')))
    else:
        generations = sorted(list(set(nx.get_node_attributes(g, 'layer').values())))
        prev_h_nodes = []
        running_total_h_nodes = []
        if not rows or not cols:
            cols = 4
            rows = max(generations) // cols + 1
            if rows == 1:
                cols = max(generations) + 1
        subplot_ind = 0
        fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

        for gen_num in generations:
            subplot_ind += 1
            ax = plt.subplot(rows, cols, subplot_ind)
            # plot the new generation
            gen = [node for node in g.nodes if g.nodes[node]['layer'] == gen_num]
            h = g.subgraph(gen)
            prev_h = g.subgraph(gen + prev_h_nodes)
            gen_marr_edges = [edge for edge in h.edges() if h.edges[edge]['Relationship']=='Marriage']
            gen_pc_edges = [edge for edge in prev_h.edges() if prev_h.edges[edge]['Relationship']=='Parent-Child']

            nx.draw_networkx_edges(prev_h,
                                    pos,
                                    gen_pc_edges,
                                    ax=ax,
                                    edge_color='b',
                                    width=parent_child_edge_width,
                                    arrowstyle='-',
                                    alpha=parent_child_edge_alpha,
                                    label='parent-child',)
            nx.draw_networkx_edges(h,
                                   pos,
                                   gen_marr_edges,
                                   ax=ax,
                                   edge_color='r',
                                   label='marriage',
                                   width=marriage_edge_width,
                                   arrowstyle='-',
                                   connectionstyle=marriage_edge_connection_style,
                                   alpha=marriage_edge_alpha)
            nx.draw_networkx_nodes(h,pos,h.nodes, ax=ax, node_shape = 's',node_color='k',node_size=node_size*new_gen_scale, label='person')

            running_total_h = g.subgraph(running_total_h_nodes)
            prev_marr_edges = [edge for edge in running_total_h.edges() if running_total_h.edges[edge]['Relationship']=='Marriage']
            prev_pc_edges = [edge for edge in running_total_h.edges() if running_total_h.edges[edge]['Relationship']=='Parent-Child']
            nx.draw_networkx_edges(running_total_h,
                                    pos,
                                    prev_pc_edges,
                                    ax=ax,
                                    edge_color='b',
                                    width=parent_child_edge_width,
                                    arrowstyle='-',
                                    alpha=0.18,
                                    label='parent-child',)
            nx.draw_networkx_edges(running_total_h,
                                   pos,
                                   prev_marr_edges,
                                   ax=ax,
                                   edge_color='r',
                                   label='marriage',
                                   width=marriage_edge_width,
                                   arrowstyle='-',
                                   connectionstyle=marriage_edge_connection_style,
                                   alpha=0.25)
            nx.draw_networkx_nodes(running_total_h,pos,running_total_h.nodes, ax=ax, node_shape = 's',node_color='k',node_size=node_size*1/new_gen_scale, label='person', alpha=0.25)
            ax.set_title('generation {}'.format(gen_num), fontsize=14)
            if layout == nx.multipartite_layout:
                unique_ys = set([k[1] for k in pos.values()])
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(min(unique_ys) - 0.1 * np.abs(min(unique_ys)), max(unique_ys) + 0.1 * np.abs(max(unique_ys)))
            prev_h_nodes = list(h.nodes)
            running_total_h_nodes += list(h.nodes)
        #fig.legend(fontsize=16)
        while subplot_ind != rows * cols:
            subplot_ind += 1
            ax = plt.subplot(rows, cols, subplot_ind)
            ax.axis('off')
        fig.suptitle(title, fontsize=24)
        plt.tight_layout(w_pad=0)
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        plt.savefig(os.path.join(out_directory, file_name + '_family_network_gen_by_gen.png'), format='png')
        plt.show()
        print('Saved to: {}'.format(os.path.join(out_directory, file_name + '_family_network_gen_by_gen.png')))
        #%%

# graph_name = 'Original_Sources/kinsources-arara-oregraph.paj'
# draw_paj_graph(graph_name,
#                title='Arara (actual)',
#                out_directory='target_network_visualizations',
#                file_name='arara_actual',
#                node_size=40,
#                 layout=graphviz_layout,
#                 layout_args=['dot'],
#                 edge_width=2,
#                 )
#
# #%%
#
# graph_name = "output/arara_1/arara_G.gpickle"
# draw_nx_graph(graph_name, title='Arara (model)',
#                out_directory='target_network_visualizations',
#                file_name='arara_{}'.format(layout),
#                marriage_edge_width=1,
#                marriage_edge_connection_style='arc3,rad=0.75',
#                #marriage_edge_connection_style='arc,angleA=45,angleB=315,armA=0.5, armB=0.5,rad=1',
#                parent_child_edge_width=1,
#                layout_args=['layer', 'horizontal'],
#                draw_generation_by_generation=True,
#                rows=2,
#                cols=3
#                )
#
# #%%
# layout = "kamada_kawai_layout"
# graph_name = "output/arara_1/arara_G.gpickle"
# draw_nx_graph(graph_name, title='Arara (model)', #.format(layout),
#                out_directory='target_network_visualizations',
#                file_name='arara_{}'.format(layout),
#                marriage_edge_width=10,
#                marriage_edge_alpha=0.75,
#                #marriage_edge_connection_style='arc3,rad=0.15',
#                parent_child_edge_width=2,
#                parent_child_edge_alpha=0.35,
#                node_size=10,
#                # layout_args=['layer', 'horizontal']
#                layout=nx.kamada_kawai_layout,
#                layout_args=[],
#                draw_generation_by_generation=True,
#                rows=2,
#                cols=3
#                )
#
# #%%
# layout = "dot"
# graph_name = "output/arara_1/arara_G.gpickle"
# draw_nx_graph(graph_name, title='Arara (model)', #.format(layout),
#                out_directory='target_network_visualizations',
#                file_name='arara_{}'.format(layout),
#                marriage_edge_width=5,
#                marriage_edge_alpha=0.75,
#                #marriage_edge_connection_style='arc3,rad=0.15',
#                parent_child_edge_width=2,
#                parent_child_edge_alpha=0.75,
#                node_size=10,
#                # layout_args=['layer', 'horizontal']
#                layout=graphviz_layout,
#                layout_args=['dot'],
#                draw_generation_by_generation=True,
#                rows=2,
#                cols=3
#                )
#
# #%%
# graph_name = "output/saudi_royal_genealogy_5/saudi_royal_genealogy_G.gpickle"
# draw_nx_graph(graph_name, title='Saudi Royal Genealogy (model)',
#                out_directory='target_network_visualizations',
#                file_name='saudi_royal_genealogy_model_by_generation',
#                marriage_edge_width=1,
#                marriage_edge_connection_style='arc3,rad=0.15',
#                parent_child_edge_width=1,
#                layout_args=['layer', 'horizontal'],
#                draw_generation_by_generation=True,
#                )
