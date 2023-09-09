# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:40:26 2023

@author: kolto
"""
import networkx as nx 

def format_as_pajek(G, name):
    """
    Accepts a networkx graph object (with edges labeled with Relationship='Marriage' or Relationship='Parent-Child')
    and returns a string formatted as a pajek ore graph file 

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # get list of nodes 
    nodes = G.nodes()
    # get list of arcs (parent-child relationships)
    edges_with_attributes = G.edges.data('Relationship')  # (u, v, 'Marriage') or (u, v, 'Parent-Child')
    arcs = [(u, v) for u, v, relationship in edges_with_attributes if relationship=='Parent-Child']
    edges = [(u, v) for u, v, relationship in edges_with_attributes if relationship=='Marriage']

    paj = '*Network Ore graph, model of ' + name
    paj += '\n'
    paj += '\n'
    paj += '*vertices ' + str(G.number_of_edges())
    paj += '\n'
    for node in nodes:
        paj += str(node) + f' \'name place holder ({node})\' ' + 'square'
        # should I format with no-name and square (denotes unknown sex in pajek ore graph filing system)
        paj += '\n'
    paj += '*arcs'
    paj += '\n'
    for arc in arcs: 
        paj += str(arc[0]) + ' ' + str(arc[1]) + ' 1'  # weight 
        paj += '\n'
    paj += '*edges'
    paj += '\n'
    for edge in edges:
        paj += str(edge[0]) + ' ' + str(edge[1]) + ' 1'  # weight 
        paj += '\n'
      
    return paj 