from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import get_model_parameters as gmp

def calculatePseudoGen(g, node):
    """Calculates the generation number of a node 
    (which is defined to be the length of the longest path that only goes up the tree)"""
    
    def bfs_layers(G, source):   # Similar to code from networkX; it is a generator that yields candidates at a specified distance
        current_layer = [source]
        visited = set(current_layer)

        # this is basically BFS, except that the current layer only stores the nodes at same distance from sources at each iteration
        while current_layer:
            yield current_layer
            next_layer = []
            for node in current_layer:
                parents = [par for par in G[node] if G[node][par]["Relationship"] != "Marriage"] 
                for parent in parents:
                    if parent not in visited:
                        # visited.add(parent)     # Don't comment out if we only want the shortest path between the farthest ancestor
                        next_layer.append(parent)
            current_layer = next_layer

    bfs_generator = bfs_layers(g, node)
    for i, layer in enumerate(bfs_generator): pass # Call generator until it no longer finds paths
    return i
    
def getPseudoGen(g):
    """
    Returns a dictionary that maps the nodes to their pseudoGeneration Number
    
    Param:
        g(nx.graph): A directed graph that only includes edges pointing from parents to children
    """    
    g_reversed = g.reverse() # Reverse the order so that the edges point from child to parent
    node_to_gen = {}
    for node in g_reversed:
        node_to_gen[node] = calculatePseudoGen(g_reversed, node)

    print('node_to_gen:', node_to_gen)
    # for node, gen in node_to_gen.items():
    #     if gen == 0:
    #         print(node)
    return node_to_gen

def plotPseudoGen(name, g):
    """
    Plots a bar graph to measure the distribution of pseudo generation numbers

    Args:
        name(str): Name of the model
        g(nx.graph): A directed graph that only includes edges pointing from parents to children
    """
    node_to_gen = getPseudoGen(g)
    distances = list(node_to_gen.values())

    x = np.arange(max(distances) + 1)
    y = np.zeros(max(distances) + 1)
    for i in range(len(distances)):
        y[distances[i]] += 1

    plt.bar(x, y)
    plt.xlabel("Pseudo-Generation Number")
    plt.ylabel("Number of people")
    plt.title(name + " Data Set")
    # plt.show()

def plotChildrenByGen(name, g):
    """
    Plots a bar graph to measure how the children distribution changes with respect to the pseudoGeneration number

    Args:
        g(nx.graph): A directed graph that only includes edges pointing from parents to children
    """
    
    # Get a dictionary the maps the names to the graphs (this time not directed so that it includes marriage edges)
    stuff = gmp.get_graphs_and_names(directed=False)
    graphs = stuff[0]
    bigNames = stuff[1]
    smallNames = []
    for the_name in bigNames:
        smallNames.append(the_name[28:-13])    
    data = dict(map(lambda i,j : (i,j) , smallNames, graphs))
    G = data[name]
    
    node_to_gen = getPseudoGen(g)
    largest_pseudo_gens = max(set(node_to_gen.values()))   # Largest possible pseudo-generation numbers
    pseudo_gens = np.arange(largest_pseudo_gens + 1)
    pseudo_gen_to_children = {}    # Maps the pseudo generation number to a list of all the number of childrens corresponding to it
    for i in pseudo_gens:
        pseudo_gen_to_children[i] = []
    
    alreadyVisited = set()
    for node in g:
        if node in alreadyVisited: continue   # No reason to repeat the distribution for both parents
        
        children = set(g[node])
        pseudo_gen = node_to_gen[node]
        
        spouses = [par for par in G[node] if G[node][par]["Relationship"] == "Marriage"] 
        for spouse in spouses:
            spouses_children = set(g[spouse])
            if len(children) == 0: num_children = 0
            else:  num_children = len(children.union(spouses_children))   # Only account for the children they share with that spouse
            pseudo_gen_to_children[pseudo_gen].append(num_children)
            alreadyVisited.add(spouse)
        alreadyVisited.add(node)
        
    # Determine mean number of children for each generation number
    expected_num_children = np.zeros(largest_pseudo_gens + 1)
    for i in pseudo_gens:
        if len(pseudo_gen_to_children[i]) == 0: expected_num_children[i] = 0
        else: expected_num_children[i] = np.mean(pseudo_gen_to_children[i])

    plt.bar(pseudo_gens, expected_num_children)
    plt.xlabel("Pseudo-Generation Number")
    plt.ylabel("Expected Number of Children")
    plt.title(name + " Data Set")
    plt.show()
        


# Get a dictionary the maps the names to the graphs
stuff = gmp.get_graphs_and_names(directed=True)
graphs = stuff[0]
bigNames = stuff[1]
smallNames = []
for name in bigNames:
    smallNames.append(name[28:-13])    
data = dict(map(lambda i,j : (i,j) , smallNames, graphs))

# Take an example graph for testing
# data_sets = [x[:-4] for x in os.listdir("Kolton_distances")]  # get from folder, excluding .txt part (last 4 characters)
# name = np.random.choice(data_sets)   # Get a random data set
# name = "tikopia_1930"
# name = "arawete"
# name = "dogon_boni"
name = "kelkummer"
target = data[name]


# Calculate the pseudo distances for each node
# node_to_gen = getPseudoGen(target)

# plotPseudoGen(name, target)
# plotChildrenByGen(name, target)
