#import pickle
import ast
import regex as re
from scipy.special import rel_entr
#from scipy.spatial.distance import jensenshannon


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
def KL_Divergence(model_distance, target_distance):
    """(parameter):
    model_marriage_distance: [list of integers]
                           : the list of the number of our model's distance
    target_marriage_distance: [list of integers]
                           : the list of the number of the target, data distance
       (return):
    JensenShannon: [float between 0 to 1]
                 : the half of the sum of KL KL_Divergence of X and Y and Y and X since
                   those are not symmetric."""

    #find the highest number of distance to get the highest key for the dictionary
    max_key = max(set(model_distance+target_distance))

    #setting the dictionaries from 0 to the highest key number
    #Then setting all the dictionary values to 0 to reduce the big(O)
    m_dict = dict.fromkeys(range(max_key+1), 0)
    t_dict = dict.fromkeys(range(max_key+1), 0)

    #getting the probability for the model and the target
    for m in (sorted(set(model_distance))):
      m_dict[m] = model_distance.count(m)/len(model_distance)
    for t in (sorted(set(target_distance))):
      t_dict[t] = target_distance.count(t)/len(target_distance)

    #convert the probabilities into a list*(numpy array causes an error)
    # [-1] entry corresponds to infite distance marriage
    target = list(t_dict.values())
    model = list(m_dict.values())

    #adding 1s to prevent divide by zero errors since KL_divergence requires division
    target[:] = map(lambda x: x+1, target)
    model[:] = map(lambda x: x+1, model)

    # renormilze to obtain probabilties again
    denominator_target = sum(target)
    target = [k/denominator_target for k in target]
    denominator_model = sum(model)
    model = [k/denominator_model for k in model]

    #jensenshannon- (KL(X||Y)+KL(Y||X))/2 without using scipy.jensenshannon
    JensenShannon = (sum(rel_entr(target,model))+sum(rel_entr(model,target)))/2

    return JensenShannon

#%%
# # example to load model marriage distribution
# with open('output\\tikopia_1930_1\\tikopia_1930_marriage_distances.pkl', 'rb') as infile:
#     model_distance = pickle.load(infile)
# # example to load target marriage distribution
# name = 'tikopia_1930'
# target_distance, num_marriages, prob_inf_marriage, prob_finite_marriage, children_dist, num_people = get_graph_stats(name)

# #%%

# kl = KL_Divergence(model_distance, target_distance)

# #%%
# with open('output\\tikopia_1930_6\\tikopia_1930_children_per_couple.pkl', 'rb') as infile:
#     model_distance = pickle.load(infile)
# childkl = KL_Divergence(model_distance, children_dist)
############################################################################################################################
#%%
# def test_model_using_KL_Divergence():
#     """Get all the names of tribes that we have and find the descrepancies between the target output and the model output.
#     """
#     #to make a full path such as: 'output\\tikopia_1930_1\\tikopia_1930_marriage_distances.pkl'
#     #cut the first part: 'output\\tikopia_1930_1\\ and the last part: _marriage_distances.pkl'
#     #and make the full path by chaning only the name of the tribe
#     firstPart = "output\\tikopia_1930_1\\"
#     lastPart = "_marriage_distances.pkl"

#     #setting a returning descrepancies list
#     Jensen_value = []

#     #get the each name to compare in the folder
#     for name in the folder:
#         #combine the firstPart + name + lastPart to make a full path
#         full_path = firstPart+str(name)+lastPart
#         #get the output with the full path
#         with open(full_path, 'rb') as infile:
#             model_distance = pickle.load(infile)
#         #originally target_distace contains 6 kind of data as below but cut it to look nice
#         # target_distance, num_marriages, prob_inf_marriage, prob_finite_marriage, children_dist, num_people
#         #maybe we should cut the returnn values?
#         target_distance = get_graph_stats(name)
#         #get the descrepancies betweenn the data and the model
#         Jensen_value.append(KL_Divergence(model_distance, target_distance[0]))
#     return Jensen_value
