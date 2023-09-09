# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:00:13 2023

@author: kolto
"""

from family_model import * 
from get_model_parameters import * 
import pandas as pd
import os

names = get_names(save_paths=False)
names.remove('warao')

mylist = []
for name in names: 
    marriage_dist, num_marriages, prob_inf_marriage, prob_finite_marriage, child_dist, num_people = get_graph_stats(name)
    # df = pd.concat([df, pd.Series({'name': name,
    #                                'num_marriages': num_marriages,
    #                                'prob_inf_marriage': prob_inf_marriage,
    #                                'prob_finite_marriage': prob_finite_marriage,
    #                                'num_people': num_people
    #                                }).T], 
    #                ignore_index=True,
    #                axis=1)
    mylist.append({'name': name,
                                    'num_marriages': num_marriages,
                                    'prob_inf_marriage': prob_inf_marriage,
                                    'prob_finite_marriage': prob_finite_marriage,
                                    'num_people': num_people
                                    })

df = pd.DataFrame(mylist)

if not os.path.exists('data_set_stats'):
    os.makedirs('data_set_stats')
with pd.ExcelWriter(os.path.join('data_set_stats', 'original_data_set_stats.xlsx'), engine='xlsxwriter') as writer:
    df.to_excel(writer)




