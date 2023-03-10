'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random

def get_entropy_of_dataset(df):
    entropy = 0
    column_val = df[[df.columns[-1]]].values
    a, unique_count = np.unique(column_val, return_counts = True)
    no_of_instances = len(column_val)
    if no_of_instances <= 1:
        return 0

    probs_array = []
    for i in range(0, len(unique_count)):
        probs = unique_count[i]/no_of_instances 
        probs_array.append(probs)

    for probabilities in probs_array:
        if(probabilities!=0):
            entropy = entropy - (probabilities * np.log2(probabilities))
    return entropy


def get_avg_info_of_attribute(df, attribute):
    avg_info = 0
    attribute_values = df[attribute].values
    unique_attribute_values,unique_attribute_array = np.unique(attribute_values,return_counts = True)
    no_of_instances = len(attribute_values)

    for attribute_value in unique_attribute_values:
        sliced_dataframe = df[df[attribute] == attribute_value]
        instances = sliced_dataframe[[sliced_dataframe.columns[-1]]].values
        instances_unique_values,instances_unique_counts = np.unique(instances, return_counts = True)
        total_count_in_an_instance = len(instances)

        S_of_attribute_value = 0
        for i in instances_unique_counts:
            j = i/total_count_in_an_instance
            if j != 0:
                S_of_attribute_value = S_of_attribute_value - (j*np.log2(j))
        avg_info = avg_info + S_of_attribute_value*(total_count_in_an_instance/no_of_instances)
    return(abs(avg_info))
    
def get_information_gain(df, attribute):
    information_gain = 0
    entropy_of_dataset = get_entropy_of_dataset(df)
    entropy_of_attribute = get_avg_info_of_attribute(df, attribute)
    information_gain = entropy_of_dataset - entropy_of_attribute
    return information_gain

def get_selected_attribute(df):
    max_information_gain = 0
    information_gain_of_all_attributes = {}
    selected_attribute = ''

    for attribute in df.columns[:-1]:
        information_gain_of_an_attribute = get_information_gain(df, attribute)
        if information_gain_of_an_attribute > max_information_gain:
            max_information_gain = information_gain_of_an_attribute
            selected_attribute = attribute
        information_gain_of_all_attributes[attribute] = information_gain_of_an_attribute
    return (information_gain_of_all_attributes, selected_attribute)
    
