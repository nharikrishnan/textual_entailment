"""This file consists of the functions that are required to convert the predicted output in the required format"""

#import statements
import pandas as pd
import numpy as np

# converting the predicted probabilities to class
def convert_prob(pred_values):
    """ Function to convert the predicted probabilities to class encoded values
            Args:
                k: predicted values"""
    
    result = []
    for i in pred_values: 
        if (i[0]> i[1]) & (i[0]>i[2]):
            result.append(0)
        elif (i[1]>i[0]) & (i[1]> i[2]):
            result.append(1)
        else:
            result.append(2)
    return result

#function to predicted entaiment_encoded values to the entailment_function
def map_entailment(lbl_encoder, result, test, cols=['pair_ID']):  
    """function which maps the predicted entailment enocded values entailment_judgmen.
       returns the data frame with the pair_ID an entailment_judgment"""
    entailment_classes = lbl_encoder.classes_
    
    entailment_result = []
    
    for i in result:
        if i ==0:
            entailment_result.append(entailment_classes[0])
        elif i ==1:
            entailment_result.append(entailment_classes[1])
        else:
            entailment_result.append(entailment_classes[2])
    
    df = test[cols]
    
    df['entailment_judgment'] =  entailment_result
    
    return df

#map the predicted relatedness score
def map_relatedness(test, relatedness_result, cols = ['pair_ID', 'entailment_judgment']):
    df= test[cols]
    df['relatedness_score']= relatedness_result
    return df

# write csv
def output_csv(df, file_name, index = False):
    df.to_csv(file_name, index = index)