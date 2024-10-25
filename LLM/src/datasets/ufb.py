import os, sys
import re
import ast
import math
import json
import pickle
import datasets
import transformers
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from rms import *

sys.path.insert(0, "../../")
sys.path.insert(0, "../")

def polish_responses(
    df
):
    df["chosen"] = df["chosen"].apply(lambda x: x[1]["content"])
    df["rejected"] = df["rejected"].apply(lambda x: x[1]["content"])
    return df

def clean_ufb(
    df
):
    # Remove rows with blank strings in one of prompt, chosen, rejected
    df = df[(df.prompt != "") & (df.chosen != "") & (df.rejected != "")]

    return df
    
def subsample_ufb(
    dataset,
    seed=13,
    size_subsample=10000,
):
    dataset = dataset.dropna()
    np.random.seed(seed)
    len_df = dataset.shape[0]
    idxs = np.random.choice(len_df, size_subsample)

    df = polish_responses(dataset.iloc[idxs])
    
    return df

def sample_preference(logits):
    probs = 1 / (1 + np.exp(-logits))
    output = np.random.binomial(1, probs)
    return output

def ufb_process_into_dict(dataset:pd.DataFrame):

    #Create the output dictionary
    question_dict = dict()
    output_dict = defaultdict(list)

    #Convert the dataframe into a list of dict types    
    df_dict = dataset.dropna().replace({np.nan: None}).to_dict('records')
    
    for datapoint in df_dict:
        
        try:
            #Set the question as the key and process the dict:
            question = datapoint['prompt'] + f"| Time step: {datapoint['time']} |"
            if question not in question_dict:
                question_dict[question] = 1
            else:
                question_dict[question] = question_dict[question] + 1
            question += f" STRIP THIS AWAY FOR VAR {question_dict[question]} STRIP THIS AWAY FOR VAR "
            output_dict[question] = datapoint
        except:
            print('Unable to append time value to input prompt on datapoint:')
            print(datapoint)
        
    return output_dict

def ufb_into_dict(
    dataset:pd.DataFrame,
    col_pref
):
    df_dict = ufb_process_into_dict(dataset)
    print(df_dict)

    output_dict = dict()
    
    for prompt, data in df_dict.items():
        output_inner_dict = defaultdict(list)
        no_pref = False
        
        output_inner_dict['responses'] = [data['chosen'], data['rejected']]
        output_inner_dict['timestep'] = data['time']
        
        #If preference 1.0 then the first option is preferred:
        #pairs used as: responses[p[0]], responses[p[1]] in get_batch_iterator method
        pref_label = data[col_pref]
        
        if pref_label == 1.0:
            output_inner_dict['pairs'] = [(0, 1)]
            sft_target = output_inner_dict['responses'][0]
        elif pref_label == 0.0:
            output_inner_dict['pairs'] = [(1, 0)]
            sft_target = output_inner_dict['responses'][1]
        elif pref_label is None:
            #The pref label is None
            no_pref = True
        else:
            raise NotImplementedError(
                f'Pref label: {pref_label} type not implemented')
    
        if no_pref == False:
            output_inner_dict['sft_target'] = sft_target
            output_dict[prompt] = output_inner_dict
    
    return output_dict

def create_ufb_2rm_dataset(
    dataset_train:pd.DataFrame,
    dataset_test:pd.DataFrame,
    timesteps:int,
    changepoint:int,
    rm1:str="pairrm",
    rm2:str="betterpairRM",
    sample=True,
):
    size_timestep = dataset_train.shape[0] // timesteps
    if dataset_train.shape[0] % size_timestep > 0:
        size_timestep += 1
    dataset_train["time"] = 1
    dataset_test["time"] = timesteps + 1
    dataset_train["sampled_pref"] = 0.
    dataset_test["sampled_pref"] = 0.

    for t in range(timesteps):
        idx_start = t * size_timestep
        df_current = dataset_train.iloc[idx_start:idx_start + size_timestep]
        df_current["time"] = t+1
        
        if t < changepoint: 
            # apply earlier RM
            if sample:
                df_current["sampled_pref"] = sample_preference(df_current[f"logits_{rm1}"])
            else:
                df_current["sampled_pref"] = df_current[f"prefs_{rm1}"]
                # df_current["sampled_pref"] = (df_current[f"logits_{rm1}"] > 0) * 1.0
        else: 
            # apply latter RM
            if sample:
                df_current["sampled_pref"] = sample_preference(df_current[f"logits_{rm2}"])
            else:
                df_current["sampled_pref"] = df_current[f"prefs_{rm2}"]
                # df_current["sampled_pref"] = (df_current[f"logits_{rm2}"] > 0) * 1.0
        dataset_train.iloc[idx_start:idx_start + size_timestep] = df_current

    if sample:
        dataset_test["sampled_pref"] = sample_preference(dataset_test[f"logits_{rm2}"])
    else:
        dataset_test["sampled_pref"] = dataset_test[f"prefs_{rm2}"]

    dict_train = ufb_into_dict(dataset_train, "sampled_pref")
    dict_test = ufb_into_dict(dataset_test, "sampled_pref")

    return dict_train, dict_test, dataset_train, dataset_test

def get_varied_alignment(
    df,
    col1,
    col2,
    num_total,
    num_aligned,
):
    df_aligned = df[df[col1] == df[col2]]
    df_naligned = df[df[col1] != df[col2]]
    print(df.shape, df_aligned.shape, df_naligned.shape)

    num_naligned = num_total - num_aligned

    df_ta = df_aligned.sample(n=num_aligned)
    df_tna = df_naligned.sample(n=num_naligned)
    df_res = pd.concat([df_ta, df_tna]).sample(n=num_total)

    print(df_res.shape)

    return df_res