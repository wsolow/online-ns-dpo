import os
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
from .non_stationary_datasets import *


def filter_two_countries(
    dataset,
    country1,
    country2
):
    df_n0 = dataset[~dataset[country1].isnull() & ~dataset[country2].isnull()]
    return df_n0[
        [
            "question",
            "options",
            country1,
            country2,
            "time",
            "pref_label_" + country1,
            "pref_label_" + country2,
        ]
    ]

def ns_schedule_midchange(
    dataset,
    country1,
    country2,
    num_timesteps=10,
    coef_shift=1.,
    threshold_low=0.01,
    threshold_high=0.99
):
    '''
        return the prob. dist.s and sampled preference labels 
            from interpolation between two countries.
        The schedule of interpolation:
            - time 1 ~ (num_timesteps / 3) : country 1's prob.dist.
            - time (num_timesteps / 3) ~ (2 * num_timesteps / 3):
                gradual transition from country 1 to country 2.
            - time (2 * num_timesteps / 3) ~ num_timesteps: country 2's prob.dist.
    '''

    th1 = num_timesteps // 3
    th2 = th1 * 2
    itv = th2 - th1

    prob1 = dataset[country1].to_numpy()
    prob2 = dataset[country2].to_numpy()
    q_indices = np.arange(dataset.shape[0])

    num_adjustments = 0
    
    res = list()
    for i in range(num_timesteps):
        df_temp = dataset.copy()
        df_temp["time"] = i + 1
        pref_labels = list()

        if i < th1:
            w_interpolate = 1. + (coef_shift - 1.) / 2
        elif (i >= th1 and i < th2):
            w_interpolate = 1. + ((coef_shift - 1.) / 2) - (coef_shift / itv) * (i - th1)
        else:
            w_interpolate = 0. - (coef_shift - 1.) / 2

        probs_i = list()
        for j in range(dataset.shape[0]):
            p0 = prob1[j][0] * w_interpolate + prob2[j][0] * (1 - w_interpolate)
            if p0 > threshold_high:
                p0 = threshold_high
                num_adjustments += 1
            elif p0 < threshold_low:
                p0 = threshold_low
                num_adjustments += 1
            p1 = 1 - p0
            probs_i.append((p0, p1))

        df_temp["probs"] = probs_i
        df_temp["index_q"] = q_indices
        df_temp.loc[:, "pref_label"] = df_temp["probs"].apply(create_preferences, sample=True)

        res.append(df_temp)

    print(f"[nsgo-2c] preference probability adjusted {num_adjustments} times / {dataset.shape[0] * num_timesteps} samples")
    return pd.concat(res).reset_index().drop(["index"], axis=1)

def create_ns_dataset_midchange(
    dataset,
    country1,
    country2,
    num_timesteps=10,
    coef_shift=1.,
    threshold_low=0.01,
    threshold_high=0.99,
    min_diff=0.3
):
    dataset = filter_two_countries(
        dataset,
        country1,
        country2,
    )

    # filter out responses with (0.5, 0.5) prob
    # filter out responses with extreme probabilities, probably one of the responses was never picked
    v_extreme_low = 0.01
    v_extreme_high = 0.99
    dataset = dataset[
        ((dataset[country1] != (0.5, 0.5)) & (dataset[country2] != (0.5, 0.5))) &
        (abs(dataset[country1].str[0] - dataset[country2].str[0]) > min_diff)
    ]

    dataset = ns_schedule_midchange(
        dataset,
        country1,
        country2,
        num_timesteps=num_timesteps,
        coef_shift=coef_shift,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )

    return dataset

def process_to_prompt_key_form_2countries(df_dict: dict, countries: list):

    output_dict = dict()
    
    for prompt, data in df_dict.items():
        output_inner_dict = defaultdict(list)
        
        no_pref = False
        output_inner_dict['responses'] = data['options']
        output_inner_dict['timestep'] = data['time']
        
        #If preference 1.0 then the first option is preferred:
        #pairs used as: responses[p[0]], responses[p[1]] in get_batch_iterator method
        pref_label = data[f'pref_label']
        
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
            #For now until we understand this better        
            output_inner_dict['sft_target'] = sft_target
            output_dict[prompt + '. Countries: ' + str(countries)] = output_inner_dict
    
    return output_dict

def get_nsgo_2countries(
    split:str, 
    silent:bool=False, 
    cache_dir:str=None,
    force_new:bool=False,
    country1:str="United States",
    country2:str="Japan",
    timesteps:int=100,
    sample_to_size:int=50000,
    seed=42,
    coef_shift=1.,
    threshold_low=0.01,
    threshold_high=0.99,
    min_diff=0.3,
    **kwargs
):
    
    """
    Converts dataset to this format:
    {
        'prompt1': {
            'responses': List[str],
            'pairs': List[Tuple[int, int]],
            'sft_target': str
        },
        'prompt2': {
            ...
        },
    }
    """
    
    #Setup our own cache for the nsgo dataset:
    created_cache_dir = '.cache' if cache_dir is None else cache_dir
    cache_path = os.path.join(
        created_cache_dir, 
        f'nsgo_2c_dataset_{country1}_{country2}_t{timesteps}_size{sample_to_size}_seed{seed}_coef{coef_shift}_low{threshold_low}_high{threshold_high}_mindiff{min_diff}'
    )
    train_path = os.path.join(cache_path, 'train.pkl')
    test_path = os.path.join(cache_path, 'test.pkl')
    
    if not os.path.exists(cache_path): #Create the dataset if it doesn't exist
    
        countries = COUNTRIES
        
        dataset = datasets.load_dataset('Anthropic/llm_global_opinions', 
                                        cache_dir=cache_dir)["train"]
        
        dataset = pd.DataFrame(dataset)

        #filter out question = None:
        dataset = dataset[~dataset['question'].isnull()]
        
        #Reformat the raw options and selections column data
        print('Reformatting nsgo option and selection columns')
        dataset['options'] = process_options(dataset['options'])    
        dataset['num_options'] = dataset['options'].apply(lambda x: len(x))
        selections = process_selections(dataset['selections'], countries)
        
        #Combine these processed columns back into the dataset
        dataset = pd.concat([dataset[['question', 'options',
                                      'num_options', 'source']], 
                             selections], axis=1)

        ########################## FILTER DATASET #############################
        #To avoid short response questions we only select questions with wordy responses
        dataset['len_options'] = dataset['options'].apply(lambda x: np.mean([len(str(i)) for i in x]))
        long_opt_questions = dataset.loc[dataset['len_options'] > 16, 'question'].unique()
        long_opt_filter = dataset['question'].isin(long_opt_questions)
        dataset = dataset[long_opt_filter]
        
        #Process the options and country probs to a binary setup
        print('Processing preferences to binary preferences')
        option_and_countries = process_to_binary_preferences(dataset, countries)
        
        #Merge the question with the options and processed pref probabilities
        dataset = pd.merge(dataset[['question']], 
                            option_and_countries,
                            left_index=True, right_index=True).\
                        reset_index(drop=True)
        
        #Here we filter the dataset, create preference labels and re-norm distributions
        print('Filtering, cleaning, creating preference labels and re-normalising probs')
        dataset = clean_base_dataset(dataset, countries)    
        
        # Apply non-stationarity schedule
        np.random.seed(seed=seed)
        print('Creating non-stationary dataset') 
        dataset["time"] = 1
        dataset = create_ns_dataset_midchange(
            dataset,
            country1,
            country2,
            num_timesteps=timesteps,
            coef_shift=coef_shift,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_diff=min_diff
        )
        train_qs, test_qs = sample_train_test_questions(dataset['question'])
        print(f"created ns dataset: {dataset.shape}")
        
        #Separate into train and test via the timestep:
        if timesteps == 0:
            df_train = dataset[dataset['question'].isin(train_qs)]
            df_test = dataset[dataset['question'].isin(test_qs)]    
        else:
            final_ts = dataset['time'].max()
            df_train = dataset[(dataset['time'] < final_ts) &\
                            (dataset['question'].isin(train_qs))]
            df_test = dataset[(dataset['time'] == final_ts) &\
                            (dataset['question'].isin(test_qs))]

        countries_selected = [country1, country2]
        df_train = process_into_dict(df_train)
        df_train = process_to_prompt_key_form_2countries(df_train, countries_selected)

        #random sampling step here to properly reduce the dataset size:
        keys = list(df_train.keys())
        
        if len(keys) < sample_to_size: sample_to_size = len(keys)        
        
        sampled_keys = np.random.choice(keys, replace=False, size=sample_to_size)
        df_train_out = {key: df_train[key] for key in sampled_keys}
        df_train = df_train_out

        df_test = process_into_dict(df_test)
        df_test = process_to_prompt_key_form_2countries(df_test, countries_selected)
        
        #Create path and save train and test set to path
        os.makedirs(cache_path, exist_ok=True)
        
        print(f'saving dataset to {cache_path}')
        with open(train_path, 'wb') as f:
            pickle.dump(df_train, f)
        with open(test_path, 'wb') as f:
            pickle.dump(df_test, f)
    
    else:    
        #Load the train and test set from the path...
        print(f'loading dataset from {cache_path}')
        with open(train_path, 'rb') as f:
            df_train = pickle.load(f)
        with open(test_path, 'rb') as f:
            df_test = pickle.load(f)
            
    #Return the correct split:
    if split == 'train':
        output = df_train
        print(f"[train] {len(output)} points present")
    elif split == 'test':
        output = df_test
        print(f"[test] {len(output)} points present")
    else:
        raise NotImplementedError(f'get_nsgo_2countries split type: {split} not implemented')
    
    return output 

if __name__ == "__main__":
    df_result = get_nsgo_2countries(
        "test", 
        "Germany",
        "United States",
        num_timesteps=101,
    )
    print(len(df_result.values()))
