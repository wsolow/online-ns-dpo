# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:57:09 2024

@author: William
"""

import os
import re
import ast
import math
import json
import pickle
import random
import datasets
import transformers
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from transformers import pipeline
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
              
COUNTRIES=[
    'Nigeria', 
    'Egypt', 
    'India (Current national sample)', 
    'China', 
    'Japan', 
    'Germany', 
    'France', 
    'Spain', 
    'United States', 
    'Canada', 
    'Brazil', 
    'Argentina', 
    'Australia', 
    'New Zealand'
]

def process_options(options:pd.Series):
    
    def process_str_list(str_list:str) -> List:
        
        #Remove open and close brackets:
        str_list = str_list[1:-1]
        
        #Split on commas:
        str_list = re.split(r",(?=')", str_list)
                                
        #For each element remove opening quotes:
        return [el.strip() for el in str_list]

    return options.apply(lambda x: ast.literal_eval(x))

def process_selections(selections: pd.Series, country_filter:list):
    
    def process_str_dict(str_dict:dict) -> pd.DataFrame:
        
        #Remove the defaultdict list term:
        str_dict = str_dict[28:-1]
        
        #Prepare str to be parsed into a dict type:
        str_dict = str_dict.replace("'", '"')
        str_dict = str_dict.replace("[", "[[").replace("]", "]]")
        
        return str_dict

    indices = selections.index

    selections = selections.apply(process_str_dict).values
    
    #For each element of the dataset create a dataframe:
    df = pd.concat([pd.DataFrame(json.loads(el)) for el in selections])
    
    #Remove unwanted countries
    df = df[country_filter].set_index(indices)
    
    return df

def process_to_binary_preferences(df:pd.DataFrame, countries:list):
        
    def process_to_binary_preference(opt, num_opt):
        
        if hasattr(opt, '__iter__'):
            output = [el for el in itertools.combinations(opt, 2)]
        else:
            num_combs = math.comb(num_opt, 2)
            output = ['']*num_combs
            
        return output
    
    #Process the options column:
    df['num_comb'] = df['num_options'].apply(lambda x: math.comb(x, 2))
    
    
    #Process the options and each country:
    options = df.apply(lambda x: process_to_binary_preference(
                        x['options'], x['num_options']), axis=1).explode()    

    df_countries = [df.apply(lambda x: process_to_binary_preference(
                        x[country], x['num_options']), axis=1).explode()\
            for country in countries]
    
    #Ensure dataframes the same size:
    for country in df_countries:
        assert len(country) == len(options),\
            f'length {len(country)} should be {len(options)}'
            
        #Convert '' to None type:
        country[country == ''] = None
              
    #Concat the dataset together
    output = pd.concat([options] + df_countries, axis=1)
    output.columns = ['options'] + countries
        
    return output

def clean_base_dataset(dataset:pd.DataFrame, countries):
    
    #Add time column: probably will remove this

    for country in countries:
        dataset.loc[:, f'pref_label_{country}'] = dataset[country].apply(create_preferences)
        dataset.loc[:, country] = dataset[country].apply(normalise_probs)
        
    return dataset
    

def create_preferences(probs, sample:bool=False):
    
    #Preference is 1 if the first opinion is preferred
    
    if probs is None:
        
        output = None
    else: 
        log0, log1 = np.log(probs[0]+1e-6), np.log(probs[1]+1e-6)
        if sample:
            prob = 1 / (1 + np.exp(log1 - log0))       
            output = np.random.binomial(1, prob)
        else: #Deterministic preferences based on rewards:
            if log0 > log1:
                output = 1
            elif log1 > log0:
                output = 0
            else: #Random tie breaker setup:
                output = np.random.binomial(1, 0.5)
        
    return output

def normalise_probs(probs):
    
    if probs is None:
        output = None
        
    else:
        #calculate log space and binary prob:
        log1, log2 = np.log(probs[0]+1e-6), np.log(probs[1]+1e-6)
        prob = 1 / (1 + np.exp(log2 - log1))
        
        #Prob is in terms of 1 being preferred to 2
        output = (prob, 1-prob)
    
    return output

def lse(array:np.array):
    
    max_val = array.max()
    adjusted_array = array - max_val
    return max_val + np.log(np.exp(adjusted_array).sum())
    

def softmax(logits:np.array):

    Z = lse(logits)
    
    return np.exp(logits - Z)

def ar1_process_sample(probs, epsilon:float):
    """
    Method that adjusts the probabilities by sampling from an AR(1) process
    on the underlying rewards
    """
    
    if probs is None:
        output = None
    
    else:
        x = np.random.uniform(0,1,size=2)
        
        #take the preferences and convert to rewards:
        log0, log1 = np.log(probs[0]+1e-6), np.log(probs[1]+1e-6)
        
        log0 = (1-epsilon)*log0 + epsilon*x[0]
        
        log1 = (1-epsilon)*log1 + epsilon*x[1]
        
        prob = 1 / (1 + np.exp(log1 - log0))
        output = (prob, 1 - prob)
    
    return output 

def ar1_process_sample_general(probs:np.array, target:np.array,
                               time:int, epsilon:float):
    
    """
    Method that adjusts the probabilities by sampling from an AR(1) process
    on the underlying rewards
    """
    
    if probs is None or probs is np.nan:
        output = None
    
    else:
    
        probs = np.array(probs)
        target = np.array(target)
                        
        x = np.random.dirichlet(alpha=target+1e-6, size=(1))      
        reward = np.log(probs + 1e-6)
        
        #take the preferences and convert to rewards:
        reward = (1 - epsilon)*reward + epsilon*x
        output = softmax(reward)[0]
    
    return output

    
def apply_sent_analysis(model, question, options):
    
    sent_output = [model(
        'When asked: ' + question + ' Response: ' + option)[0] for option in options]
    
    return sent_output

def create_target_from_score_and_trend(scores, trend, temp:float=1):
    
    #scores are a list of two sets of scores:
    outputs = list()
    for score in scores:
        for entry in score:
            if trend == 0 and entry['label'] == 'positive':
                outputs.append(entry['score'])
            elif trend == 1 and entry['label'] == 'negative':
                outputs.append(entry['score'])
            elif trend in (0,1) or entry['label'] in ('positive', 'negative', 'neutral'):
                continue
            else:
                raise NotImplementedError()
    return softmax(np.array(outputs)*temp)

def change_point_switch(pref_label):
    
    if pref_label is None:
        return None
    else:
        return (pref_label[1], pref_label[0])
        
   
def create_non_stationary_dataset(dataset:pd.DataFrame,
                                  trends:np.array,
                                  timesteps:int,
                                  epsilon: float,
                                  countries: list,
                                  temp: float=1,
                                  changepoint:Union[float, None] = None,
                                  start_time: int = 0):
    
    dataset['time'] = timesteps
    df_timesteps = [dataset]
    
    #Download sentiment model for the pipeline:
    sentiment_model = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True)

    if (timesteps > 0) and changepoint is None:
        #Calculate sentiment scores for each option
        print('Running Sentiment model ...')
        dataset['scores'] = dataset[['question', 'options']].apply(
            lambda x: apply_sent_analysis(sentiment_model, x.question, x.options), axis=1)
        
        for i, country in enumerate(countries):
            dataset[f'{country}_target'] = dataset.apply(
                lambda x: create_target_from_score_and_trend(x.scores, trends[i], 
                                                              temp=temp), axis=1)
        
    #For each time step vary the previous timesteps elements
    for t in range(timesteps - 1):
        
        df = df_timesteps[-1].copy(deep=True)
        df['time'] = timesteps - 1 - t #starting from 2 as current dataset is '1'
    
        #Change point shift in preference data:
            
        if changepoint is not None:
            if t == changepoint:
                
                for i, country in enumerate(countries):
                    df[country] = df_timesteps[-1][country].\
                        apply(lambda x: change_point_switch(x))
                    df[f'pref_label_{country}'] = df[country].apply(create_preferences)
             
        #Sentiment shift in change point dataset:
        elif (timesteps - 1 - t) <= start_time:
            for i, country in enumerate(countries):
                
                #Update the prior timesteps probabilities:
                df[country] = df_timesteps[-1][[country, f'{country}_target']].\
                    apply(lambda x: ar1_process_sample_general(x[country], x[f'{country}_target'],
                                                                timesteps, epsilon=epsilon), axis=1)
        
                #Update the preference labels based on the datasets probabilities:
                df[f'pref_label_{country}'] = df[country].apply(create_preferences)
            
        df_timesteps.append(df)
        
    return pd.concat(df_timesteps, axis=0).reset_index(drop=True)

def sample_train_test_questions(questions, split:int=100):
    
    qs = list(questions.unique())
    random.shuffle(qs)
    
    return qs[:-split], qs[-split:]

def remove_eos(sequence):
    pattern_eos = re.compile(f'</s>')
    sequence = pattern_eos.sub('</sequence>', sequence)
    return sequence

def remove_time_and_var_from_prompt(prompt, timestep, remove_timestep=True):
    #Create ts phrase:
    pattern_ts = re.compile(f'\| Time step: {timestep} \|')
    pattern_var = re.compile("(\s*)STRIP THIS AWAY FOR VAR [0-9]* STRIP THIS AWAY FOR VAR(\s*)")
    
    if remove_timestep:
        prompt = pattern_ts.sub('', prompt)
    output = pattern_var.sub('', prompt)
    
    return output    

def process_into_dict(dataset:pd.DataFrame):

    #Create the output dictionary
    question_dict = dict()
    output_dict = defaultdict(list)

    #Convert the dataframe into a list of dict types    
    df_dict = dataset.replace({np.nan: None}).to_dict('records')
    
    for datapoint in df_dict:
        
        try:
            #Set the question as the key and process the dict:
            question = datapoint['question'] + f"| Time step: {datapoint['time']} |"
            if question not in question_dict:
                question_dict[question] = 1
            else:
                question_dict[question] = question_dict[question] + 1
            question += f" STRIP THIS AWAY FOR VAR {question_dict[question]} STRIP THIS AWAY FOR VAR "
            output_dict[question] = datapoint
        except:
            print('Unable to append time value to question prompt on datapoint:')
            print(datapoint)
        
    return output_dict

def process_to_prompt_key_form(df_dict: dict, countries: list):

    output_dict = dict()
    
    for prompt, data in df_dict.items():
        for country in countries:
        
            output_inner_dict = defaultdict(list)
            
            no_pref = False
            output_inner_dict['responses'] = data['options']
            output_inner_dict['timestep'] = data['time']
            
            #If preference 1.0 then the first option is preferred:
            #pairs used as: responses[p[0]], responses[p[1]] in get_batch_iterator method
            pref_label = data[f'pref_label_{country}']
            
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
                
                if prompt[-1] in ['.', '?', '!']:
                    country_prompt = f' The survey country was {country}. '
                elif prompt[-1] in [',']:
                    country_prompt = f' the survey country was {country}. '
                else:
                    country_prompt = f'. The survey country was {country}. '
                
                output_dict[prompt + country_prompt] = output_inner_dict
    
    return output_dict

def create_series_change_label(srs):
    """
    Analyse srs, if the label changes across the series return 1 else return 0
    if entries are null return None

    Parameters
    ----------
    srs : pd.Series
        Series of preference labels across various timesteps

    Returns
    -------
    int
        Indicator if the preference label changes or not

    """
    
    if srs.isnull().any():
        return None
    else:
        if srs.iloc[0] != srs.iloc[-1]:
            return 1
        else:
            return 0
        
def only_changing_labels_in_test(pref_label, chng_label):
    
    if chng_label == 0:
        return None
    else:
        return pref_label


def get_nsgo(split:str, silent:bool = False,
             cache_dir:str=None, force_new:bool=False,
             timesteps:int=5, epsilon:float=0.3, temp:int=10,
             sample_to_size:int=50000, countries:Union[list, None]=None,
             final_ts_stationary:bool=False, changepoint:Union[int, None]=None,
             start_time:int=2, only_changing_in_test:bool=False, **kwargs):
    
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
    cache_path = os.path.join(created_cache_dir, 
                              f'nsgo_dataset_{timesteps}_{epsilon}_{temp}_{sample_to_size}_{changepoint}_{start_time}')
    train_path = os.path.join(cache_path, 'train.pkl')
    test_path = os.path.join(cache_path, 'test.pkl')
    
    if (not os.path.exists(cache_path)) or (force_new == True):
    
        #Run main script:
        if countries is None:
            countries=COUNTRIES
            
        print('Countries are', countries)
                
        ############################ LOAD DATA ################################
        #Load dataset (only has train set):
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
                    
        ############## CREATE BINARY OPTIONS FROM MULTI-OPTIONS ###############
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
        
        ################### CREATE NON STATIONARY DATASET #####################
        #Assign countries positive or negative preferences:
        np.random.seed(seed=42)
        country_trends = np.random.binomial(size=len(countries), n=1, p=0.5)
                
        #Apply an AR(1) process to the dataset
        print(f'Creating non-stationary dataset with timesteps {timesteps}, epsilon {epsilon} and temp {temp}')
        dataset = create_non_stationary_dataset(dataset=dataset,
                                                trends=country_trends, 
                                                timesteps=timesteps,
                                                epsilon=epsilon,
                                                countries=countries,
                                                temp=temp,
                                                changepoint=changepoint,
                                                start_time=start_time) 
             
        
        #Group the dataset 
        
        unique_qs = list(dataset['question'].unique())        
        split_ratio = int(0.2 * len(unique_qs))
        train_qs, test_qs = sample_train_test_questions(dataset['question'], split=split_ratio)
        
        #Process and analyse the dataset right here
        ################### SPLIT INTO TRAIN AND TEST DATASET #################
        #Separate into train and test via the timestep and prompt:
        
        if timesteps == 0:
            df_train = dataset[dataset['question'].isin(train_qs)]
            df_test = dataset[dataset['question'].isin(test_qs)]
            
        elif final_ts_stationary:
            
            final_ts = dataset['time'].max()
            df_train = dataset[(dataset['time'] == final_ts) &\
                               (dataset['question'].isin(train_qs))]
            df_test = dataset[(dataset['time'] == final_ts) &\
                               (dataset['question'].isin(test_qs))]
            
        else:
        
            final_ts = dataset['time'].max()
            df_train = dataset[(dataset['time'] < final_ts) &\
                               (dataset['question'].isin(train_qs))]
                            
            if only_changing_in_test:
                                
                df_test = dataset[dataset['question'].isin(test_qs)]
                
                agg_dict = dict()
                rename_dict = dict()
                for country in countries:
                    rename_dict[f'pref_label_{country}'] = f'chg_label_{country}'
                    agg_dict[f'pref_label_{country}'] = create_series_change_label
                    
                grp = df_test.groupby(['question', 'options']).\
                        agg(agg_dict).\
                            rename(rename_dict, axis='columns').\
                                reset_index()
                df_test_chng = pd.merge(df_test, grp, on=['question', 'options'])
                
                #If there is no                 
                for country in countries:
                    df_test_chng[f'pref_label_{country}'] = df_test_chng.\
                        apply(lambda x : only_changing_labels_in_test(
                            x[f'pref_label_{country}'], 
                            x[f'chg_label_{country}']), axis=1)    

                #Finally filter to only the final timesteps
                df_test = df_test_chng[df_test_chng['time'] == final_ts]

            else:

                df_test = dataset[(dataset['time'] == final_ts) &\
                                   (dataset['question'].isin(test_qs))]
                
                

        print(f'Unique prompts in train dataset: {len(np.unique(df_train["question"]))}')
        
        #These are so much smaller because we get rid of all the nans...
        df_train = process_into_dict(df_train)
        df_train = process_to_prompt_key_form(df_train, countries)
        
        #random sampling step here to properly reduce the dataset size:
        keys = list(df_train.keys())
        
        if len(keys) < sample_to_size: sample_to_size = len(keys)        
        
        sampled_keys = np.random.choice(keys, replace=False, size=sample_to_size)
        df_train_out = {key: df_train[key] for key in sampled_keys}
        df_train = df_train_out
        
        print(f'Unique prompts in test dataset: {len(np.unique(df_test["question"]))}')
        
        df_test = process_into_dict(df_test)
        df_test = process_to_prompt_key_form(df_test, countries)
        
        #Create path and save train and test set to path
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        
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
    elif split == 'test':
        output = df_test
    else:
        raise NotImplementedError(f'get_nsgo split type: {split} not implemented')
        
    return output

def get_ufb_2rm(
    split,
    path_train=None,
    path_test=None,
    **kwargs
):
    
    #Return the correct split:
    if split == 'train':
        print(f'loading dataset from {path_train}')
        with open(path_train, 'rb') as f:
            df_train = pickle.load(f)
        output = df_train
    elif split == 'test':
        print(f'loading dataset from {path_test}')
        with open(path_test, 'rb') as f:
            df_test = pickle.load(f)
        output = df_test
    else:
        raise NotImplementedError(f'get_ufb_2rm split type: {split} not implemented')
        
    return output
    