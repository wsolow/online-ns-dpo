# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:47:09 2024

@author: William
"""


import os
import tqdm
import torch
import random
import pickle
import datasets
import pandas as pd
import numpy as np
from collections import defaultdict

from src.utils import rank0_print

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer)

from typing import (
    Dict,
    Tuple,
    List,
    Union,
    Optional)


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def split_prompt_and_responses(ex):
    prompt = extract_anthropic_prompt(ex['chosen'])
    chosen_response = ex['chosen'][len(prompt):]
    rejected_response = ex['rejected'][len(prompt):]
    return prompt, chosen_response, rejected_response

def sample_train_test_questions(unique_prompts, split:int=100):
    
    random.shuffle(unique_prompts)
    
    return unique_prompts[:-split], unique_prompts[-split:]

class output_struct:
    rewards = None

class test_model:
    
    def __init__(self):
        pass
    
    def __call__(self, _input):
        
        out = torch.randn((1,19))
        
        output = output_struct()
        output.rewards = out
        
        return output

def assert_token_for_gating(lst, device):
    """Find the last occurrence of a token_pattern in a list."""
    
    token_pattern = torch.tensor([128009, 128006, 78191, 128007, 271]).to(device)
    
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):                
        if (lst[j:j + token_pattern_len] == token_pattern).all():
            return True
    return False

def process_tokenized_input(_input, tokenizer, max_length=256, device='cpu'):
    
    out = tokenizer(_input, truncation=True, max_length=max_length, return_tensors="pt")['input_ids'].to(device)    
    out = torch.concat([torch.tensor([[tokenizer.pad_token_id]*(max_length - len(out[0]))]).to(device).T,
                        out.T]).T

    #If the search token isn't in the output add it:
    if not assert_token_for_gating(out[0], device):
        out[0, -5:] = torch.tensor([128009, 128006, 78191, 128007, 271])
        
    return out.to(torch.int64)

def score_preferences(chosen, reject, model, tokenizer, max_length:int=256, device='cuda'):
    
    # chosen_emb = tokenizer(chosen, return_tensors="pt", 
    #                       padding=True, truncation=True, max_length=20)
    # reject_emb = tokenizer(reject, return_tensors="pt",
    #                       padding=True, truncation=True, max_length=20)
        
    chosen_emb = process_tokenized_input(chosen, tokenizer,
                                         max_length=max_length, device=device)
    reject_emb = process_tokenized_input(reject, tokenizer,
                                         max_length=max_length, device=device)
    
    chosen_out = model(chosen_emb).rewards.cpu()
    reject_out = model(reject_emb).rewards.cpu()
    
    return (chosen_out > reject_out).to(torch.int64)

def create_list_defaultdict():
    return defaultdict(list)


def get_tvhh(split:str, silent:bool = False,
             cache_dir:str=None, force_new:bool=False,
             timesteps:int=5, sample_to_size:int=50000,
             changepoint:int=0, test_dataset:bool=False,
             init_reward_index:int=0, final_reward_index:int=1,
             device:str='cuda', max_length:int=256,
             test_changes_only:bool=False,
             train_changes_percent:Union[False, float]=False,
             **kwargs):
    
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
    
    print('Creating\Loading dataset with:',
          f"""
          test_dataset={test_dataset}\n
          """)
    
    
    #Setup our own cache for the nsgo dataset:
    created_cache_dir = '.cache' if cache_dir is None else cache_dir
    file_name = f'tvhh_dataset_{timesteps}' +\
        f'_{sample_to_size}' +\
        f'_{changepoint}' +\
        f'_{train_changes_percent}'.replace('.','') +\
        f'_{init_reward_index}{final_reward_index}' +\
        f'_{test_changes_only}'
    
    cache_path = os.path.join(created_cache_dir, file_name)
    train_path = os.path.join(cache_path, 'train.pkl')
    test_path = os.path.join(cache_path, 'test.pkl')
    
    if (not os.path.exists(cache_path)) or (force_new == True):
            
        ############################ LOAD DATA ################################
        #Load dataset (only has train set):
        #Create the dataset:
        print('Loading the Dataset')
        data = datasets.load_dataset('Anthropic/hh-rlhf', 
                           data_files="harmless-base/test.jsonl.gz")['train']
        df = pd.DataFrame(data)
        
        #Create the model and tokenizer:
        if test_dataset:
            print('Loading Test Model')
            model = test_model()  
        
        else:
            print('Loading ArmoRM model')
            model = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                        device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                        cache_dir=cache_dir, revision='main')
        tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                                  use_fast=True, revision='main')
            
        ########################## FILTER DATASET #############################
        #TODO: To do this step once the dataset is outputting
        
        if test_dataset:
            df = df.iloc[:10000]  
                
        ########################## CALCULATE PREFERENCES ######################
        print('Calculating Preferences')
        
        df['preferences'] = df.apply(lambda x: score_preferences(chosen=x['chosen'],
                                                                 reject=x['rejected'], 
                                                                 model=model,
                                                                 tokenizer=tokenizer,
                                                                 max_length=max_length,
                                                                 device=device), axis=1)

        df[['prompt','chosen','rejected']] = df.apply(split_prompt_and_responses,
                              axis=1, result_type='expand')
        
        unique_prompts = list(df['prompt'].unique())        
        split_ratio = int(0.2 * len(unique_prompts))
        train_qs, test_qs = sample_train_test_questions(unique_prompts, split=split_ratio)
        
        ################### CREATE NON STATIONARY DATASET #####################
        
        print('Create Non-Stationary Dataset')
        
        df_timesteps = list()
        for t in range(timesteps):
                
            #Copy the dataset and adjust the timestep
            dft = df.copy(deep=True)
            dft['time'] = t + 1
            
            #Update preferences:
            if t + 1 >= changepoint:
                dft['preference'] = dft['preferences'].apply(lambda x: x[0, final_reward_index])
            else:
                dft['preference'] = dft['preferences'].apply(lambda x: x[0, init_reward_index])
                
            dft['change'] = dft['preferences'].\
                apply(lambda x: int(x[0, final_reward_index] != x[0, init_reward_index]))
                
            df_timesteps.append(dft)
            
        df_out = pd.concat(df_timesteps)
             
        ################### SPLIT INTO TRAIN AND TEST DATASET #################
        #Separate into train and test via the timestep and prompt:
        
        if timesteps == 0: #write a test script for this...
            df_train = df_out[df_out['prompt'].isin(train_qs)]
            df_test = df_out[df_out['prompt'].isin(test_qs)]
        
        else:
            
            final_ts = df_out['time'].max()
            
            if test_changes_only:
            
                df_test  = df_out[(df_out['time'] == final_ts) &\
                              (df_out['prompt'].isin(test_qs)) &\
                              (df_out['change'] == 1)]
            else:
                
                df_test = df_out[(df_out['time'] == final_ts) &\
                             (df_out['prompt'].isin(test_qs))]
                             
            if train_changes_percent:
                
                df_train_change = df_out[(df_out['time'] < final_ts) &\
                                         (df_out['prompt'].isin(train_qs)) &\
                                         (df_out['change'] == 1)].reset_index(drop=True)
                
                #shuffle this dataset to ensure data points are randomly distributed w.r.t time
                df_train_no_change = df_out[(df_out['time'] < final_ts) &\
                                         (df_out['prompt'].isin(train_qs)) &\
                                         (df_out['change'] == 0)].reset_index(drop=True).\
                                    sample(frac=1)
                    
                len_chng = len(df_train_change)
                len_nchng = len(df_train_no_change)
                                
                samples_to_select = np.floor(len_chng * ((1/train_changes_percent) - 1))
                samples_to_select = int(samples_to_select)    
                                
                df_train = pd.concat([df_train_change, 
                                      df_train_no_change.iloc[:samples_to_select]])
                            
            else:
                
                df_train = df_out[(df_out['time'] < final_ts) &\
                              (df_out['prompt'].isin(train_qs))]
            
        #Reshuffle the dataframes
        df_train = df_train.sample(frac=1)
        df_test = df_test.sample(frac=1)
        
        #Subsample down to size:
        sample_to_size = np.min([len(df_train), sample_to_size])
        df_train = df_train.sample(n=sample_to_size)
                                
        #Analyse the training percentages
        pre_change_train = df_train[(df_train['change'] == 1) &\
                                    (df_train['time'] < changepoint)]
            
        post_change_train = df_train[(df_train['change'] == 1) &\
                                     (df_train['time'] >= changepoint)] 
            
        no_change_train = df_train[(df_train['change'] == 0)]
            
        ################### ANY ANALYTICS NEEDED TO BE PRINTED ################
        print(f"Size of training set {len(df_train)}")
        print(f"Size of testing set {len(df_test)}")        
        
        print(f"Preference Change from {init_reward_index} to {final_reward_index}")
        
        print(f"Preference change in training set {df_train['change'].mean()}")
        print(f"Preference change in testing set {df_test['change'].mean()}")
        
        pre_change_percent = len(pre_change_train)/len(df_train)     
        post_change_percent = len(post_change_train)/len(df_train)
        no_change_percent = len(no_change_train)/len(df_train)
        print(f"Percent examples with old preferences chng: {100*pre_change_percent}%")
        print(f"Percent examples with new preferences chng: {100*post_change_percent}%")
        print(f"Percent examples with no preferences chng: {100*no_change_percent}%")
        
        ############### PROCESS TRAIN AND TEST SET INTO DICT ##################
        print('Processing Train and Test Set into Dictionary')
        
        data_train = defaultdict(create_list_defaultdict)
        for index, row in tqdm.tqdm(df_train.iterrows(), desc='Processing HH', disable=silent):
                        
            prompt = row['prompt'] + f"| Time step: {str(row['time'])} |"
            chosen, rejected = row['chosen'], row['rejected']
            
            if row['preference'] == 1: chosen, rejected = row['chosen'], row['rejected']
            else: chosen, rejected = row['rejected'], row['chosen']
                        
            responses = [chosen, rejected]
            n_responses = len(data_train[prompt]['responses'])
            data_train[prompt]['pairs'].append((n_responses, n_responses + 1))
            data_train[prompt]['responses'].extend(responses)
            data_train[prompt]['sft_target'] = chosen
            data_train[prompt]['timestep'] = row['time']
                        
            
        print('Process Test Dataset into Dict')        
        data_test = defaultdict(create_list_defaultdict)
        for index, row in tqdm.tqdm(df_test.iterrows(), desc='Processing HH', disable=silent):
                        
            prompt = row['prompt'] + f"| Time step: {str(row['time'])} |"
            chosen, rejected = row['chosen'], row['rejected']
            
            if row['preference'] == 1: chosen, rejected = row['chosen'], row['rejected']
            else: chosen, rejected = row['rejected'], row['chosen']
                        
            responses = [chosen, rejected]
            n_responses = len(data_test[prompt]['responses'])
            data_test[prompt]['pairs'].append((n_responses, n_responses + 1))
            data_test[prompt]['responses'].extend(responses)
            data_test[prompt]['sft_target'] = chosen
            data_test[prompt]['timestep'] = row['time']
        
        #Create path and save train and test set to path
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
                
        print(f'saving dataset to {cache_path}')
        with open(train_path, 'wb') as f:
            pickle.dump(data_train, f)
            
        with open(test_path, 'wb') as f:
            pickle.dump(data_test, f)
    
    else:    
        #Load the train and test set from the path...
        print(f'loading dataset from {cache_path}')
        with open(train_path, 'rb') as f:
            data_train = pickle.load(f)
        with open(test_path, 'rb') as f:
            data_test = pickle.load(f)
            
    #Return the correct split:
    if split == 'train':
        output = data_train
    elif split == 'test':
        output = data_test
    else:
        raise NotImplementedError(f'get_nsgo split type: {split} not implemented')
        
    return output


############################# SECOND IMPLEMENTATION ###########################

def adjust_preference_label(preferences, time, changepoint, 
                            init_reward_index, final_reward_index,
                            timesteps, gradual:bool=False):
    
    lower_change_time = timesteps // 3 
    upper_change_time = lower_change_time * 2
    
    w = (time - lower_change_time)/lower_change_time
    
    
    if gradual: #In the gradual setting we vary the preferences linearly        
        if time < lower_change_time:
            output = preferences[0, init_reward_index]
            
        elif ((time >= lower_change_time) and (time <= upper_change_time)):
            output = preferences[0, init_reward_index] * w +\
                (1 - w) * preferences[0, final_reward_index]
        else:
            output = preferences[0, final_reward_index]
        
    else:
        if time <= changepoint:
            output = preferences[0, init_reward_index]
        else:
            output = preferences[0, final_reward_index]

    return output


def get_tvhh2(split:str, silent:bool = False,
             cache_dir:str=None, force_new:bool=False,
             timesteps:int=5, sample_to_size:int=50000,
             changepoint:int=0, test_dataset:bool=False,
             init_reward_index:int=0, final_reward_index:int=1,
             device:str='cuda', max_length:int=256,
             train_changes_percent:Union[False, float]=False,
             train_test_split:float=0.2, testing_mode=False,
             gradual:bool=False, **kwargs):
    
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
    
    rank0_print('Creating\Loading dataset with:',
          f"""
          test_dataset={test_dataset}\n
          """)
    
    
    #Setup our own cache for the nsgo dataset:
    created_cache_dir = '.cache' if cache_dir is None else cache_dir
    file_name = f'tvhh_dataset_{timesteps}' +\
        f'_{sample_to_size}' +\
        f'_{changepoint}' +\
        f'_{gradual}' +\
        f'_{train_changes_percent}'.replace('.','') +\
        f'_{init_reward_index}{final_reward_index}' +\
        f'_{train_test_split}'
    
    cache_path = os.path.join(created_cache_dir, file_name)
    train_path = os.path.join(cache_path, 'train.pkl')
    test_path = os.path.join(cache_path, 'test.pkl')
    
    if (not os.path.exists(cache_path)) or (force_new == True):
            
        ############################ LOAD DATA ################################
        #Load dataset (only has train set):
        #Create the dataset:
        rank0_print('Loading the Dataset')
        data = datasets.load_dataset('Anthropic/hh-rlhf', 
                           data_files="harmless-base/train.jsonl.gz")['train']
        df = pd.DataFrame(data)
        
        #Create the model and tokenizer:
        if test_dataset:
            rank0_print('Loading Test Model')
            model = test_model()  
        
        else:
            rank0_print('Loading ArmoRM model')
            model = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                        device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                        cache_dir=cache_dir, revision='main')
        tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                                  use_fast=True, revision='main')
            
        ########################## FILTER DATASET #############################
        #TODO: To do this step once the dataset is outputting
        
        if test_dataset:
            df = df.iloc[:10000]  
                
        ########################## CALCULATE PREFERENCES ######################
        rank0_print('Calculating Preferences')
        
        df['preferences'] = df.apply(lambda x: score_preferences(chosen=x['chosen'],
                                                                 reject=x['rejected'], 
                                                                 model=model,
                                                                 tokenizer=tokenizer,
                                                                 max_length=max_length,
                                                                 device=device), axis=1)

        df[['prompt','chosen','rejected']] = df.apply(split_prompt_and_responses,
                              axis=1, result_type='expand')
        
        ################### CREATE NON STATIONARY DATASET #####################
        
        rank0_print('Create Non-Stationary Dataset')
        
        #calculate the changes:
        df['change'] = df['preferences'].\
            apply(lambda x:int(x[0, final_reward_index] != x[0, init_reward_index]))
        
        #Create the test set out of only changing examples:        
        size_test = int(train_test_split * len(df))
        df_test = df[df['change'] == 1].iloc[:size_test]
        test_qs = list(df_test['prompt'].unique())
    
        if train_changes_percent:
            #Create the train set out of the remaining examples + with suitable train_change_percent
            df_train_change = df[(~df['prompt'].isin(test_qs)) &\
                                 (df['change'] == 1)]
                
            #shuffle this dataset to ensure data points are randomly distributed w.r.t time
            df_train_no_change = df[(~df['prompt'].isin(test_qs)) &\
                                    (df['change'] == 0)]
                
            len_chng = len(df_train_change)
                            
            samples_to_select = np.floor(len_chng * ((1/train_changes_percent) - 1))
            samples_to_select = int(samples_to_select)    
                            
            df_train = pd.concat([df_train_change, 
                                  df_train_no_change.iloc[:samples_to_select]])
            
        else:
            df_train = df[~df['prompt'].isin(test_qs)]
            
        df_train = df_train.sample(frac=1)
        
        #Assign the test timestep and preference label:
        df_test['time'] = timesteps
        df_test['preference'] = df_test['preferences'].\
            apply(lambda x: x[0, final_reward_index])
        
        times = [i + 1 for i in range(timesteps-1)]
        df_train['time'] = np.random.choice(times, replace=True, size=len(df_train))
        
        #set the preference label depending on the time point:
        df_train['preference'] = df_train[['preferences', 'time']].\
            apply(lambda x: adjust_preference_label(preferences=x['preferences'],
                                                    time=x['time'], changepoint=changepoint,
                                                    init_reward_index=init_reward_index,
                                                    final_reward_index=final_reward_index,
                                                    timesteps=timesteps,
                                                    gradual=gradual), axis=1)
        #When preference == 1 chosen is the preferred response...

        #Analyse the training percentages
        pre_change_train = df_train[(df_train['change'] == 1) &\
                                    (df_train['time'] < changepoint)]
            
        post_change_train = df_train[(df_train['change'] == 1) &\
                                     (df_train['time'] >= changepoint)] 
            
        no_change_train = df_train[(df_train['change'] == 0)]
        
        ################### ANY ANALYTICS NEEDED TO BE PRINTED ################
        rank0_print(f"Size of training set {len(df_train)}")
        rank0_print(f"Size of testing set {len(df_test)}")        
        
        rank0_print(f"Preference Change from {init_reward_index} to {final_reward_index}")
        
        rank0_print(f"Preference change in training set {df_train['change'].mean()}")
        rank0_print(f"Preference change in testing set {df_test['change'].mean()}")
        
        pre_change_percent = len(pre_change_train)/len(df_train)     
        post_change_percent = len(post_change_train)/len(df_train)
        no_change_percent = len(no_change_train)/len(df_train)
        rank0_print(f"Percent examples with old preferences chng: {100*pre_change_percent}%")
        rank0_print(f"Percent examples with new preferences chng: {100*post_change_percent}%")
        rank0_print(f"Percent examples with no preferences chng: {100*no_change_percent}%")
        
        ############### PROCESS TRAIN AND TEST SET INTO DICT ##################
        rank0_print('Processing Train and Test Set into Dictionary')
        
        data_train = defaultdict(create_list_defaultdict)
        for index, row in tqdm.tqdm(df_train.iterrows(), desc='Processing HH', disable=silent):
                        
            prompt = row['prompt']
            
            if row['preference'] == 1: chosen, rejected = row['chosen'], row['rejected']
            else: chosen, rejected = row['rejected'], row['chosen']
                
            responses = [chosen, rejected]
            n_responses = len(data_train[prompt]['responses'])
            data_train[prompt]['pairs'].append((n_responses, n_responses + 1))
            data_train[prompt]['responses'].extend(responses)
            data_train[prompt]['sft_target'] = chosen
            data_train[prompt]['timestep'] = row['time']
                        
            
        rank0_print('Process Test Dataset into Dict')        
        data_test = defaultdict(create_list_defaultdict)
        for index, row in tqdm.tqdm(df_test.iterrows(), desc='Processing HH', disable=silent):
                        
            prompt = row['prompt']
            chosen, rejected = row['chosen'], row['rejected']
            
            if row['preference'] == 1: chosen, rejected = row['chosen'], row['rejected']
            else: chosen, rejected = row['rejected'], row['chosen']
                        
            responses = [chosen, rejected]
            n_responses = len(data_test[prompt]['responses'])
            data_test[prompt]['pairs'].append((n_responses, n_responses + 1))
            data_test[prompt]['responses'].extend(responses)
            data_test[prompt]['sft_target'] = chosen
            data_test[prompt]['timestep'] = row['time']
        
        #Create path and save train and test set to path
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
                
        rank0_print(f'saving dataset to {cache_path}')
        with open(train_path, 'wb') as f:
            pickle.dump(data_train, f)
            
        with open(test_path, 'wb') as f:
            pickle.dump(data_test, f)
    
    else:    
        #Load the train and test set from the path...
        rank0_print(f'loading dataset from {cache_path}')
        with open(train_path, 'rb') as f:
            data_train = pickle.load(f)
        with open(test_path, 'rb') as f:
            data_test = pickle.load(f)
            
    #Return the correct split:
    if split == 'train':
        output = data_train
        if force_new:
            df_out = df_train
    elif split == 'test':
        output = data_test
        if force_new:
            df_out = df_test
    else:
        raise NotImplementedError(f'get_nsgo split type: {split} not implemented')
                
    if force_new and testing_mode:
        return output, df_out
    else:    
        return output


if __name__ == '__main__':
    
    data = get_tvhh2(split='train', silent=True,
                    cache_dir='.\.cache\William', force_new=True,
                    timesteps=20, sample_to_size=50000,
                    changepoint=17, device='cpu', max_length=10,
                    test_dataset=True, train_changes_percent=False, gradual=True)
                    #test_changes_only=True)







    