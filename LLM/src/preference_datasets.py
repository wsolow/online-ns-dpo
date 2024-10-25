import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import get_local_dir, TemporarilySeededRandom
from src.datasets.non_stationary_datasets import get_nsgo, get_ufb_2rm, remove_time_and_var_from_prompt, remove_eos
from src.datasets.ns_datasets_2countries import get_nsgo_2countries
from src.datasets.two_model_non_stationary_datasets import get_tvhh, get_tvhh2
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
from datetime import datetime 
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

import os
import pickle

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string, keep_code:bool=True):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
            
        if keep_code:            
            if element.name == 'pre':
                for code in element.find_all('code'):
                    text.append("<code>" + code.get_text() + "</code>")
            elif element.name == 'code':
                text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text

def date_filter(example, cut_off_date):
    
    try:
        ex_datetime = datetime.strptime(example['date'], '%Y/%m/%d')
        cut_off_date = datetime.strptime(cut_off_date, '%Y/%m/%d')
        output = ex_datetime < cut_off_date
    except UnicodeEncodeError as e:
        print(e)
        output = False    
    return output

def metadata_filter(example, subjects):
    
    metadata_list = example['metadata']
    output = False
    
    #Find out if the subject is in the metadata list
    for subject in subjects:
        for metadata in metadata_list:
            if subject in metadata:
                output=True
    
    return output

def keyword_filter(example, keywords, keyword_target="question"):

    prompt = example[keyword_target]    
    output=False

    for word in keywords:
        if word in prompt.lower():
            output=True
    return output
    
def create_timestep(example, max_unix_timestep, min_unix_timestep):
    
    #Get unix timestamp:
    ts = time.mktime(datetime.strptime(example['date'], '%Y/%m/%d').timetuple())
    
    scaled_ts = (ts - min_unix_timestep)/(max_unix_timestep - min_unix_timestep)
    
    return scaled_ts * 100


def get_se(split, silent=False, cache_dir: str = None,
           temporal: bool = False,
           test: bool = False, 
           downsample_ratio:int=20,
           sample_train_to_size:int=100000, 
           sample_test_to_size:int=2000,
           subjects:list=['Stackoverflow'], 
           keywords:Union[list, None]=None,
           keep_code:bool=True,
           keyword_target:str="question",
           sample:bool=False)\
    -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
       
       Temporal splits the dataset by date and introduces timesteps to batches:
    """

    if test: #only process a small portion of the entire dataset:
        
        print(f'Loading test dataset from huggingface using streaming setup for {split} split')        
        dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', 
                                        cache_dir=cache_dir, split='train[:100]')
        dataset = dataset.map(lambda x: {'date': random.choice(['2022/01/01', '2008/01/01'])})
        print('done')
                
    else:
        
        print('Loading SE dataset (train split) from Huggingface...')
        dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', 
                                        cache_dir=cache_dir)['train']
        dataset = dataset.filter(lambda example: metadata_filter(example, subjects))
        dataset = dataset.shuffle(seed=42)
        print('done')
    
    if temporal:
        
        #Bring the dataset down to manageable standards:
        print('Sample the dataset to reasonable size')
        if downsample_ratio*sample_train_to_size < len(dataset):
            dataset = dataset.select(range(downsample_ratio*sample_train_to_size))
                  
        print('Splitting dataset by timestamp')
        #Split the dataset as before and after 2021:
        if split ==  'train': dataset = dataset.filter(lambda example: date_filter(example, '2021/01/01'))
        elif split == 'test': dataset = dataset.filter(lambda example: not date_filter(example, '2021/01/01'))
        else: raise NotImplementedError(f'split {split} not implemented for stack exchange dataset')
        
        #Create the timestep field, Max and min values hard coded for the time being:
        max_unix = time.mktime(datetime.strptime('2023/03/05', '%Y/%m/%d').timetuple())
        min_unix = time.mktime(datetime.strptime('2008/07/31', '%Y/%m/%d').timetuple())

        dataset = dataset.map(lambda example: {"timestep": create_timestep(example, max_unix, min_unix)})
        
    else:
        print(f'Creating {split} split of the dataset')
        dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
            range(int(len(dataset) * 0.01), len(dataset)))
        
    if keywords is not None:
        
        #Create a minimum dataset:
        keyword_dataset = dataset.filter(lambda example: keyword_filter(example, keywords, keyword_target))
        not_keyword_dataset = dataset.filter(lambda example: not keyword_filter(example, keywords, keyword_target))
        
        print('keyword dataset size:', len(keyword_dataset))
        print('non keyword dataset size:', len(not_keyword_dataset))
        
        #Create the keyword size and the size of the train\test split
        keyword_size = len(keyword_dataset)
        if split == 'train': sample_size = sample_train_to_size
        elif split == 'test': sample_size = sample_test_to_size
        else: raise NotImplementedError() 
        
        #Fill out dataset if required:
        if keyword_size < sample_size:
            data_list = [keyword_dataset,
                         not_keyword_dataset.select(\
                        range(sample_size - keyword_size))]
            dataset = datasets.concatenate_datasets(data_list)
        else:
            dataset = keyword_dataset.select(range(sample_size))
        
    else:       
        if split == 'train' and (len(dataset) > sample_train_to_size): dataset = dataset.select(range(sample_train_to_size))
        if split == 'test' and (len(dataset) > sample_test_to_size): dataset = dataset.select(range(sample_test_to_size))

    def strip_html(x):
        
        try:        
            x['question'] = strip_html_tags(x['question'], keep_code=keep_code)
            for a in x['answers']:
                a['text'] = strip_html_tags(a['text'], keep_code=keep_code)
            output = x    
            
        except UnicodeEncodeError as e:
            print(e)
            output = None
        
        return output

    #Strip html
    if len(dataset) < 1e6:
        dataset = dataset.to_pandas().apply(strip_html, axis=1).iterrows()
    else:
        dataset = dataset.map(strip_html, num_proc=1)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        if isinstance(row, Tuple): row = row[1]
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                
                if sample: #Sample from BT model using the scores as rewards
                    prob = 1 / (1 + np.exp(scores[j] - scores[i]))       
                    sample_output = np.random.binomial(1, prob)
                
                    #When sample_output = 1 score[i] is preferred
                    pairs.append((i, j) if sample_output > 0.5 else (j, i))
                
                else: #Use deterministic preferences based on the scores:
                    pairs.append((i, j) if scores[i] > scores[j] else (j, i))
                
        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])
        
        if temporal: data[prompt]['timestep'] = row['timestep']

    return data

def get_sekw(split, silent=False, cache_dir: str = None,
           temporal: bool = False,
           test: bool = False, 
           sample_train_to_size:int=100000, 
           sample_test_to_size:int=2000,
           subjects:list=['Stackoverflow'], 
           keywords:Union[list, None]=None,
           keep_code:bool=True,
           earliest_date:str='2008/07/31',
           split_date:str='2021/01/01',
           keyword_target:str="question",
           sample:bool=False)\
    -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
       
       Temporal splits the dataset by date and introduces timesteps to batches:
    """

    if test: #only process a small portion of the entire dataset:
        
        print(f'Loading test dataset from huggingface using streaming setup for {split} split')        
        dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', 
                                        cache_dir=cache_dir, split='train[:100]')
        dataset = dataset.map(lambda x: {'date': random.choice(['2022/01/01', '2008/01/01'])})
        print('done')
                
    else:

        #Setup our own cache for the nsgo dataset:
        subjs = "_".join(subjects)
        if keywords is None:
            kws = ""
        else:
            kws = "_".join(keywords)
        created_cache_dir = '.cache' if cache_dir is None else cache_dir
        edate = "".join(earliest_date.split("/"))
        sdate = "".join(split_date.split("/"))
        cache_path = os.path.join(
            created_cache_dir, 
            f'sekw_dataset_temporal{temporal}_{split}_{subjs}_{kws}_kwtarget{keyword_target}_train{sample_train_to_size}_test{sample_test_to_size}_earliest{edate}_splitdate{sdate}_sample{sample}'
        )
        train_path = os.path.join(cache_path, 'train.pkl')
        test_path = os.path.join(cache_path, 'test.pkl')

        if not os.path.exists(cache_path): #Create the dataset if it doesn't exist
        
            print('Loading SE dataset (train split) from Huggingface...')
            dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', 
                                            cache_dir=cache_dir)['train']
            dataset = dataset.filter(lambda example: metadata_filter(example, subjects))
            dataset = dataset.shuffle(seed=42)
            print('done')
    
            if temporal:
                
                #Bring the dataset down to manageable standards:
                print('Sample the dataset to reasonable size')
                # dataset = dataset.select(range(20*sample_train_to_size))
                        
                print('Splitting dataset by timestamp')
                #Cut out the dataset before earliest_date:
                dataset = dataset.filter(lambda example: not date_filter(example, earliest_date))
                #Split the dataset as before and after 2021:
                if split ==  'train': dataset = dataset.filter(lambda example: date_filter(example, split_date))
                elif split == 'test': dataset = dataset.filter(lambda example: not date_filter(example, split_date))
                else: raise NotImplementedError(f'split {split} not implemented for stack exchange dataset')
                
                #Create the timestep field, Max and min values hard coded for the time being:
                max_unix = time.mktime(datetime.strptime('2023/03/05', '%Y/%m/%d').timetuple())
                min_unix = time.mktime(datetime.strptime(earliest_date, '%Y/%m/%d').timetuple())
        
                dataset = dataset.map(lambda example: {"timestep": create_timestep(example, max_unix, min_unix)})
                
            else:
                print(f'Creating {split} split of the dataset')
                dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
                    range(int(len(dataset) * 0.01), len(dataset)))
                
            if keywords is not None:
                
                #Create a minimum dataset:
                dataset = dataset.filter(lambda example: keyword_filter(example, keywords, keyword_target))
                  
            if split == 'train' and (len(dataset) > sample_train_to_size): dataset = dataset.select(range(sample_train_to_size))
            if split == 'test' and (len(dataset) > sample_test_to_size): dataset = dataset.select(range(sample_test_to_size))
        
            def strip_html(x):
                
                try:        
                    x['question'] = strip_html_tags(x['question'], keep_code=keep_code)
                    for a in x['answers']:
                        a['text'] = strip_html_tags(a['text'], keep_code=keep_code)
                    output = x    
                    
                except UnicodeEncodeError as e:
                    print(e)
                    output = None
                
                return output
        
            #Strip html
            if len(dataset) < 1e6:
                dataset = dataset.to_pandas().apply(strip_html, axis=1).iterrows()
            else:
                dataset = dataset.map(strip_html, num_proc=1)
        
            data = defaultdict(dict)
            for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
                if isinstance(row, Tuple): row = row[1]
                prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
                responses = [' ' + a['text'] for a in row['answers']]
                scores = [a['pm_score'] for a in row['answers']]
        
                pairs = []
                for i in range(len(responses)):
                    for j in range(i + 1, len(responses)):
                        
                        if sample: #Sample from BT model using the scores as rewards
                            prob = 1 / (1 + np.exp(scores[j] - scores[i]))       
                            sample_output = np.random.binomial(1, prob)
                        
                            #When sample_output = 1 score[i] is preferred
                            pairs.append((i, j) if sample_output > 0.5 else (j, i))
                        
                        else: #Use deterministic preferences based on the scores:
                            pairs.append((i, j) if scores[i] > scores[j] else (j, i))
                        
                data[prompt]['responses'] = responses
                data[prompt]['pairs'] = pairs
                data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])
                
                if temporal: data[prompt]['timestep'] = row['timestep']

            #Create path and save train and test set to path
            os.makedirs(cache_path, exist_ok=True)
            
            print(f'saving dataset to {cache_path}')
            if split == "train":
                with open(train_path, 'wb') as f:
                    
                    pickle.dump(data, f)
            if split == "test":
                with open(test_path, 'wb') as f:
                    pickle.dump(data, f)
        else:
            print(f'loading dataset from {cache_path}')
            if split == "train":
                with open(train_path, 'rb') as f:
                    data = pickle.load(f)
            if split == "test":
                with open(test_path, 'rb') as f:
                    data = pickle.load(f)

    #Return the correct split:
    if split == 'train':
        output = data
    elif split == 'test':
        output = data
    else:
        raise NotImplementedError(f'get_sekw split type: {split} not implemented')
    print(f"[sekw] {len(data)} datapoints present")
    return output


def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
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

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, test:bool = False, **kwargs):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir, test=test, **kwargs)
    elif name == 'sekw':
        data = get_sekw(split, silent=silent, cache_dir=cache_dir, test=test, **kwargs)
    elif name == 'nsgo':
        data = get_nsgo(split, silent=silent, cache_dir=cache_dir, **kwargs)
    elif name == 'nsgo-2c':
        data = get_nsgo_2countries(split, silent=silent, cache_dir=cache_dir, **kwargs)
    elif name == 'ufb-2rm':
        data = get_ufb_2rm(split, **kwargs)
    elif name == 'tvhh':
        data = get_tvhh(split, silent=silent, cache_dir=cache_dir, test_dataset=test, **kwargs)
    elif name == 'tvhh2':
        data = get_tvhh2(split, silent=silent, cache_dir=cache_dir, test_dataset=test, **kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    keys = set(list(data.values())[0].keys())
    assert (keys == {'responses', 'pairs', 'sft_target'}) or \
        (keys == {'responses', 'pairs', 'sft_target', 'timestep'}), \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
        
        #If test mode reduce the dataset size to only 10 datapoints
    if test:
        print('Using test data config')
        data_keys, new_data = list(data.keys())[:32], dict()
        for key in data_keys:
            new_data[key] = data[key]
        data = new_data
        print('Pruned test data config')

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k == 'timestep':
                padded_batch[k] = torch.FloatTensor([ex[k] for ex in batch])
                
            else:
                padded_batch[k] = [ex[k] for ex in batch]
                        
        return padded_batch
    
    
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, 
                           max_length: int, max_prompt_length: int, timestep: Union[None,float]) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens
            
    if timestep is not None:
        batch['timestep'] = timestep
        
    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       test_dataset: bool = False,
                       remove_timestep=True,
                       loss_config: Optional[dict] = None,
                       **kwargs) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()
        
    #Setup sampling only on the training split
    sample = loss_config.get('sample', None)
    if split == 'test': sample=False
    
    if sample:
        assert loss_config is not None, 'if sample:True, loss_config must be provided'
        assert loss_config.name == 'ns_dpo', f'if sample:True loss_config.name must be ns_dpo, got {loss_config.name}'

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**20, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, test=test_dataset, **kwargs).items():
                
                if 'timestep' in data.keys():
                    flat_data.append((prompt, data['responses'], data['pairs'],
                                      data['sft_target'], data['timestep'], truncation_mode))
            
                else: 
                    flat_data.append((prompt, data['responses'], data['pairs'],
                                      data['sft_target'], None, truncation_mode))

        #Shuffle the timestep dataset:
        if 'timestep' in data.keys():
            with TemporarilySeededRandom(42):
                random.shuffle(flat_data)
            
            #Check the unique timesteps in the data:
            #unique_ts = np.unique([d[4] for d in flat_data])
            #print(f'no. unique ts on {split}', unique_ts)
            
            if ('nsgo' in names) or \
               ('nsgo-2c' in names) or \
               ('ufb-2rm' in names) or \
               ('tvhh' in names): #edit this otherwise
                #Remove the timestep detail in the prompt:
                print('Removing Timestep and Variation from non stationary prompt')
                flat_data = [
                    (remove_eos(remove_time_and_var_from_prompt(d[0], d[4], remove_timestep)),\
                    [remove_eos(d[1][0]), remove_eos(d[1][1])], \
                    d[2], remove_eos(d[3]), d[4], d[5]) \
                    for d in flat_data]

    #This is where the data is put into a Pytorch Tensor
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(int(next(permutation_seeds))):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, timestep, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length,
                                                       timestep=timestep)
                
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    
                    #If random number < gamma exponential thing then add to batch otherwise skip
                    if sample:
                        
                        gamma = loss_config['gamma']
                        ct = loss_config['current_time']
                        
                        exp_term = ct - 1 - timestep
                        gamma_exp = gamma ** exp_term                                               
                        
                        #Add the point to the batch proportionally to the gamma term
                        if torch.rand(1) < gamma_exp:
                        
                            batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]],
                                                                   truncation_mode, tokenizer, max_length,
                                                                   max_prompt_length, timestep=timestep)
                            batch.append(batch_element)
                            example_idx += 1
                        
                    else:
                    
                        batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]],
                                                               truncation_mode, tokenizer, max_length,
                                                               max_prompt_length, timestep=timestep)
                        batch.append(batch_element)
                        example_idx += 1
                    
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True   
