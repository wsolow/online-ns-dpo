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
import torch
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, "../../")
sys.path.insert(0, "../")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
    ArmoRM
'''

def load_armorm(
    device="cuda"
):
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model = AutoModelForSequenceClassification.from_pretrained(
        path, 
        device_map=device,                           
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    return model, tokenizer

def construct_input(
    tokenizer,
    sources, 
    candidate1s, 
    candidate2s, 
    device="cuda"
):

    messages = [{"role": "user", "content": sources},
           {"role": "assistant", "content": candidate1s}]
    ids1 = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    messages = [{"role": "user", "content": sources},
           {"role": "assistant", "content": candidate2s}]
    ids2 = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    
    return ids1, ids2

def apply_armoRM(
    model,
    tokenizer,
    data,
    col_prompt,
    col_response_a,
    col_response_b,
):
    prefs_all = list()
    logits_all = list()

    for idx in range(data.shape[0]):
        inputs = data[col_prompt].iloc[idx]
        candidates_A = data[col_response_a].iloc[idx]
        candidates_B = data[col_response_b].iloc[idx]
        if len(inputs) > 0:
            with torch.no_grad():
                idsA, idsB = construct_input(tokenizer, inputs, candidates_A, candidates_B)

                outputA = model(idsA)
                preference_scoreA = outputA.score.cpu().float()  
                outputB = model(idsB)
                preference_scoreB = outputB.score.cpu().float()  
                
                logit = float(preference_scoreA - preference_scoreB)
                comparison_result = float((logit > 0) * 1.0)
                
                prefs_all.append(comparison_result)
                logits_all.append(logit)
            
                del idsA
                del idsB
                del outputA
                del outputB
    
        if (idx + 1) % 200 == 0:
            print(f"[{(idx + 1)} items] GPU VRAM {torch.cuda.memory_allocated()/1024/1024/1024} GB allocated")
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return prefs_all, logits_all

'''
    PairRM, Better-PairRM
'''
def load_pairrm(
    model
):
    pairrm = DebertaV2PairRM.from_pretrained(model, device_map="cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(model)
    return pairrm, tokenizer

def tokenize_pair(
    tokenizer,
    sources:List[str], 
    candidate1s:List[str], 
    candidate2s:List[str], 
    source_max_length=1224, 
    candidate_max_length=412
):
    source_prefix = "<|source|>"
    cand1_prefix = "<|candidate1|>"
    cand2_prefix = "<|candidate2|>"
    ids = []
    assert len(sources) == len(candidate1s) == len(candidate2s)
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        source_ids = tokenizer.encode(
            source_prefix + sources[i], max_length=source_max_length, truncation=True
        )
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = tokenizer.encode(
            cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True
        )
        candidate2_ids = tokenizer.encode(
            cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True
        )
        ids.append(source_ids + candidate1_ids + candidate2_ids)
        
    encodings = tokenizer.pad(
        {"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length
    )
    del ids
    
    return encodings
    
def apply_pairrm(
    pairrm,
    tokenizer,
    data,
    col_prompt,
    col_response_a,
    col_response_b,
    batch_size_prompts=8,
):
    
    prefs_all = list()
    logits_all = list()

    for idx_mult in range((len(data) // batch_size_prompts) + 1):
        idx_start = idx_mult * batch_size_prompts
        idx_end = idx_start + batch_size_prompts
    
        inputs = data[col_prompt].iloc[idx_start:idx_end].tolist()
        candidates_A = data[col_response_a].iloc[idx_start:idx_end].tolist()
        candidates_B = data[col_response_b].iloc[idx_start:idx_end].tolist()
        if len(inputs) > 0:
            with torch.no_grad():
                encodings = tokenize_pair(tokenizer, inputs, candidates_A, candidates_B)
                encodings = {k:v.to(pairrm.device) for k,v in encodings.items()}
                outputs = pairrm(**encodings)
                logits = outputs.logits.tolist()
                comparison_results = (outputs.logits > 0) * 1.0
                prefs_all += comparison_results.cpu().tolist()
                logits_all += logits
            
                del encodings
                del outputs
                del logits
                del comparison_results
    
        if (idx_mult + 1) % 20 == 0:
            print(f"[{(idx_mult + 1) * batch_size_prompts} items] GPU VRAM {torch.cuda.memory_allocated()/1024/1024/1024} GB allocated")
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return prefs_all, logits_all
