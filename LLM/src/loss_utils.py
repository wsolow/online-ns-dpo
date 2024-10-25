# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:59:35 2024

@author: William
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple
import torch.nn.functional as F
from src.utils import pad_to_length

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    loss_config: dict = None,
                    timestep: Union[torch.FloatTensor, None] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        loss_config: dictionary with loss specific data components

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # if reference_free: #-> remove reference free argument for now... this all needs a good sort out...
    #     ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if loss_config.name == 'ipo':
        beta = loss_config['beta']
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    
    elif loss_config.name == 'ns_dpo':
        
        assert timestep is not None, 'For ns_dpo loss timestep cannot be None'
        
        beta = loss_config['beta']
        gamma = loss_config['gamma']
        norm = loss_config['normalise']
        current_time = loss_config['current_time']
        sample = loss_config['sample']
    
        if sample:
            gammas = torch.ones(len(timestep))
        else: 
            exponents = current_time - timestep - 1
            gammas = gamma ** exponents
    
        if norm: #Normalise the affect of the gammas on the grad norm:
            norm_term = (1/gammas.sum())
        else:
            norm_term = torch.tensor([1])
            
        norm_term = norm_term.to(timestep.device)
        gammas = gammas.to(timestep.device)

        losses = - norm_term * gammas * F.logsigmoid(beta * logits)

    elif loss_config.name == 'sw_dpo':
        
        assert timestep is not None, 'For sw_dpo loss timestep cannot be None'

        beta = loss_config['beta']
        current_time = loss_config['current_time']
        window_size = loss_config['window_size']
    
        within_window = ((current_time - timestep) <= window_size)
                
        losses = - F.logsigmoid(beta * logits) * within_window

    elif loss_config.name == 'dpo':
                
        beta = loss_config['beta']
        label_smoothing = loss_config['label_smoothing']
       
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        
    else:
        raise NotImplementedError(f'Loss {loss_config.name} has not been implemented')

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

def concatenated_forward(model: nn.Module, 
                         batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Run the given model on the given batch of inputs, concatenating the chosen
    and rejected inputs together. We do this to avoid doing two forward passes,
    because it's faster for FSDP.
    """
    
    concatenated_batch = concatenated_inputs(batch)
    all_logits = model(concatenated_batch['concatenated_input_ids'], \
                attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
    
    all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
    
    chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
    rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
    
    return chosen_logps, rejected_logps


def process_samples(prompt_input_ids, prompt_attention_masks, sample_input_ids):
    """
    Create an attention mask and label for sentences sampled form the language model.

    Parameters
    ----------
    prompt_input_ids : torch.tensor
        pytorch tensor of input ids
    prompt_attention_mask : torch.tensor
        pytorch tensor of masks for the prompt input
    sample_input_ids : torch.tensor
        pytorch tensor of input ids from the sample

    Returns
    -------
    None.

    """

    sample_attn_masks = list()
    sample_labels = list()
    
    for i, prompt_input_id in enumerate(prompt_input_ids):
        
        prompt_length = len(prompt_input_id)
        sample_length = len(sample_input_ids[i])
        
        diff = sample_length - prompt_length 
        
        #Create the sample attention mask
        sample_attn_mask = torch.concat([prompt_attention_masks[i].clone(),
                                         torch.ones(diff).to(prompt_attention_masks.device)])
        sample_attn_masks.append(sample_attn_mask)
        
        #Create the sample label 
        sample_label = torch.concat([torch.ones(prompt_length).\
                                     to(prompt_attention_masks.device)*-100,
                                     sample_input_ids[i][prompt_length:]])
                
        #Ensure the sample label is the same length as the sample it is labelling
        assert len(sample_label) == len(sample_input_ids[i]),\
            f'sample label: {len(sample_label)} must match length of sample input ids {len(sample_input_ids[i])}'
    
        sample_labels.append(sample_label)
    
    return torch.stack(sample_attn_masks), torch.stack(sample_labels).to(torch.int64)








