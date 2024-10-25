# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:22:45 2024

@author: William
"""

import os
import matplotlib.pyplot as plt
from src.visualisations.wandb_api import (
    download_runs)


entity='william_bankes'
project='nsdpo_nsgo_tv_results'
output_path = os.path.join('.', 'images')
    
#sft_filter = {'config.loss.name':'sft'}
#sft_runs, sft_configs = download_runs(entity=entity, project=project, filters=sft_filter)

dpo_filter = {'$and':[{'config.loss.name':'dpo'}, {'config.model.name_or_path':'EleutherAI/pythia-6.9b'}]}
dpo_runs, dpo_configs = download_runs(entity=entity, project=project, filters=dpo_filter)

nsdpo_filter = {'$and':[{'config.loss.name':'ns_dpo'}, {'config.model.name_or_path':'EleutherAI/pythia-6.9b'}]}
nsdpo_runs, nsdpo_configs = download_runs(entity=entity, project=project, filters=nsdpo_filter)


def process_runs(runs, run_configs, y_field):

    outputs = list()
    
    for i, run in enumerate(runs):
        
        #find the value at the final step:
            
        null_filter = run[y_field].isnull()
            
        max_step = run[~null_filter]['_step'].max()
        end_value = run[run['_step'] == max_step][y_field].values[0]
            
        #find the max value over the run:
        max_value = run[y_field].max() 
    
        x_label = run_configs[i]['dataset']['changepoint']
        x_label = 20 - 1 - x_label
               
        outputs.append((x_label, max_value*100, end_value*100))

    return outputs

dpo_data = process_runs(dpo_runs, dpo_configs, 'rewards_eval/accuracies')
nsdpo_data = process_runs(nsdpo_runs, nsdpo_configs, 'rewards_eval/accuracies')


#%%

fig, axs = plt.subplots()

width=0.45

axs.bar([x[0] for x in dpo_data], [x[1] for x in dpo_data], width=width, label='DPO')
axs.bar([x[0] + width for x in nsdpo_data], [x[1] for x in nsdpo_data], width=width, label='NS-DPO')

axs.set_title('Eval Reward Accuracies for Varying Changepoints out of 20 TimeSteps')
axs.set_xlabel('Change Point time')
axs.set_ylabel('Eval Reward Accuracies (%)')

axs.set_xticks([x[0] + width/2 for x in dpo_data], [str(x[0]) for x in dpo_data])
plt.legend(loc='best')
axs.set_ylim([40,80])


#%% Visualise the Time Varying AutoRegressive setting:
    
import os
import matplotlib.pyplot as plt
from src.visualisations.wandb_api import (
    download_runs)
    
entity='william_bankes'
project='nsdpo_nsgo_tv_ar_results'
output_path = os.path.join('.', 'images')
    

dpo_filter = {'config.loss.name':'dpo'}
dpo_runs, dpo_configs = download_runs(entity=entity, project=project, filters=dpo_filter)

nsdpo_filter = {'config.loss.name':'ns_dpo'}
nsdpo_runs, nsdpo_configs = download_runs(entity=entity, project=project, filters=nsdpo_filter)

def process_runs(runs, run_configs, y_field):

    outputs = list()
    
    for i, run in enumerate(runs):
        
        #find the value at the final step:
            
        null_filter = run[y_field].isnull()
            
        max_step = run[~null_filter]['_step'].max()
        end_value = run[run['_step'] == max_step][y_field].values[0]
            
        #find the max value over the run:
        max_value = run[y_field].max() 
    
        x_label = run_configs[i]['dataset']['start_time']
               
        outputs.append((x_label, max_value*100, end_value*100))

    return outputs

dpo_data = process_runs(dpo_runs, dpo_configs, 'rewards_eval/accuracies')
nsdpo_data = process_runs(nsdpo_runs, nsdpo_configs, 'rewards_eval/accuracies')

#%%

fig, axs = plt.subplots()

width=0.45

axs.bar([x[0] for x in dpo_data], [x[1] for x in dpo_data], width=width, label='DPO')
axs.bar([x[0] + width for x in nsdpo_data], [x[1] for x in nsdpo_data], width=width, label='NS-DPO')

axs.set_title('Eval Reward Accuracies vs Timepoint at which AR Process starts')
axs.set_xlabel('Start of AR Process')
axs.set_ylabel('Eval Reward Accuracies (%)')

axs.set_xticks([x[0] + width/2 for x in dpo_data], [str(x[0]) for x in dpo_data])
plt.legend(loc='lower left')
axs.set_ylim([60,72])

#%% visualise adapted tv ar setting:
    
import os
import matplotlib.pyplot as plt
from src.visualisations.wandb_api import (
    download_runs)
    
entity='william_bankes'
project='nsdpo_nsgo_tv_ar_adapted_test_results'
output_path = os.path.join('.', 'images')
    

dpo_filter = {'config.loss.name':'dpo'}
dpo_runs, dpo_configs = download_runs(entity=entity, project=project, filters=dpo_filter)

nsdpo_filter = {'config.loss.name':'ns_dpo'}
nsdpo_runs, nsdpo_configs = download_runs(entity=entity, project=project, filters=nsdpo_filter)

def process_runs(runs, run_configs, y_field):

    outputs = list()
    
    for i, run in enumerate(runs):
        
        #find the value at the final step:
            
        null_filter = run[y_field].isnull()
            
        max_step = run[~null_filter]['_step'].max()
        end_value = run[run['_step'] == max_step][y_field].values[0]
            
        #find the max value over the run:
        max_value = run[y_field].max() 
    
        x_label = run_configs[i]['dataset']['start_time']
               
        outputs.append((x_label, max_value*100, end_value*100))

    return outputs

dpo_data = process_runs(dpo_runs, dpo_configs, 'rewards_eval/accuracies')
nsdpo_data = process_runs(nsdpo_runs, nsdpo_configs, 'rewards_eval/accuracies')

#%% Plot the results 

fig, axs = plt.subplots()

width=0.45

axs.bar([x[0] for x in dpo_data], [x[1] for x in dpo_data], width=width, label='DPO')
axs.bar([x[0] + width for x in nsdpo_data], [x[1] for x in nsdpo_data], width=width, label='NS-DPO')

axs.set_title('Eval Reward Accuracies vs Timepoint at which AR Process starts (adj test)')
axs.set_xlabel('Start of AR Process')
axs.set_ylabel('Eval Reward Accuracies (%)')

axs.set_xticks([x[0] + width/2 for x in dpo_data], [str(x[0]) for x in dpo_data])
plt.legend(loc='best')
axs.set_ylim([60,80])



#%% visualise effective window size:
import numpy as np   
import matplotlib.pyplot as plt 
    
T = 100

gammas = np.arange(0.1, 1.0, 0.2)
fig, axs = plt.subplots()
xs = np.linspace(0, T, 100)

for gamma in gammas:

    ys = gamma ** (T - xs)
        
    axs.plot(xs, ys, label=f'{gamma:.1f}')
    
    x = T - np.log(T)/(1 - gamma)   
        
    if x < 0: x=0
    y = gamma ** (T - x)
    
    axs.scatter(x,y)
    
    
plt.legend(title='gamma')
axs.set_xlabel('timestep')
axs.set_ylabel('discount')
axs.set_title('Effective Window Size (Faury et al 2021.) For Different Gamma')

    
    
#%% Create plot for paper around the 

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualisations.wandb_api import (
    download_runs)

entity='william_bankes'
project='nsdpo_nsgo_tv_ar_results'
output_path = os.path.join('.', 'images')
    
dpo_filter = {'config.loss.name':'dpo'}
dpo_runs, dpo_configs = download_runs(entity=entity, project=project, filters=dpo_filter)

nsdpo_filter = {'config.loss.name':'ns_dpo'}
nsdpo_runs, nsdpo_configs = download_runs(entity=entity, project=project, filters=nsdpo_filter)
    
    
#%% Process the runs:

def process_runs(runs, run_configs, y_field):

    outputs = list()
    
    for i, run in enumerate(runs):
        
        #find the value at the final step:
            
        null_filter = run[y_field].isnull()
            
        max_step = run[~null_filter]['_step'].max()
        end_value = run[run['_step'] == max_step][y_field].values[0]
            
        #find the max value over the run:
        max_value = run[y_field].max() 
    
        x_label = run_configs[i]['dataset']['start_time']
               
        outputs.append((x_label, max_value*100, end_value*100))

    return outputs

t = 'end'

process_dpo_run = process_runs(dpo_runs, dpo_configs, 'rewards_eval/accuracies')
process_nsdpo_run = process_runs(nsdpo_runs, nsdpo_configs, 'rewards_eval/accuracies')

df_dpo = pd.DataFrame(process_dpo_run)
df_nsdpo = pd.DataFrame(process_nsdpo_run)

df_dpo.columns = ['start_time', 'max_val', 'end_val']
df_nsdpo.columns = ['start_time', 'max_val', 'end_val']
    
grp_dpo = df_dpo.groupby('start_time').agg({f'{t}_val': ['mean', 'std']})
grp_nsdpo = df_nsdpo.groupby('start_time').agg({f'{t}_val': ['mean', 'std']})
    
fig, axs = plt.subplots()

width=0.5

x_vals = grp_dpo.index
y_vals_dpo = grp_dpo[f'{t}_val']['mean']
y_errs_dpo = grp_dpo[f'{t}_val']['std']

y_vals_nsdpo = grp_nsdpo[f'{t}_val']['mean']
y_errs_nsdpo = grp_nsdpo[f'{t}_val']['std']

axs.bar(x_vals, y_vals_dpo, label='DPO', width=width)
axs.bar([x + width for x in x_vals], y_vals_nsdpo, width=width, label='NS-DPO')

axs.errorbar(x_vals, y_vals_dpo, yerr=y_errs_dpo, fmt="o", c='r')
axs.errorbar([x + width for x in x_vals], y_vals_nsdpo, yerr=y_errs_nsdpo, fmt="o", c='r')

axs.set_title('Eval Reward Accuracies vs Timepoint at which AR Process starts')
axs.set_xlabel('Start of AR Process')
axs.set_ylabel('Eval Reward Accuracies (%)')

axs.set_xticks([x + width/2 for x in x_vals], [str(x) for x in x_vals])
plt.legend(loc='best')
axs.set_ylim([60,80])

#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualisations.wandb_api import (
    download_runs)

entity='william_bankes'
project='llama2_tvhh2_gradual_results'
output_path = os.path.join('.', 'images')

    
dpo_filter = {'$and':[{'config.loss.name':'dpo'},
                     {'config.dataset.changepoint':70}]}
dpo_runs, dpo_configs = download_runs(entity=entity, project=project, filters=dpo_filter)

nsdpo_filter = {'$and':[{'config.loss.name':'ns_dpo'},
                     {'config.dataset.changepoint':70}]}
nsdpo_runs, nsdpo_configs = download_runs(entity=entity, project=project, filters=nsdpo_filter)

dpo_filter = {'$and':[{'config.loss.name':'dpo'},
                     {'config.dataset.changepoint':10}]}
dpo10_runs, dpo10_configs = download_runs(entity=entity, project=project, filters=dpo_filter)

nsdpo_filter = {'$and':[{'config.loss.name':'ns_dpo'},
                     {'config.dataset.changepoint':10}]}
nsdpo10_runs, nsdpo10_configs = download_runs(entity=entity, project=project, filters=nsdpo_filter)
    
    
#%% Process the runs:
import matplotlib
    
def process_runs(runs, run_configs, y_field):

    outputs = list()
    
    for i, run in enumerate(runs):
        
        #find the value at the final step:
            
        null_filter = run[y_field].isnull()
            
        max_step = run[~null_filter]['_step'].max()
        end_value = run[run['_step'] == max_step][y_field].values[0]
            
        #find the max value over the run:
        max_value = run[y_field].max() 
    
        x_label = run_configs[i]['dataset']['changepoint']
        outputs.append((x_label, max_value*100, end_value*100))

    return outputs

t = 'end'

process_dpo_run = process_runs(dpo_runs, dpo_configs, 'rewards_eval/accuracies')
process_nsdpo_run = process_runs(nsdpo_runs, nsdpo_configs, 'rewards_eval/accuracies')

process_dpo10_run = process_runs(dpo10_runs, dpo10_configs, 'rewards_eval/accuracies')
process_nsdpo10_run = process_runs(nsdpo10_runs, nsdpo10_configs, 'rewards_eval/accuracies')


df_dpo = pd.DataFrame(process_dpo_run)
df_nsdpo = pd.DataFrame(process_nsdpo_run)

df_dpo.columns = ['changepoint', 'max_val', 'end_val']
df_nsdpo.columns = ['changepoint', 'max_val', 'end_val']
    
grp_dpo = df_dpo.groupby('changepoint').agg({f'{t}_val': ['mean', 'std']})
grp_nsdpo = df_nsdpo.groupby('changepoint').agg({f'{t}_val': ['mean', 'std']})

df_dpo10 = pd.DataFrame(process_dpo10_run)
df_nsdpo10 = pd.DataFrame(process_nsdpo10_run)

df_dpo10.columns = ['changepoint', 'max_val', 'end_val']
df_nsdpo10.columns = ['changepoint', 'max_val', 'end_val']
    
grp_dpo10 = df_dpo10.groupby('changepoint').agg({f'{t}_val': ['mean', 'std']})
grp_nsdpo10 = df_nsdpo10.groupby('changepoint').agg({f'{t}_val': ['mean', 'std']})

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
    
fig, axs = plt.subplots()

width=0.04 
capsize=15  
elinewidth=2 
markeredgewidth=2

x_vals = [0.85, 0.95]
y_vals_dpo = pd.concat([grp_dpo[f'{t}_val']['mean'], grp_dpo[f'{t}_val']['mean']])/100
y_errs_dpo = pd.concat([grp_dpo[f'{t}_val']['std'], grp_dpo[f'{t}_val']['std']])/100

y_vals_nsdpo = pd.concat([grp_nsdpo[f'{t}_val']['mean'], grp_nsdpo10[f'{t}_val']['mean']])/100
y_errs_nsdpo = pd.concat([grp_nsdpo[f'{t}_val']['std'], grp_nsdpo10[f'{t}_val']['std']])/100

axs.bar(x_vals, y_vals_dpo, label='DPO', width=width, color='tomato') #yerr=y_errs_dpo,
axs.bar([x + width for x in x_vals], y_vals_nsdpo, width=width, label='NS-DPO', color='dodgerblue') #yerr=y_errs_nsdpo,

axs.errorbar(x_vals, y_vals_dpo, yerr=y_errs_dpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)
axs.errorbar([x + width for x in x_vals], y_vals_nsdpo, yerr=y_errs_nsdpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)

#axs.set_title('Eval Reward Accuracies vs Timepoint at which AR Process starts')
axs.set_xlabel('$\gamma$')
axs.set_ylabel('Reward Accuracy')

axs.set_xticks([x + width/2 for x in x_vals], [str(x) for x in x_vals])
plt.legend(loc='best')
axs.set_ylim([0.35,0.68])

plt.grid(alpha=0.2)

plt.legend(loc='upper left', ncol=2)


output_path = r'C:\Users\William\Documents\Work\UCL PhD\Projects\NS-DPO\Paper Images'
fig.savefig(os.path.join(output_path, 'NSDPO_TVHH2_Gradual_Llama2.pdf'),
                         bbox_inches='tight',
                         dpi=200,
                         transparent=True)

#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualisations.wandb_api import (
    download_runs)

entity='william_bankes'

project='llama2_tvhh2_changepoint_results_07_lr'
dpo_filter = {'config.loss.name':'dpo'}
dpo_runs, dpo_configs = download_runs(entity=entity, project=project, filters=dpo_filter)
nsdpo_filter = {'config.loss.name':'ns_dpo'}
nsdpo_runs, nsdpo_configs = download_runs(entity=entity, project=project, filters=nsdpo_filter)
ipo_filter = {'config.loss.name':'ipo'}
ipo_runs, ipo_configs = download_runs(entity=entity, project=project, filters=ipo_filter)


project='llama2_tvhh2_changepoint_results_08_lr'
dpo_filter = {'config.loss.name':'dpo'}
dpo_runs08, dpo_configs08 = download_runs(entity=entity, project=project, filters=dpo_filter)
nsdpo_filter = {'config.loss.name':'ns_dpo'}
nsdpo_runs08, nsdpo_configs08 = download_runs(entity=entity, project=project, filters=nsdpo_filter)
ipo_filter = {'config.loss.name':'ipo'}
ipo_runs08, ipo_configs08 = download_runs(entity=entity, project=project, filters=ipo_filter)


project='llama2_tvhh2_changepoint_results_09_lr'
dpo_filter = {'config.loss.name':'dpo'}
dpo_runs09, dpo_configs09 = download_runs(entity=entity, project=project, filters=dpo_filter)
nsdpo_filter = {'config.loss.name':'ns_dpo'}
nsdpo_runs09, nsdpo_configs09 = download_runs(entity=entity, project=project, filters=nsdpo_filter)
ipo_filter = {'config.loss.name':'ipo'}
ipo_runs09, ipo_configs09 = download_runs(entity=entity, project=project, filters=ipo_filter)

    
#%% Process the runs:
import matplotlib
    
def process_runs(runs, run_configs, y_field):

    outputs = list()
    
    for i, run in enumerate(runs):
        
        #find the value at the final step:
            
        null_filter = run[y_field].isnull()
            
        max_step = run[~null_filter]['_step'].max()
        end_value = run[run['_step'] == max_step][y_field].values[0]
            
        #find the max value over the run:
        max_value = run[y_field].max() 
    
        x_label = run_configs[i]['dataset']['changepoint']
               
        outputs.append((x_label, max_value*100, end_value*100))

    return outputs

def process_to_df(processed_run):
    
    df = pd.DataFrame(processed_run)
    df.columns = ['changepoint', 'max_val', 'end_val']
    
    return df.groupby('changepoint').agg({f'{t}_val': ['mean', 'std']})



t = 'end'


process_dpo_run = process_runs(dpo_runs, dpo_configs, 'rewards_eval/accuracies')
process_ipo_run = process_runs(ipo_runs, ipo_configs, 'rewards_eval/accuracies')
process_nsdpo_run = process_runs(nsdpo_runs, nsdpo_configs, 'rewards_eval/accuracies')

process_dpo_run08 = process_runs(dpo_runs08, dpo_configs08, 'rewards_eval/accuracies')
process_ipo_run08 = process_runs(ipo_runs08, ipo_configs08, 'rewards_eval/accuracies')
process_nsdpo_run08 = process_runs(nsdpo_runs08, nsdpo_configs08, 'rewards_eval/accuracies')

process_dpo_run09 = process_runs(dpo_runs09, dpo_configs09, 'rewards_eval/accuracies')
process_nsdpo_run09 = process_runs(nsdpo_runs09, nsdpo_configs09, 'rewards_eval/accuracies')
process_ipo_run09 = process_runs(ipo_runs09, ipo_configs09, 'rewards_eval/accuracies')

df_dpo = process_to_df(process_dpo_run)
df_nsdpo = process_to_df(process_nsdpo_run)
df_ipo = process_to_df(process_ipo_run)

df_dpo08 = process_to_df(process_dpo_run08)
df_nsdpo08 = process_to_df(process_nsdpo_run08)
df_ipo08 = process_to_df(process_ipo_run08)

df_dpo09 = process_to_df(process_dpo_run09)
df_nsdpo09 = process_to_df(process_nsdpo_run09)
df_ipo09 = process_to_df(process_ipo_run09)


#%%

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
    
fig, axs = plt.subplots()

width=5 
capsize=7  
elinewidth=2 
markeredgewidth=2

x_vals = df_dpo.index
y_vals_dpo = df_dpo09[f'{t}_val']['mean']/100
y_errs_dpo = df_dpo09[f'{t}_val']['std']/100

y_vals_nsdpo = df_nsdpo09[f'{t}_val']['mean']/100
y_errs_nsdpo = df_nsdpo09[f'{t}_val']['std']/100

y_vals_ipo = df_ipo09[f'{t}_val']['mean']/100
y_errs_ipo = df_ipo09[f'{t}_val']['std']/100

axs.bar(x_vals, y_vals_dpo, label='DPO', width=width, color='tomato') #yerr=y_errs_dpo,
axs.bar([x + width for x in x_vals], y_vals_ipo, width=width, label='IPO', color='darkred')
axs.bar([x + 2*width for x in x_vals], y_vals_nsdpo, width=width, label='NS-DPO', color='dodgerblue') #yerr=y_errs_nsdpo,

axs.errorbar(x_vals, y_vals_dpo, yerr=y_errs_dpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)
axs.errorbar([x + width for x in x_vals], y_vals_ipo, yerr=y_errs_ipo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)
axs.errorbar([x + 2*width for x in x_vals], y_vals_nsdpo, yerr=y_errs_nsdpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)

#axs.set_title('Eval Reward Accuracies vs Timepoint at which AR Process starts')
axs.set_xlabel('Change Point')
axs.set_ylabel('Reward Accuracies')

axs.set_xticks([x + width/2 for x in x_vals], [str(x) for x in x_vals])
plt.legend(loc='best')
axs.set_ylim([0.2, 0.85])
plt.grid(alpha=0.2)


plt.legend(loc='upper right', ncol=3)


output_path = r'C:\Users\William\Documents\Work\UCL PhD\Projects\NS-DPO\Paper Images'
fig.savefig(os.path.join(output_path, 'NSDPO_TVHH2_chngpoint_09_llama2.pdf'),
                         bbox_inches='tight',
                         dpi=200,
                         transparent=True)

#%%

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
    
fig, axs = plt.subplots()

width=5 
capsize=7  
elinewidth=2 
markeredgewidth=2 

x_vals = df_dpo08.index
y_vals_dpo = df_dpo08[f'{t}_val']['mean']/100
y_errs_dpo = df_dpo08[f'{t}_val']['std']/100

y_vals_nsdpo = df_nsdpo08[f'{t}_val']['mean']/100
y_errs_nsdpo = df_nsdpo08[f'{t}_val']['std']/100

y_vals_ipo = df_ipo08[f'{t}_val']['mean']/100
y_errs_ipo = df_ipo08[f'{t}_val']['std']/100

axs.bar(x_vals, y_vals_dpo, label='DPO', width=width, color='tomato') #yerr=y_errs_dpo,
axs.bar([x + width for x in x_vals], y_vals_ipo, width=width, label='IPO', color='darkred')
axs.bar([x + 2*width for x in x_vals], y_vals_nsdpo, width=width, label='NS-DPO', color='dodgerblue') #yerr=y_errs_nsdpo,

axs.errorbar(x_vals, y_vals_dpo, yerr=y_errs_dpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)
axs.errorbar([x + width for x in x_vals], y_vals_ipo, yerr=y_errs_ipo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)
axs.errorbar([x + 2*width for x in x_vals], y_vals_nsdpo, yerr=y_errs_nsdpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)

#axs.set_title('Eval Reward Accuracies vs Timepoint at which AR Process starts')
axs.set_xlabel('Change Point')
axs.set_ylabel('Reward Accuracies')

axs.set_xticks([x + width/2 for x in x_vals], [str(x) for x in x_vals])
plt.legend(loc='best')
axs.set_ylim([.25,.84])


plt.legend(loc='upper right', ncol=3)
plt.grid(alpha=0.2)

output_path = r'C:\Users\William\Documents\Work\UCL PhD\Projects\NS-DPO\Paper Images'
fig.savefig(os.path.join(output_path, 'NSDPO_TVHH2_chngpoint_08_llama2.pdf'),
                         bbox_inches='tight',
                         dpi=200,
                         transparent=True)

#%%

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
    
fig, axs = plt.subplots()

width=5 
capsize=7   
elinewidth=2 
markeredgewidth=2 

x_vals = df_dpo08.index
y_vals_dpo = df_dpo[f'{t}_val']['mean']/100
y_errs_dpo = df_dpo[f'{t}_val']['std']/100

y_vals_nsdpo = df_nsdpo[f'{t}_val']['mean']/100
y_errs_nsdpo = df_nsdpo[f'{t}_val']['std']/100 

y_vals_ipo = df_ipo[f'{t}_val']['mean']/100
y_errs_ipo = df_ipo[f'{t}_val']['std']/100

axs.bar(x_vals, y_vals_dpo, label='DPO', width=width, color='tomato') #yerr=y_errs_dpo,
axs.bar([x + width for x in x_vals], y_vals_ipo, width=width, label='IPO', color='darkred')
axs.bar([x + 2*width for x in x_vals], y_vals_nsdpo, width=width, label='NS-DPO', color='dodgerblue') #yerr=y_errs_nsdpo,

axs.errorbar(x_vals, y_vals_dpo, yerr=y_errs_dpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)
axs.errorbar([x + width for x in x_vals], y_vals_ipo, yerr=y_errs_ipo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)
axs.errorbar([x + 2*width for x in x_vals], y_vals_nsdpo, yerr=y_errs_nsdpo, c='k', fmt='none', capsize=capsize,
             elinewidth=elinewidth, markeredgewidth=markeredgewidth)

#axs.set_title('Eval Reward Accuracies vs Timepoint at which AR Process starts')
axs.set_xlabel('Change Point')
axs.set_ylabel('Reward Accuracies')

axs.set_xticks([x + width/2 for x in x_vals], [str(x) for x in x_vals])
plt.legend(loc='best')
axs.set_ylim([.25,.84])

plt.legend(loc='upper right', ncol=3)
plt.grid(alpha=0.2)

output_path = r'C:\Users\William\Documents\Work\UCL PhD\Projects\NS-DPO\Paper Images'
fig.savefig(os.path.join(output_path, 'NSDPO_TVHH2_chngpoint_07_llama2.pdf'),
                         bbox_inches='tight',
                         dpi=200,
                         transparent=True)














