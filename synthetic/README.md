# Synthetic Experiments for Non-Stationary Direct Preference Optimization (NS-DPO)

Our implementation of NS-DPO is an adaptation from Li et al.'s [Policy Optimization in RLHF: The Impact of Out-of-Preference Data](https://github.com/liziniu/policy_optimization).

## Installation
Run `$ pip install -r requirements.txt` to install the required libraries to run codes in this repository.

## Running experiments

### 1. Reproducing the experiment results in the paper
- The simplest way is to run `$ bash reproduce_results_paper.sh`.
- To run the experiments individually, you can consider:
    ```
    $ python3 run.py --project nsdpo_gamma_test1
    $ python3 run.py --project nsdpo_gamma_test2
    $ python3 run.py --project swdpo_w_test1
    $ python3 run.py --project nsdpo_swdpo_comparison
    ```

Please find the results in `logs/` directory.

### 2. Setup a configuration file in `configs/`
- Refer to the `default.yaml` file to check the default values.
- write your own configuration file `CONFIG_NAME.yaml`  with specifying values to be overwritten.

### 3. Use your configuration file to run the experiment
**We recommend leaving `default.yaml` intact.**
**Please refer to `configs/sample.yaml` and `configs/sample_group` for how to configure experiments.**

for a single configuration file such as `configs/CONFIG_NAME.yaml`:
```
$ python3 run.py --name_config CONFIG_NAME # No .yaml attached!
```

You can also store several configuration files in a directory, such as `configs/PROJECT_NAME/{CONFIG1.yaml, CONFIG2.yaml...}`:
```
$ python3 run.py --project PROJECT_NAME
```

