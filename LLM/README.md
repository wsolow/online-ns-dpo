# LLM experiment codebase for NS-DPO: Non Stationary Direct Preference Optimization

Our implementation of this repository is an adaptation from eric-mitchell's [DPO repository](https://github.com/eric-mitchell/direct-preference-optimization).

## Installation
- If you are using linux, please run `$ pip install -r requirements_linux.txt` for setting up the python libraries.
- If you are using Windows, please run `$ pip install -r requirements.txt`.

## Setting up experiments: 2C NSGO dataset
**Please note that wandb access is required to aggregate the experiment results.**
- First, train a SFT model:
    - you can run `$ bash scripts/nsgo2c_sft.sh` to train the SFT model. Please refer to the file for the detailed command.
    - The script above will download `Llama-2-7b-chat-hf` model to your repository, while creating 2C NSGO dataset by downloading [GlobalOpinionQA](https://huggingface.co/datasets/Anthropic/llm_global_opinions).
    - Navigate to the directory inside `./.cache/` and check the name of the directory where the trained SFT model is stored
- Train the NS-DPO model using the SFT model trained above.
    - Please refer to `scripts/nsgo2c_nsdpo.sh` and other DPO training scripts in `scripts/` directory for the detailed commands.
    - Use the directory of the saved SFT model to assign `$DIRECTORY_SFT` variable inside the NS-DPO training script. 



