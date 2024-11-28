#!/bin/sh
#SBATCH -J exp2_tv
#SBATCH -p eecs,gpu,share
#SBATCH -o output/exp2_tv.out
#SBATCH -e output/exp2_tv.err
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1

python3 run.py --project exp2_tv