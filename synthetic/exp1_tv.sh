#!/bin/sh
#SBATCH -J exp1_tv
#SBATCH -p eecs,gpu,share
#SBATCH -o output/exp1.out
#SBATCH -e output/exp1.err
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1

python3 run.py --project exp1_tv