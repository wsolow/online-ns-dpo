#!/bin/sh
#SBATCH -J exp1_notv
#SBATCH -p eecs,gpu,share
#SBATCH -o output/exp1_notv.out
#SBATCH -e output/exp1_notv.err
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1

python3 run.py --project exp1_notv