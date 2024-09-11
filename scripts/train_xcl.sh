#!/bin/zsh
#SBATCH --partition=main
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/sound-separation/logs/mw/xcl_%x_%a.log
#SBATCH --job-name=dll-separation
#SBATCH --gres=gpu:1
#SBATCH --array=0
##SBATCH --exclude=gpu-a100-3

source /mnt/stud/home/mwirth/.zshrc
source /mnt/stud/home/mwirth/miniconda3/etc/profile.d/conda.sh
conda activate sound-separation
conda env list

cd /mnt/stud/work/deeplearninglab/ss2024/sound-separation

SEED=$SLURM_ARRAY_TASK_ID
EXP=cluster_xcl.yaml # Path to the experiment config

export HYDRA_FULL_ERROR=1

srun python -u src/train.py experiment=$EXP seed=$SEED
