#!/bin/zsh
#SBATCH --partition=main
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/sound-separation/logs/mw/HSN_perch_%x_%a.log
#SBATCH --job-name=dll-separation
#SBATCH --gres=gpu:0
#SBATCH --array=0
#SBATCH --exclude=gpu-v100-[1-4],gpu-a100-[1-4]

source /mnt/stud/home/mwirth/.zshrc
source /mnt/stud/home/mwirth/miniconda3/etc/profile.d/conda.sh
conda activate sound-separation
conda env list

cd /mnt/stud/work/deeplearninglab/ss2024/sound-separation


export HYDRA_FULL_ERROR=1

echo "starting..."

srun python -u src/separation_perch.py
