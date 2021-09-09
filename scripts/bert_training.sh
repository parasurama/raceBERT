#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=bert_training
#SBATCH --mail-type=END
#SBATCH --mail-user=pp1994@nyu.edu
#SBATCH --output=slurm_out/bert_training.out
  
module purge;
module load anaconda3/2020.07;
module load cuda/11.1.74

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/pp1994/projects/conda_envs/lang_env;
export PATH=/scratch/pp1994/projects/conda_envs/lang_env/bin:$PATH;

python -m bert_training bert