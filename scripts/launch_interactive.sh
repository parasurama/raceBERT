srun --mem=100g --cpus-per-task=6 --gres=gpu:1 --time=6:00:00 --pty /bin/bash -c '
  module purge
  module load anaconda3/2020.07
  module load cuda/11.1.74
  source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
  conda activate /scratch/pp1994/projects/conda_envs/lang_env
  export PATH=/scratch/pp1994/projects/conda_envs/lang_env/bin:$PATH
  python
'