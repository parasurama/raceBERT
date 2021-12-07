#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=shap
#SBATCH --mail-type=END
#SBATCH --mail-user=pp1994@nyu.edu
#SBATCH --output=slurm_out/shap.out

singularity exec --nv \
--overlay /scratch/pp1994/singularity_images/overlay-10GB-400K.ext3:ro \
/scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "source /ext3/env.sh; python -m models.shap_values"