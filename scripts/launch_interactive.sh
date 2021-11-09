srun \
--mem=50g \
--cpus-per-task=8 \
--time=2:00:00 \
--pty /bin/bash -c \
'singularity exec --nv --overlay /scratch/pp1994/singularity_images/overlay-10GB-400K.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python"'
