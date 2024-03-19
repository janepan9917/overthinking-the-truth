#!/bin/bash
#SBATCH --job-name=overthinking
#SBATCH --open-mode=append
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --array=0

DATA_ARR=(
    commonsense_qa
    strategy_qa
)
MODEL_STR="llama-2-13b-chat"
data=${DATA_ARR[$SLURM_ARRAY_TASK_ID]}
singularity exec --nv --overlay /scratch/jp5865/overlay-50G-10M-cot.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/miniconda3/bin/activate overthinking

python main.py -d $data --model $MODEL_STR --n_samples 20 --n_examples 2000 --log INFO
"
# # python main.py -d $data --model $MODEL_STR --n_samples 20 --n_examples 2000 --cot --log INFO
