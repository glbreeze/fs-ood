#!/bin/bash

#SBATCH --job-name=nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=13:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
lambda=$4

# Singularity path
ext3_path=/scratch/$USER/python10/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/python10/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv --overlay /scratch/lg154/python10/overlay-25GB-500K.ext3:ro \
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
/scratch/lg154/python10/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh

echo $PWD
python train.py \
--root /vast/lg154/datasets \
--seed 1 \
--trainer LoCoOp \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/LoCoOp/${CFG}.yaml \
--output-dir output/${DATASET}/LoCoOp/${CFG}_${SHOTS}shots/text_prompt_lamb${lambda} \
--lambda_value ${lambda} \
--topk 200 \
TRAINER.LOCOOP.N_CTX 16 \
TRAINER.LOCOOP.CSC False \
TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS ${SHOTS}

"