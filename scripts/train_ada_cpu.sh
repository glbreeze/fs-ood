#!/bin/bash

#SBATCH --job-name=nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=24:00:00

# job info
DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
lambda=$7
topk=$8

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
python train_ada.py \
--root /vast/lg154/datasets \
--seed 1 \
--trainer AdaClip \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/LoCoOp/${CFG}.yaml \
--output-dir output/${DATASET}/adaclip/${CFG}_${SHOTS}shots/test1_cpu \
--lambda_value ${lambda} \
--topk ${topk} \
TRAINER.LOCOOP.N_CTX ${NCTX} \
TRAINER.LOCOOP.CSC ${CSC} \
TRAINER.LOCOOP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS}

"