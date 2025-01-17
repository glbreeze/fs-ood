#!/bin/bash

#SBATCH --job-name=nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000


CSC=False
CTP=end
NCTX=16

DATASET=$1
CFG=$2

T=$3
MODEL_dir=$4
Output_dir=$4

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

python eval_ood_detection.py \
--root /vast/lg154/datasets \
--trainer LoCoOp \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/LoCoOp/${CFG}.yaml \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--load-epoch 50 \
--T ${T} \
TRAINER.LOCOOP.N_CTX ${NCTX} \
TRAINER.LOCOOP.CSC ${CSC} \

"