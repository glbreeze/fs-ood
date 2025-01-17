#!/bin/bash

#SBATCH --job-name=nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=5:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)

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
--output-dir output/${DATASET}/adaclip/${CFG}_${SHOTS}shots/text_prompt_ood_id_loss_0.08 \
--topk 200 \
--load-epoch 50 \
DATASET.NUM_SHOTS ${SHOTS} \
TRAINER.ADAPTERS.USE_TEXT_ADAPTER False \
TRAINER.ADAPTERS.TRAIN_TEXT_ADAPTER False \
TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER False \
TRAINER.ADAPTERS.USE_TEXT_PROMPT True \
TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT True \
INPUT.GLOBAL_RRCROP_SCALE \"[0.08, 1.0]\"

"