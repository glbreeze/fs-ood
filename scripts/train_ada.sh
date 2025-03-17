#!/bin/bash

#SBATCH --job-name=nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100

SHOT=$1
NEGR=$2

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
export SSL_CERT_FILE=/scratch/lg154/sseg/fs-ood/cacert.pem

echo $PWD
python train_ada.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ep50.yaml \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_${SHOT}shots/text_shot${SHOT}_nr${NEGR} \
        --topk 8 \
        --load-epoch 50 \
        --model-dir output/imagenet/LoCoOp/vit_b16_ep50_8shots/nctx16_cscFalse_ctpend/seed1 \
        DATASET.NUM_SHOTS ${SHOT} \
        TRAINER.ADAPTERS.USE_TEXT_PROMPT True \
        TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT True \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER False \
        INPUT.NUM_CROPS 10 \
        TRAINER.ADAPTERS.LAMBDA_NEG ${NEGR}

"