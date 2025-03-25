# ==== Train LoCoOp ====

python train.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer LoCoOp\
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ep50.yaml \
        --output-dir output/imagenet/LoCoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/test \
        --lambda_value 0 \
        --topk 200 \
        TRAINER.LOCOOP.N_CTX 16 \
        TRAINER.LOCOOP.CSC False \
        TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
        DATASET.NUM_SHOTS 16


# ====== Fix Text adapter and train image adapter 
python train_ada.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ada.yaml \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_16shots/test \
        --lambda_value 0.25 \
        --topk 200 \
        --load-epoch 50 \
        --text-adapter-dir output/imagenet/adaclip/vit_b16_ada_16shots/text_adapter_ood \
        TRAINER.ADAPTERS.USE_TEXT_ADAPTER True \
        TRAINER.ADAPTERS.TRAIN_TEXT_ADAPTER False \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER True \
        TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER True \
        DATASET.NUM_SHOTS 16

# ==== Train AdaClip Text adapter ====
python train_ada.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ada.yaml \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_16shots/test \
        --topk 200 \
        --load-epoch 50 \
        DATASET.NUM_SHOTS 16 \
        TRAINER.ADAPTERS.USE_TEXT_ADAPTER True \
        TRAINER.ADAPTERS.TRAIN_TEXT_ADAPTER True \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER False \
        INPUT.GLOBAL_RRCROP_SCALE "[0.08, 1.0]"


# ==== Train AdaClip Text Prompt ====
python train_ada.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ep50.yaml \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_16shots/test \
        --topk 8 \
        --load-epoch 50 \
        --model-dir output/imagenet/LoCoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1 \
        DATASET.NUM_SHOTS 8 \
        INPUT.RRCROP_SCALE "(0.2, 1.0)" \
        TRAINER.ADAPTERS.USE_TEXT_PROMPT True \
        TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT True \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER False \
        INPUT.NUM_CROPS 10




python train_ada.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ep50.yaml \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_16shots/test \
        --topk 8 \
        --load-epoch 50 \
        --model-dir output/imagenet/adaclip/vit_b16_ep50_8shots/text_shot8_nr0.2 \
        MODEL.INIT_WEIGHTS output/imagenet/adaclip/vit_b16_ep50_8shots/text_shot8_nr0.2 \
        MODEL.INIT_EPOCH 10\
        DATASET.NUM_SHOTS 8 \
        INPUT.RRCROP_SCALE "(0.2, 1.0)" \
        TRAINER.ADAPTERS.USE_TEXT_PROMPT True\
        TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT False \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER False \
        INPUT.NUM_CROPS 10



python train_ada.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ep50.yaml \
        --load-epoch 50 \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_8shots/test \
        DATASET.NUM_SHOTS 8 \
        MODEL.INIT_WEIGHTS output/imagenet/adaclip/vit_b16_ep50_8shots/text_shot8_nr0.2 \
        MODEL.INIT_EPOCH ${EP} \
        TRAINER.ADAPTERS.USE_TEXT_PROMPT True \
        TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT False \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER False \
        TRAINER.ADAPTERS.LORA vision \
        INPUT.NUM_CROPS 10 \
        TRAINER.ADAPTERS.LAMBDA_NEG 0.2 \
        TRAINER.ADAPTERS.LAMBDA 1.0



# ====== train image adapter only



python train_ada.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ep50.yaml \
        --load-epoch 50 \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_8shots/both_shot8_nr0.2_lam1.0_test \
        DATASET.NUM_SHOTS 8 \
        MODEL.INIT_WEIGHTS output/imagenet/adaclip/vit_b16_ep50_8shots/text_shot8_nr0.2 \
        MODEL.INIT_EPOCH 10 \
        TRAINER.ADAPTERS.USE_TEXT_PROMPT True \
        TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT False \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER True \
        TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER True \
        INPUT.NUM_CROPS 10 \
        TRAINER.ADAPTERS.LAMBDA_NEG 0.2 \
        TRAINER.ADAPTERS.LAMBDA 1.0
"


singularity exec --nv --overlay /scratch/lg154/python10/overlay-25GB-500K.ext3:ro --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro /scratch/lg154/python10/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash

export SSL_CERT_FILE=/scratch/lg154/sseg/fs-ood/cacert.pem

# ====== For locoop
# train
sbatch scripts/locoop/train.sh imagenet vit_b16_ep50 end 16 16 False 0.25 200
# eval 
sbatch scripts/locoop/eval.sh imagenet vit_b16_ep50 1 output/imagenet/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed2

python eval_ood_detection.py \
--root /vast/lg154/datasets \
--trainer LoCoOp \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/LoCoOp/vit_b16_ep50.yaml \
--output-dir output/imagenet/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1 \
--model-dir output/imagenet/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1 \
--load-epoch 50 \
--T 1 \
TRAINER.LOCOOP.N_CTX 16 \
TRAINER.LOCOOP.CSC False 


# ====== TSNE plot with Text Adapter
python tsne_visualization.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ada.yaml \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_16shots/test \
        --lambda_value 0.25 \
        --topk 200 \
        --load-epoch 50 \
        --model-dir output/imagenet/adaclip/vit_b16_ada_16shots/text_adapter_ood \
        TRAINER.ADAPTERS.USE_TEXT_PROMPT False \
        TRAINER.ADAPTERS.USE_TEXT_ADAPTER True \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        DATASET.NUM_SHOTS 16


# ====== TSNE plot with LoCoOp
python tsne_visualization.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer LoCoOp \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ada.yaml \
        --output-dir output/imagenet/LoCoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/test \
        --lambda_value 0.25 \
        --topk 200 \
        --load-epoch 50 \
        --model-dir output/imagenet/LoCoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1 \
        TRAINER.ADAPTERS.USE_TEXT_PROMPT True \
        TRAINER.ADAPTERS.USE_TEXT_ADAPTER False \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        DATASET.NUM_SHOTS 16



# ====== TSNE plot with Nothing
python tsne_visualization.py \
        --root /vast/lg154/datasets \
        --seed 1 \
        --trainer AdaClip \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/LoCoOp/vit_b16_ada.yaml \
        --output-dir output/imagenet/adaclip/vit_b16_ep50_16shots/test \
        --lambda_value 0.25 \
        --topk 200 \
        --load-epoch 50 \
        --model-dir output/imagenet/adaclip/vit_b16_ada_16shots/image_adapter_ood \
        TRAINER.ADAPTERS.USE_TEXT_ADAPTER False \
        TRAINER.ADAPTERS.USE_IMAGE_ADAPTER False \
        DATASET.NUM_SHOTS 16





# ====== train adaclip ======
sbatch scripts/train_ada.sh imagenet vit_b16_ada end 16 16 False 0.25 200

sbatch scripts/train_ada2.sh imagenet vit_b16_ada 16

sbatch scripts/train_ada_cpu.sh imagenet vit_b16_ada end 16 16 False 0.25 200