import argparse
import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DatasetWrapper
import numpy as np
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution
import trainers.adaclip
import datasets.imagenet
import trainers.locoop
from PIL import Image
from sklearn.manifold import TSNE
import os

import random
from collections import defaultdict

def subset_data(val_dataset, n_cls, n_per_cls):
    class_to_samples = defaultdict(list)

    # Assuming val_dataset is an instance of DatasetWrapper
    for idx, item in enumerate(val_dataset.data_source):
        class_label = item.label
        class_to_samples[class_label].append(idx)

    # Step 2: Randomly select 10 classes
    selected_classes = np.arange(n_cls) # random.sample(class_to_samples.keys(), n_cls)

    # Step 3: For each selected class, pick 100 samples
    selected_indices = []
    for class_label in selected_classes:
        selected_indices_for_class = random.sample(class_to_samples[class_label], n_per_cls)
        selected_indices.extend(selected_indices_for_class)

    # Create a new dataset containing only the selected indices
    subsampled_val_data_source = [val_dataset.data_source[idx] for idx in selected_indices]

    subsampled_val_dataset = DatasetWrapper(
        cfg=val_dataset.cfg, 
        data_source=subsampled_val_data_source, 
        transform=val_dataset.transform, 
        is_train=False
    )

    subsampled_val_dataloader = torch.utils.data.DataLoader(
        subsampled_val_dataset, batch_size=32, shuffle=False
        )
    return subsampled_val_dataloader


def subset_torch_data(dataset, max_count, num_classes=None, batch_size=32):
    indices = []
    
    class_to_indices = defaultdict(list)
    for i, label in enumerate(dataset.targets):
        class_to_indices[label].append(i)
    
    if num_classes is not None and num_classes<=len(class_to_indices.keys()):
        selected_classes = list(np.arange(num_classes)) # random.sample(list(class_to_indices), num_classes)
    else:
        selected_classes = list(class_to_indices.keys())
        
    for label in selected_classes:
        count = min(max_count,len(class_to_indices[label]))
        selected_indices = class_to_indices[label][:count]
        indices.extend(selected_indices)
    
    dataset = torch.utils.data.Subset(dataset, indices)
    subset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return subset_loader


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.lambda_value:
        cfg.lambda_value = args.lambda_value

    if args.topk:
        cfg.topk = args.topk


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    cfg.TRAINER.ADAPTERS = CN()
    cfg.TRAINER.ADAPTERS.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT = False
    cfg.TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT = False
    
    cfg.TRAINER.ADAPTERS.USE_TEXT_ADAPTER = True
    cfg.TRAINER.ADAPTERS.TRAIN_TEXT_ADAPTER = True
    
    cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER = False
    cfg.TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER = False
    
    cfg.TRAINER.TEXT_ADAPTER = CN()
    cfg.TRAINER.TEXT_ADAPTER.REDUCTION = 4
    cfg.TRAINER.TEXT_ADAPTER.RATIO = 0.2

    cfg.TRAINER.IMAGE_ADAPTER = CN()
    cfg.TRAINER.IMAGE_ADAPTER.REDUCTION = 4
    cfg.TRAINER.IMAGE_ADAPTER.RATIO = 0.2

    # Add configurations specific to LoCoOp
    cfg.TRAINER.LOCOOP = CN()
    cfg.TRAINER.LOCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LOCOOP.CSC = False  # class-specific context
    cfg.TRAINER.LOCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LOCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def extract_features(data_loader, model, device, dtype):
        """
        Extract features for the given data loader.
        """
        model.eval()  # Set model to evaluation mode
        all_feats = []
        all_labels = []

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
            
                features, _ = model(inputs.type(dtype))
                features = features / features.norm(dim=-1, keepdim=True)
                all_feats.append(features)
                all_labels.append(labels)

        all_feats = torch.cat(all_feats, dim=0)  
        all_labels = torch.cat(all_labels, dim=0)
        return all_feats, all_labels


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = setup_cfg(args)
    import clip_w_local
    _, preprocess = clip_w_local.load(cfg.MODEL.BACKBONE.NAME)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    
    import pdb; pdb.set_trace()
    
    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    if cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER:
        image_extractor = torch.nn.Sequential(trainer.model.image_encoder,
                                              trainer.model.image_adapter
                                              )
    else:
        image_extractor = trainer.model.image_encoder

    # ==== ID data
    id_loader = set_val_loader(args, preprocess)
    id_loader = subset_torch_data(id_loader.dataset, max_count=100, num_classes=args.num_classes, batch_size=args.batch_size)
    
    # ==== OD data
    out_dataset = 'iNaturalist'
    od_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
    od_loader = subset_torch_data(od_loader.dataset, max_count=100, batch_size=args.batch_size)
    
    # ==== extract image features
    id_feats, id_labels = extract_features(id_loader, image_extractor, device, dtype=trainer.model.dtype)
    od_feats, _ = extract_features(od_loader, image_extractor, device, dtype=trainer.model.dtype)
    
    # ==== extract text features
    with torch.no_grad():
        if cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT:
            text_feat_file = f"text_features_{cfg.DATASET.NAME}_{cfg.DATASET.SUBSAMPLE_CLASSES}.pt"
            if os.path.exists(text_feat_file):
                print(f"Loading precomputed text features from {text_feat_file}")
                text_feats_pre = torch.load(text_feat_file).type(trainer.model.dtype).to(device)
            embeddings = trainer.model.prompt_learner()
            text_feats_post = trainer.model.text_encoder(embeddings, trainer.model.tokenized_prompts)
        elif cfg.TRAINER.ADAPTERS.USE_TEXT_ADAPTER:
            text_feats_pre = trainer.model.text_features.to(device)
            text_feats_post = trainer.model.text_adapter(text_feats_pre)
            
        text_feats_pre = text_feats_pre / text_feats_pre.norm(dim=-1,  keepdim=True)
        text_feats_post = text_feats_post / text_feats_post.norm(dim=-1, keepdim=True)

        text_feats_pre = text_feats_pre[:args.num_classes]
        text_feats_post = text_feats_post[:args.num_classes]
    
    import matplotlib.pyplot as plt

    all_feats = torch.cat([id_feats, od_feats, text_feats_pre, text_feats_post], dim=0)
    od_labels = torch.full((len(od_feats),), -1, dtype=torch.long, device=device)
    text_labels = torch.arange(text_feats_pre.size(0), dtype=torch.long, device=device)
    all_labels = torch.cat([id_labels, od_labels, text_labels, text_labels])
    all_feats, all_labels = all_feats.cpu().numpy(), all_labels.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_feats)

    plt.figure(figsize=(12, 10), dpi=300)

    plt.scatter(tsne_results[:len(id_feats), 0], tsne_results[:len(id_feats), 1],
                c=all_labels[:len(id_feats)], cmap='tab10', label='In-distribution', alpha=0.6, marker='o', s=20)

    # Plot out-of-distribution features
    plt.scatter(tsne_results[len(id_feats):len(id_feats) + len(od_feats), 0],
                tsne_results[len(id_feats):len(id_feats) + len(od_feats), 1],
                c='gray', label='Out-of-distribution', alpha=0.6, marker='x', s=20)

    # Plot text embeddings (pre)
    start_idx = len(id_feats) + len(od_feats)
    end_idx = start_idx + len(text_feats_pre)
    plt.scatter(tsne_results[start_idx:end_idx, 0], tsne_results[start_idx:end_idx, 1],
                c=all_labels[start_idx:end_idx], cmap='tab10', label='Text-embedding (pre)', alpha=0.6, marker='s', s=40)

    # Plot text embeddings (post)
    start_idx = end_idx
    end_idx = start_idx + len(text_feats_post)
    plt.scatter(tsne_results[start_idx:end_idx, 0], tsne_results[start_idx:end_idx, 1],
                c=all_labels[start_idx:end_idx], cmap='tab10', label='Text-embedding (post)', alpha=0.6, marker='d', s=40)

    # Labeling and legend
    plt.title('t-SNE Visualization of Image and Text Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    
    plt.savefig('tsne_plot.png')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    
    parser.add_argument('--in_dataset', default='imagenet', type=str, choices=['imagenet'], help='in-distribution dataset')
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # augment for LoCoOp
    parser.add_argument('--lambda_value', type=float, default=1, help='temperature parameter')
    parser.add_argument('--topk', type=int, default=200, help='topk')
    # augment for visualization demo
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--image_path', default='', type=str, help='image path')
    parser.add_argument('--label', default=1, type=int, help='label')
    
    # augument for tsne plot
    parser.add_argument('--num_classes', default=10, type=int)
    args = parser.parse_args()
    main(args)
