import os.path as osp
import os
import time
import random
import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import mse_loss
from torch.cuda.amp import GradScaler, autocast
from scipy import stats
from sklearn.manifold import TSNE

import clip_w_local
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image

from .adaclip import CUSTOM_TEMPLATES

prompt_template = 'a photo of a {}.'

class CLIPTextProcessor:
    def __init__(self, clip_model, pos_topk=None, neg_topk=2000, neg_subsample=-1, wordnet_database='txtfiles',
                 txt_exclude=None, emb_batchsize=1000, neg_dump_path='datasets/imagenet_neg/', pencentile=1):

        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.clip_model = clip_model.to(self.device)
        self.wordnet_database = wordnet_database
        self.txt_exclude = txt_exclude
        self.neg_subsample = neg_subsample
        self.emb_batchsize = emb_batchsize
        self.pos_topk = pos_topk
        self.neg_topk = neg_topk
        self.pencentile = pencentile
        self.text_features_pos = None
        self.text_features_neg = None
        self.prompt_template = 'a photo of a {}.'

        self._load_positive_text_features()
        if neg_dump_path and os.path.exists(os.path.join(neg_dump_path, 'neg_embedding.pth')):
            self._load_precomputed_negatives(neg_dump_path)
        else:
            self._compute_negative_text_features(neg_dump_path)
            self.refine_negative_samples(neg_dump_path)

    def _load_positive_text_features(self):
        from datasets.imagenet import ImageNet
        classnames = ImageNet.read_classnames('datasets/imagenet_classes.txt')
        self.classnames = list(classnames.values())
        prompts = [prompt_template.format(c.replace('_', ' ')) for c in self.classnames]
        text_inputs_pos = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

        with torch.no_grad():
            self.text_features_pos = self.clip_model.encode_text(text_inputs_pos).to(torch.float32)
            self.text_features_pos /= self.text_features_pos.norm(dim=-1, keepdim=True)

    def _load_precomputed_negatives(self, path):
        tic = time.time()
        dump_dict = torch.load(os.path.join(path, 'neg_embedding.pth'))
        self.text_features_neg = dump_dict['neg_emb'].to(self.device)
        toc = time.time()
        print(f'Successfully loaded the negative embedding in {toc - tic:.2f}s.')

    def _compute_negative_text_features(self, path):
        """all negative text embeddings without filtering"""
        txtfiles = [file for file in os.listdir(self.wordnet_database) if file.endswith('txt') and file.startswith('noun')]
        if self.txt_exclude:
            excluded_files = set(self.txt_exclude.split(','))
            txtfiles = [f for f in txtfiles if f not in excluded_files]

        words_noun = []
        dedup = set()
        for file in txtfiles:
            filetype = file.split('.')[0]
            with open(os.path.join(self.wordnet_database, file), 'r') as f:
                for line in f:
                    word = line.strip()
                    if word in dedup:
                        continue
                    dedup.add(word)
                    if word in self.classnames:
                        continue
                    else:
                        words_noun.append(self.prompt_template.format(word))

        if self.neg_subsample > 0:
            random.seed(42)
            words_noun = random.sample(words_noun, self.neg_subsample)
        self.words_noun = words_noun

        text_inputs_neg = torch.cat([clip.tokenize(p) for p in words_noun]).to(self.device)

        with torch.no_grad():
            self.text_features_neg = []
            for i in range(0, len(text_inputs_neg), self.emb_batchsize):
                x = self.clip_model.encode_text(text_inputs_neg[i: i + self.emb_batchsize])
                self.text_features_neg.append(x)
            self.text_features_neg = torch.cat(self.text_features_neg, dim=0)
            self.text_features_neg /= self.text_features_neg.norm(dim=-1, keepdim=True)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'neg_emb': self.text_features_neg.cpu()}, os.path.join(path,'neg_embedding.pth'))

    def refine_negative_samples(self, path):
        with torch.no_grad():
            self.text_features_neg = self.text_features_neg.to(torch.float32)
            if self.pos_topk is not None:
                pos_mask = torch.zeros(len(self.text_features_neg), dtype=torch.bool, device=self.device)
                for i in range(self.text_features_pos.shape[0]):
                    sim = self.text_features_pos[i].unsqueeze(0) @ self.text_features_neg.T
                    _, ind = torch.topk(sim.squeeze(0), k=self.pos_topk)
                    pos_mask[ind] = 1
                self.text_features_pos = torch.cat([self.text_features_pos, self.text_features_neg[pos_mask]])

            neg_sim = []
            for i in range(0, len(self.text_features_neg), self.emb_batchsize):
                tmp = self.text_features_neg[i: i + self.emb_batchsize] @ self.text_features_pos.T
                tmp = tmp.to(torch.float32)
                sim = torch.quantile(tmp, q=self.pencentile, dim=-1)
                neg_sim.append(sim)
            neg_sim = torch.cat(neg_sim, dim=0)

            ind = torch.argsort(neg_sim)
            top_k_indices = ind[:int(len(ind) * self.neg_topk)].tolist()
            self.text_features_neg = self.text_features_neg[top_k_indices]
            self.selected_words = self.words_noun[ind]

        # If you want to dump the selected negative labels (with prompt), please uncomment these lines.
        with open(os.path.join(path,"selected_neg_labels.txt"), "w") as f:
            for i in top_k_indices:
                f.write("{}\n".format(self.words_noun[i]))


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


if __name__ == "__main__":
    backbone_name = "ViT-B/16"
    clip_model = load_clip_to_cpu(backbone_name)
    clip_neg_processor = CLIPTextProcessor(
        clip_model=clip_model, neg_topk=2000, pencentile = 2,
        wordnet_database='txtfiles', eg_dump_path = 'datasets/imagenet_neg/'
    )

