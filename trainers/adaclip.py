import os.path as osp
import os
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


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def compute_contrastive_loss(output, label, temperature=0.07):
    # Cosine similarity based contrastive loss (normalized embeddings)
    sim_matrix = torch.matmul(output, output.T) / temperature
    labels = label.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()  # 1 for positive pairs, 0 for negatives

    # Compute contrastive loss
    exp_sim = torch.exp(sim_matrix) * mask  # Keep positive pairs
    sum_exp_sim = torch.sum(torch.exp(sim_matrix), dim=1, keepdim=True)  # All pairs

    # Normalize the loss for each instance
    loss = -torch.log(exp_sim / sum_exp_sim)
    return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        """Compute loss for model. If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()


def entropy_select_topk(p, top_k, label):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(label.unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))


def entropy_select_topk_crop(output_local, top_k, label):
    """
    Select OOD samples based on top-K entropy and thresholds.
    """
    # Compute entropy
    p = F.softmax(output_local, dim=1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label).unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]
    
    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.LOCOOP.N_CTX
        ctx_init = cfg.TRAINER.LOCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.LOCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    

class Adapter(nn.Module):
    def __init__(self, clip_model, reduction=4, ratio=0.2):
        super(Adapter, self).__init__()
        c_in = clip_model.text_projection.shape[1]
        self.dtype = clip_model.dtype
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False).to(self.dtype),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False).to(self.dtype),
            nn.ReLU(inplace=True)
        )
        self.ratio = torch.tensor(ratio, dtype=self.dtype, requires_grad=False)

    def forward(self, x):
        ratio = self.ratio.to(x.device)
        # print(f"x device: {x.device}, ratio device: {ratio.device}")
        return self.fc(x.to(self.dtype)) * ratio + x.to(self.dtype) * (1-ratio)
        

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_token_embedding = clip_model.token_embedding
        
        if cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT:
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        else:
            temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            
            text_feat_file = f"text_features_{cfg.DATASET.NAME}_{cfg.DATASET.SUBSAMPLE_CLASSES}.pt"
            if os.path.exists(text_feat_file):
                print(f"Loading precomputed text features from {text_feat_file}")
                self.text_features = torch.load(text_feat_file).type(clip_model.dtype)
            else:
                print(f"Computing and saving text features to {text_feat_file}")
                with torch.no_grad():
                    self.text_features = clip_model.encode_text(tokenized_prompts).type(clip_model.dtype)
                    torch.save(self.text_features, text_feat_file)
        
        if cfg.TRAINER.ADAPTERS.USE_TEXT_ADAPTER:
            self.text_adapter = Adapter(clip_model, reduction=cfg.TRAINER.TEXT_ADAPTER.REDUCTION, ratio=cfg.TRAINER.TEXT_ADAPTER.RATIO)
        else:
            self.text_adapter = None
        
        if cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER:
            self.image_adapter = Adapter(clip_model, reduction=cfg.TRAINER.IMAGE_ADAPTER.REDUCTION, ratio=cfg.TRAINER.IMAGE_ADAPTER.RATIO)  # Assuming ImageAdapter is a class for the image adapter
        else:
            self.image_adapter = None

    def forward(self, image, use_ori_clip=False):
        # ===== get image features =====
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))
        
        if self.cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER and (not use_ori_clip):
            image_features = self.image_adapter(image_features)
            
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

        # ===== get text features =====
        if self.cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT:
            embeddings = self.prompt_learner()
            text_features = self.text_encoder(embeddings, self.tokenized_prompts)
        else:
            text_features = self.text_features.to(image.device)
            
        if self.cfg.TRAINER.ADAPTERS.USE_TEXT_ADAPTER and (not use_ori_clip):
            text_features = self.text_adapter(text_features)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_image_features @ text_features.T

        return logits, logits_local, image_features, local_image_features


@TRAINER_REGISTRY.register()
class AdaClip(TrainerX):
    """Local regularized Context Optimization (LoCoOp).
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPTERS.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.top_k = cfg.topk

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADAPTERS.PREC == "fp32" or cfg.TRAINER.ADAPTERS.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if ("prompt_learner" not in name) and ("adapter" not in name):
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS and cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        if cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT:
            if cfg.TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT:
                self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
                self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
                self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
            else:
                self.register_model("prompt_learner", self.model.prompt_learner)
                
        if cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER:
            if cfg.TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER:
                self.optim = build_optimizer(self.model.image_adapter, cfg.OPTIM)
                self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
                self.register_model("image_adapter", self.model.image_adapter, self.optim, self.sched)
            else:
                self.register_model("image_adapter", self.model.image_adapter)
                
        if cfg.TRAINER.ADAPTERS.USE_TEXT_ADAPTER:
            if cfg.TRAINER.ADAPTERS.TRAIN_TEXT_ADAPTER:
                self.optim = build_optimizer(self.model.text_adapter, cfg.OPTIM)
                self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
                self.register_model("text_adapter", self.model.text_adapter, self.optim, self.sched)
            else:
                self.register_model("text_adapter", self.model.text_adapter)

        self.scaler = GradScaler() if cfg.TRAINER.ADAPTERS.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
    def compute_class_prototypes(self):
        self.set_model_mode("eval")
        all_features, all_labels = [], []
        for epoch in range(2):
            for batch_idx, batch in enumerate(self.train_loader_x):
                image, label = self.parse_batch_train(batch)
                
                if isinstance(image, tuple) or isinstance(image, list):  # Two-crop augmentation scenario
                    global_crop, local_crop = image
                else:
                    global_crop = image

                with torch.no_grad():
                    _, _, feature, _ = self.model(global_crop.type(self.model.dtype))

                all_features.append(feature)
                all_labels.append(label)
            
        # Concatenate all features and labels
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        prototypes = torch.zeros(self.num_classes, all_features.shape[1]).type(self.model.dtype).to(self.device)
        for class_idx in range(self.num_classes):
            class_mask = (all_labels == class_idx)
            class_features = all_features[class_mask]
            
            if class_features.size(0) > 0:  # Avoid division by zero in case of empty class
                prototypes[class_idx] = class_features.mean(dim=0)  # Compute the mean feature vector
        
        self.model.prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.ADAPTERS.PREC

        use_amp = prec == "amp"
        with autocast(enabled=use_amp):
            if isinstance(image, tuple) or isinstance(image, list):  # Two-crop augmentation scenario
                global_crop, local_crop = image

                output, _, image_feats, _ = self.model(global_crop)
                output_local, _, image_feats_local, _ = self.model(local_crop)

                # Calculate loss (for global crop)
                loss_id = F.cross_entropy(output, label)
                
                if self.cfg.TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER:
                    with torch.no_grad():
                        output0, _, image_feats0, _ = self.model(global_crop, use_ori_clip=True)
                        output_local0, _, image_feats_local0, _ = self.model(local_crop, use_ori_clip=True)
                    # Contrastive loss 
                    ConLoss = SupConLoss(temperature=self.cfg.temp_ct)
                    loss_ct = ConLoss(torch.cat([image_feats.unsqueeze(1), image_feats_local.unsqueeze(1)], dim=1), labels=label)
                    # Distillation Loss
                    loss_dt = ( mse_loss(image_feats, image_feats0) + mse_loss(image_feats_local, image_feats_local0) ) * 10
                else:
                    loss_ct = torch.tensor(0.0, device=loss_id.device)
                    loss_dt = torch.tensor(0.0, device=loss_id.device)
                
                loss = loss_id + self.cfg.lambda_ct * loss_ct + self.cfg.lambda_dt * loss_dt
                # Calculate OOD regularization loss (for local crop)
                # loss_en = -entropy_select_topk(output_local, self.top_k, label)
                # loss = loss_id + self.lambda_value * loss_en
                
            else:
                output, output_local, _, _ = self.model(image)
                
                # calculate CoOp loss
                loss_id = F.cross_entropy(output, label)
                
                # calculate OOD regularization loss
                batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
                output_local = output_local.view(batch_size * num_of_local_feature, -1)
                loss_en = - entropy_select_topk(output_local, self.top_k, label.repeat_interleave(num_of_local_feature))

                # calculate total loss for LoCoOp
                loss = loss_id + self.lambda_value * loss_en
        
        if use_amp:
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_id": loss_id.item(),
            "loss_ct": loss_ct.item(),
            "loss_dt": loss_dt.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        if "two_crop" in self.cfg.INPUT.TRANSFORMS:
            input = [img.to(self.device) for img in input]
        else:
            input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None, module_name=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names() if module_name is None else [module_name]

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        else: 
            model_file = "model-best.pth.tar"

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if len(output) >= 2:
                output = output[0]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, data_loader, T):
        """Test-time OOD detection pipeline."""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score = []
        mcm_score = []
        im_score = []
        
        for batch_idx, (images, labels, *id_flag) in enumerate(data_loader):
            images = images.cuda()
            output, output_local, image_features, _ = self.model_inference(images)
            
            im_sim =  image_features @ self.model.prototypes.T
            im_sim, _ = torch.max(im_sim, dim=-1)
            im_score.append(-im_sim.cpu().numpy())
            
            output /= 100.0
            output_local /= 100.0
            
            smax_global = to_np(F.softmax(output/T, dim=-1))
            smax_local = to_np(F.softmax(output_local/T, dim=-1))
            
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_local_score = -np.max(smax_local, axis=(1, 2))
            
            mcm_score.append(mcm_global_score)
            glmcm_score.append(mcm_global_score+mcm_local_score)

        return concat(mcm_score).copy(), concat(glmcm_score).copy(), concat(im_score).copy()

    @torch.no_grad()
    def test_visualize(self, img_path, label):
        """code for visualization results"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        output, output_local = self.model_inference(image)

        num_regions = output_local.shape[1]
        label = torch.tensor(label).cuda()
        label_repeat = label.repeat_interleave(num_regions)
        output_local = F.softmax(output_local, dim=-1)

        output_local = output_local.view(num_regions, -1)

        # -----top 200--------
        pred_topk = torch.topk(output_local, k=200, dim=1)[1]
        contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)

        return contains_label

    @torch.no_grad()
    def tsne_visualize(self, img_path, label):
        """code for visualization results"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        output, output_local = self.model_inference(image)

        num_regions = output_local.shape[1]
        label = torch.tensor(label).cuda()
        label_repeat = label.repeat_interleave(num_regions)
        output_local = F.softmax(output_local, dim=-1)

        output_local = output_local.view(num_regions, -1)

        # -----top 200--------
        pred_topk = torch.topk(output_local, k=200, dim=1)[1]
        contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)

        return contains_label
    
    def train(self, args=None):
        """Generic training loops."""

        self.before_train()
        
        for self.epoch in range(self.start_epoch, self.max_epoch):
    
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            
            if self.epoch==0 or (self.epoch + 1) % 10 == 0:
                print(f"Running eval_ood at epoch {self.epoch + 1}")
                self.eval_ood(args)
        self.after_train()
        
    
    def eval_ood(self, args):
        self.set_model_mode("eval")
        self.compute_class_prototypes()
        
        if args.in_dataset in ['imagenet']:
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
    
        _, preprocess = clip_w_local.load(self.cfg.MODEL.BACKBONE.NAME)

        id_data_loader = set_val_loader(args, preprocess)
        in_score_mcm, in_score_gl, in_score_im = self.test_ood(id_data_loader, args.T)

        auroc_list_mcm, aupr_list_mcm, fpr_list_mcm = [], [], []
        auroc_list_gl, aupr_list_gl, fpr_list_gl = [], [], []
        auroc_list_im, aupr_list_im, fpr_list_im = [], [], []

        for idx, out_dataset in enumerate(out_datasets):
            print(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
            out_score_mcm, out_score_gl, out_score_im = self.test_ood(ood_loader, args.T)
            print(f"====== ID score: {stats.describe(in_score_mcm)}, {out_dataset} OD score: {stats.describe(out_score_mcm)}")
            print(f"====== ID score: {stats.describe(in_score_im)}, {out_dataset} OD score: {stats.describe(out_score_im)}")

            print(">>>MCM score")
            get_and_print_results(args, in_score_mcm, out_score_mcm, auroc_list_mcm, aupr_list_mcm, fpr_list_mcm)
            
            print(">>>IMG score")
            get_and_print_results(args, in_score_im, out_score_im, auroc_list_im, aupr_list_im, fpr_list_im)

            print("GL-MCM score")
            get_and_print_results(args, in_score_gl, out_score_gl, auroc_list_gl, aupr_list_gl, fpr_list_gl)

            if self.epoch == self.max_epoch - 1:
                plot_distribution(args, in_score_mcm, out_score_mcm, out_dataset, score='MCM')
                plot_distribution(args, in_score_gl, out_score_gl, out_dataset, score='GLMCM')

        print("MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_mcm), np.mean(auroc_list_mcm), np.mean(aupr_list_mcm)))
        print("MCM_IM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_im), np.mean(auroc_list_im), np.mean(aupr_list_im)))
        print("GL-MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_gl), np.mean(auroc_list_gl), np.mean(aupr_list_gl)))
        if not hasattr(self, 'num_batches'):
            self.num_batches = len(self.train_loader_x)
        wandb.log({"mcm/fpr": np.mean(fpr_list_mcm),
                   "mcm/auroc": np.mean(auroc_list_mcm), 
                   "mcm/aupr": np.mean(aupr_list_mcm)
                   }, step=(1 + self.epoch) * self.num_batches)
        
        wandb.log({"im/fpr": np.mean(fpr_list_im),
                   "im/auroc": np.mean(auroc_list_im),
                   "im/aupr": np.mean(aupr_list_im)
                   }, step=(1 + self.epoch) * self.num_batches)

        wandb.log({"gl-mcm/fpr": np.mean(fpr_list_gl),
                   "gl-mcm/auroc": np.mean(auroc_list_gl), 
                   "gl-mcm/aupr": np.mean(aupr_list_gl)
                   }, step=(1 + self.epoch) * self.num_batches)
        
        
    def eval_ood1(self, args):
        self.set_model_mode("eval")
        _, preprocess = clip_w_local.load(self.cfg.MODEL.BACKBONE.NAME)
        
        num_classes = 50
        out_dataset = 'iNaturalist'
        id_loader = set_val_loader(args, preprocess, subset=True, num_classes=num_classes)
        od_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)

        def get_feats(data_loader, T=1, datum=False):
            mcm_scores = []
            image_feats = []
            all_labels = []
            self.set_model_mode("eval")
            self.evaluator.reset()
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    if datum:
                        images, labels = batch["img"], batch["label"]
                    else:
                        images, labels = batch
                    
                    if images.ndim == 5:
                        labels = labels.repeat_interleave(images.shape[1]) 
                        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])  # Flatten the second dimension
                        
                    images = images.cuda()
                    output, _, image_feature, _ = self.model_inference(images)
                    
                    output /= 100.0
                    smax_global = F.softmax(output/T, dim=-1)
                    max_values, _ = torch.max(smax_global, dim=1)
                    mcm_global_score = -max_values
                    
                    mcm_scores.append(mcm_global_score.cpu())
                    image_feats.append(image_feature.cpu())
                    all_labels.append(labels)
                    
                    del images, output, image_feature, smax_global, max_values, mcm_global_score, labels
                    torch.cuda.empty_cache()
                
            return torch.cat(mcm_scores, dim=0), torch.cat(image_feats, dim=0), torch.cat(all_labels, dim=0)
        
        id_mcm, id_feats, id_labels = get_feats(id_loader)
        od_mcm, od_feats, od_labels = get_feats(od_loader)
        id_mcm_tr, id_feats_tr, id_labels_tr = get_feats(self.train_loader_x, datum=True)
        
        class_prototypes = []
        for cls in range(num_classes):
            class_indices = (id_labels_tr == cls).nonzero(as_tuple=True)[0]
            class_features = id_feats_tr[class_indices]
            class_prototypes.append(class_features.mean(dim=0))

        proto = torch.stack(class_prototypes, dim=0)
        
        # === text embedding 
        with torch.no_grad():
            embeddings = self.model.prompt_learner()
            text_feats = self.model.text_encoder(embeddings, self.model.tokenized_prompts)
            
        # ============== 
        import matplotlib.pyplot as plt
        text_feats = text_feats[:num_classes]
        all_feats = torch.cat([id_feats_tr, id_feats, od_feats, text_feats], dim=0)
        od_labels = torch.full((len(od_feats),), -1, dtype=torch.long, device=id_labels.device)
        text_labels = torch.arange(num_classes, dtype=torch.long, device=id_labels.device)
        
        all_labels = torch.cat([id_labels_tr, id_labels, od_labels, text_labels])
        all_feats, all_labels = all_feats.cpu().numpy(), all_labels.cpu().numpy()

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_feats)

        plt.figure(figsize=(12, 10), dpi=300)

        # Plot in-distribution training features
        plt.scatter(tsne_results[:len(id_feats_tr), 0], tsne_results[:len(id_feats_tr), 1],
                    c=all_labels[:len(id_feats_tr)], cmap='tab10', label='In-distribution (Train)', alpha=0.6, marker='o', s=20)

        # Plot in-distribution validation features
        start_idx = len(id_feats_tr)
        end_idx = start_idx + len(id_feats)
        plt.scatter(tsne_results[start_idx:end_idx, 0], tsne_results[start_idx:end_idx, 1],
                    c=all_labels[start_idx:end_idx], cmap='tab10', label='In-distribution (Val)', alpha=0.6, marker='v', s=20)

        # Plot out-of-distribution features
        start_idx = end_idx
        end_idx = start_idx + len(od_feats)
        plt.scatter(tsne_results[start_idx:end_idx, 0], tsne_results[start_idx:end_idx, 1],
                    c='gray', label='Out-of-distribution', alpha=0.6, marker='x', s=20)

        # Plot text features
        start_idx = end_idx
        plt.scatter(tsne_results[start_idx:, 0], tsne_results[start_idx:, 1],
                    c=all_labels[start_idx:] % 10, cmap='tab10', label='Text-feats', alpha=0.6, marker='s', s=20)

        # Labeling and legend
        plt.title('t-SNE Visualization of Image and Text Embeddings')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        
        plt.savefig('tsne_plot.png')
        
        id_sim = id_feats @ id_feats_tr.T
        
        od_sim = od_feats @ id_feats_tr.T
            
        
        self.compute_class_prototypes()
        
        id_sim = id_feats @ self.model.prototypes.cpu().T
        od_sim = od_feats @ self.model.prototypes.cpu().T
        
        id_sim_max, _ = torch.max(id_sim, dim=-1)
        od_sim_max, _ = torch.max(od_sim, dim=-1)
        
        
        stats.describe(id_mcm.cpu().numpy())
        stats.describe(od_mcm.cpu().numpy())
        
        stats.describe(id_sim_max.cpu().numpy())
        stats.describe(od_sim_max.cpu().numpy())
        
        tp = od_feats @ proto[950:1000].T
        tp1, _ = torch.max(tp, dim=-1)
        stats.describe(tp1.cpu().numpy())
        
        
        
        torch.save({'id_mcm_score': id_mcm, 'id_feats': id_feats}, 'id_features.pth')
        torch.save({'od_mcm_score': od_mcm, 'od_feats': od_feats}, 'od_features.pth')
        
        
