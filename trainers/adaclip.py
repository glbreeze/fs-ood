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
from utils.detection_util import get_and_print_results, get_feats
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
        
        # ===== postive and negative text embeddings 
        text_feat_file = f"datasets/text_features_{cfg.DATASET.NAME}_{cfg.DATASET.SUBSAMPLE_CLASSES}.pth"
        if os.path.exists(text_feat_file):
            print(f"Loading precomputed text features from {text_feat_file}")
            self.text_features = torch.load(text_feat_file).type(clip_model.dtype)
        else:
            print(f"Computing and saving text features to {text_feat_file}")
            with torch.no_grad():
                temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
                prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
                self.text_features = clip_model.encode_text(tokenized_prompts).type(clip_model.dtype)
                self.text_features = self.text_features/self.text_features.norm(dim=-1, keepdim=True)
                torch.save(self.text_features, text_feat_file)
        self.classnames = classnames
                
        dump_dict = torch.load('datasets/imagenet_neg/neg_embedding.pth')
        self.text_features_neg = dump_dict['neg_emb'].type(clip_model.dtype)
        self.classnames_neg = dump_dict['neg_name']
        print('Load computed negative labels from :datasets/imagenet_neg/neg_embedding.pth')
        
        # ===== init text and image adapter
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
        text_features = self.text_features.to(image.device)
            
        if self.cfg.TRAINER.ADAPTERS.USE_TEXT_ADAPTER and (not use_ori_clip):
            text_features = self.text_adapter(text_features)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_image_features @ text_features.T

        return logits, logits_local, image_features, local_image_features
    
    def refine_negative_samples(self, args, proto_filter=True):
        with torch.no_grad():
            
            neg_sims = []
            max_sims = []
            emb_batchsize=1000
            for i in range(0, len(self.text_features_neg), emb_batchsize):
                tmp = self.text_features_neg[i: i + emb_batchsize] @ self.text_features.T
                tmp = tmp.to(torch.float32)
                
                max_sim, _ = tmp.max(dim=-1)
                max_sims.append(max_sim)
                
                sim = torch.quantile(tmp, q=args.percentile, dim=-1)
                neg_sims.append(sim)
            neg_sims = torch.cat(neg_sims, dim=0)
            max_sims = torch.cat(max_sims, dim=0)

            pct_low = torch.quantile(neg_sims, q=args.pct_low)
            pct_high = torch.quantile(neg_sims, q=args.pct_low + args.pct)
            
            # Filter negatives: Keep only those within similarity thresholds
            valid_indices = (max_sims <= 0.95) & (neg_sims >= pct_low) & (neg_sims <= pct_high)
            valid_indices = torch.nonzero(valid_indices, as_tuple=True)[0].to(self.text_features_neg.device)
            self.text_features_neg = self.text_features_neg[valid_indices]
            self.selected_words = [self.classnames_neg[i] for i in valid_indices.tolist()]
            
            # Filter negatives: Keep only those which are not similar to prototypes
            if proto_filter:
                sim_proto = self.prototypes @ torch.cat([self.text_features, self.text_features_neg], dim=0).T
                _, id_max = torch.max(sim_proto, dim=-1)
                mask = id_max >= len(self.text_features)
                indices_to_drop = set((id_max[mask] - len(self.text_features)).tolist())
                indices_to_keep = torch.tensor([i for i in range(len(self.text_features_neg)) if i not in indices_to_drop], device=id_max.device)
                self.text_features_neg = self.text_features_neg[indices_to_keep]
                self.selected_words = [self.selected_words[i] for i in indices_to_keep.tolist()]

            # Save filtered embeddings and words
            torch.save({'neg_emb': self.text_features_neg.cpu(), 'neg_name': self.selected_words},
                    os.path.join('datasets/imagenet_neg', f'neg_embedding_l{args.pct_low}p{args.pct}.pth'))

            # Save selected negative words
            with open(os.path.join('datasets/imagenet_neg', f"selected_neg_l{args.pct_low}p{args.pct}.txt"), "w") as f:
                for item in self.selected_words:
                    f.write("{}\n".format(item))
                    
    def to_device(self, device):
        """Moves the model and related tensors to the specified device."""
        self.to(device)
        self.text_features = self.text_features.to(device)
        self.text_features_neg = self.text_features_neg.to(device)


def is_label_in_topk(logits, labels, k):
    """
    Check if the ground truth label is in the top-K labels.
    Args:
        logits (torch.Tensor): The logits tensor of shape (batch_size, num_classes).
        labels (torch.Tensor): The ground truth labels tensor of shape (batch_size).
        k (int): The number of top elements to select.
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size) indicating if the ground truth label is in the top-K labels.
    """
    _, topk_indices = torch.topk(logits, k, dim=1)
    return torch.eq(topk_indices, labels.view(-1, 1)).any(dim=1)


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
            clip_model = clip_model.float()  # CLIP's default precision is fp16
        elif cfg.TRAINER.ADAPTERS.PREC == "fp16":
            clip_model.half()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model.to_device(self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if ("prompt_learner" not in name) and ("adapter" not in name):
                param.requires_grad_(False)
                
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
                    logits, _, feature, _ = self.model(global_crop.type(self.model.dtype))
                    probs = softmax(logits, dim=-1)
                    keep_prob_mask = probs[torch.arange(probs.size(0)), label] > 0.05
                    
                    topk_check = is_label_in_topk(logits, label, self.top_k)

                    keep_mask = keep_prob_mask & topk_check

                    feature = feature[keep_mask]
                    label = label[keep_mask]

                all_features.append(feature/feature.norm(dim=-1, keepdim=True))
                all_labels.append(label)
            
        # Concatenate all features and labels
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        prototypes = torch.zeros(self.num_classes, all_features.shape[1], dtype=self.model.dtype, device=self.device)
        covariances = torch.zeros(self.num_classes, all_features.shape[1], dtype=self.model.dtype, device=self.device)

        for class_idx in range(self.num_classes):
            class_mask = (all_labels == class_idx)
            class_features = all_features[class_mask]

            if class_features.size(0) > 0:
                prototypes[class_idx] = class_features.mean(dim=0)
                covariances[class_idx] = class_features.var(dim=0, unbiased=False) + 1e-7

        self.model.prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
        self.model.class_covariances = covariances

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
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def train(self, args=None):
        self.before_train()
        
        for self.epoch in range(self.start_epoch, self.max_epoch):
    
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            
            if self.epoch==0 or (self.epoch + 1) % 10 == 0:
                print(f"Running eval_ood at epoch {self.epoch + 1}")
                self.eval_ood(args)
        self.after_train()

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
        
    def eval_ood(self, args):
        self.set_model_mode("eval")
        self.compute_class_prototypes()
        torch.save({'proto': self.model.prototypes, 'var': self.model.class_covariances}, 'datasets/proto_var.pth') # prototypes for ID
        self.model.refine_negative_samples(args)

        def get_id_score(data_loader, T = 0.01):
            all_feats, all_labels = get_feats(data_loader, self.model)
            maha = compute_mahalanobis_distance(all_feats, self.model.prototypes, self.model.class_covariances)
            maha, idx = maha.min(dim=-1)
            sim = all_feats @ torch.cat([self.model.text_features, self.model.text_features_neg], dim=0).T
            id_scores = F.softmax(sim / T, dim=1)
            id_score = id_scores[torch.arange(len(id_scores)), idx]
            return maha, idx, sim, id_score, all_labels

        _, preprocess = clip_w_local.load(self.cfg.MODEL.BACKBONE.NAME)

        id_data_loader = set_val_loader(args, preprocess)
        id_maha, id_idx, id_sim, id_score, id_labels = get_id_score(id_data_loader)
        torch.save({'id_maha': id_maha, 'id_idx': id_idx, 'id_sim': id_sim, 'id_labels': id_labels}, 'datasets/id_maha.pth')
        id_score = id_score.cpu().numpy()
        
        auroc_list, aupr_list, fpr_list = [], [], []
        if args.in_dataset in ['imagenet']:
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
        for idx, out_dataset in enumerate(out_datasets):
            print(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
            od_maha, od_idx, od_sim, od_score, _ = get_id_score(ood_loader)
            torch.save({'od_maha': od_maha, 'od_idx': od_idx, 'od_sim': od_sim}, f'datasets/od_maha_{out_dataset}.pth')
            od_score = od_score.cpu().numpy()
            print(f"====== ID score: {stats.describe(id_score)}, {out_dataset} OD score: {stats.describe(od_score)}")
            
            get_and_print_results(-id_score, -od_score, auroc_list, aupr_list, fpr_list)
            plot_distribution(args, -id_score, -od_score, out_dataset, score='new')

        print("MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list), np.mean(auroc_list), np.mean(aupr_list)))
        
    def eval_ood1(self, args):
        self.set_model_mode("eval")
        self.compute_class_prototypes()
        torch.save({'proto': self.model.prototypes, 'var': self.model.class_covariances}, 'datasets/proto_var.pth') # prototypes for ID
        self.model.refine_negative_samples(args)
        
         
        dt = torch.load('datasets/id.pth')
        id_feats, id_labels = dt['id_feats'], dt['id_labels']
        dt = torch.load('datasets/od.pth')
        od_feats, od_labels = dt['od_feats'], dt['od_labels']
                    
        # _, preprocess = clip_w_local.load(self.cfg.MODEL.BACKBONE.NAME)
        # out_dataset = 'iNaturalist'
        # id_loader = set_val_loader(args, preprocess)
        # od_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)

        
        
        # id_mcm, id_feats, id_labels = get_feats(id_loader)
        # od_mcm, od_feats, od_labels = get_feats(od_loader)
        
        # ===== compute mahalanobis_dist =====       
        id_maha = compute_mahalanobis_distance(id_feats, self.model.prototypes.cpu(), self.model.class_covariances.cpu())
        id_maha, id_idx = id_maha.min(dim=-1)
        id_sim = id_feats @ torch.cat([self.model.text_features, self.model.text_features_neg], dim=0).T.cpu()
        torch.save({'id_maha': id_maha, 'id_idx': id_idx, 'id_sim': id_sim}, 'datasets/id_maha.pth')
        
        
        od_maha = compute_mahalanobis_distance(od_feats, self.model.prototypes.cpu(), self.model.class_covariances.cpu())
        od_maha, od_idx = od_maha.min(dim=-1) 
        od_sim = od_feats @ torch.cat([self.model.text_features, self.model.text_features_neg], dim=0).T.cpu()
        torch.save ({'od_maha': od_maha, 'od_idx': od_idx, 'od_sim': od_sim}, 'datasets/od_maha.pth')
        
        del id_feats, od_feats
        id_mcm_tr, id_feats_tr, id_labels_tr = get_feats(self.train_loader_x, datum=True)
        id_maha_tr = compute_mahalanobis_distance(id_feats_tr, self.model.prototypes.cpu(), self.model.class_covariances.cpu())
        id_txt_score = id_feats_tr @ torch.cat([self.model.text_features, self.model.text_features_neg], dim=0).T.cpu()
        torch.save({'id_maha_tr': id_maha_tr, 'id_txt_score': id_txt_score}, 'datasets/id_maha_tr.pt')
        
        # ======
        
        sim_od = self.model.prototypes.cpu() @ od_feats.T
        sim_od_pt = torch.quantile(sim_od, q=0.95, dim=-1)
        sim_od_max, _ = sim_od.max(dim=-1)
        
        sim_id = self.model.prototypes.cpu() @ id_feats.T
        
        num_classes = self.model.prototypes.shape[0]
        sim_id_min = torch.full((num_classes,), float('inf'), device=sim_id.device)

        for class_idx in range(num_classes):
            class_mask = (id_labels == class_idx)
            sim_id_min[class_idx] = sim_id[class_idx, class_mask].min()
        
        text_feats = torch.cat([self.model.text_features, self.model.text_features_neg], dim=0)
        sim_txt = self.model.prototypes.cpu() @ text_feats.T
        sim_txt_sfm = F.softmax(sim_txt/0.01, dim=-1)
        
        sim_txt_max, _ = torch.max(sim_txt_sfm[:, :1000], dim=-1)
        sim_txt_max1, _ = torch.max(sim_txt_sfm[:, 1000:], dim=-1)
        
        # =====
        
        
        id_mcm_tr, id_feats_tr, id_labels_tr = get_feats(self.train_loader_x, datum=True)
        
        num_classes = 1000
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


def compute_mahalanobis_distance(feats, prototypes, class_covariances, batch_size=1024):
    num_samples, feature_dim = feats.shape
    num_classes = prototypes.shape[0]
    mahalanobis_dist = torch.zeros((num_samples, num_classes), device=feats.device)
    
    # Process in batches to reduce memory usage
    for i in range(0, num_samples, batch_size):
        diff = feats[i:i + batch_size, None, :] - prototypes[None, :, :]  # Shape: (batch_size, num_classes, feature_dim)
        mahalanobis_dist[i:i + batch_size] = (diff ** 2 / class_covariances[None, :, :]).sum(dim=-1).sqrt()

    return mahalanobis_dist


