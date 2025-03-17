import os.path as osp
import os, gc
import wandb
import random
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
from clip_w_local.model import LayerNorm
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .utils import SupConLoss, CUSTOM_TEMPLATES, entropy_select_topk, filter_positive_negative
from .locoop import PromptLearner
import numpy as np
from tqdm import tqdm
from PIL import Image

def debug_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2} MB")


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
        x = self.transformer(x)
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
        self.dtype = clip_model.dtype
        self.classnames = classnames
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.clip_model = clip_model
        self.clip_token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale

        # ===== postive and negative text embeddings 
        dump_dict = torch.load('datasets/imagenet_neg/neg_embedding_refined.pth')  # neg_embedding.pth
        self.text_features_neg = dump_dict['neg_emb'].type(self.dtype)
        self.classnames_neg = dump_dict['neg_name']
        print('Load computed negative labels from :datasets/imagenet_neg/neg_embedding_refined.pth')
        
        subset_num = 800
        self.text_features_neg = self.text_features_neg[:subset_num]
        self.text_features_neg = self.text_features_neg / self.text_features_neg.norm(dim=-1, keepdim=True)
        self.classnames_neg = self.classnames_neg[:subset_num]
        
        self.load_clip_text_features(classnames)

        # ===== init text prompt learner =====
        if cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT:
            self.init_text_prompt_learner(classnames + self.classnames_neg if self.cfg.TRAINER.LOCOOP.NEG else [])
            
        # ===== init image adapter =====
        if cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER:
            self.image_adapter = Adapter(clip_model, reduction=cfg.TRAINER.IMAGE_ADAPTER.REDUCTION, ratio=cfg.TRAINER.IMAGE_ADAPTER.RATIO) 
    
    def init_text_prompt_learner(self, classnames):
        self.prompt_learner = PromptLearner(self.cfg, self.clip_model.to(torch.device("cpu")), classnames=classnames)
    
    def load_text_prompt_features(self):
        with torch.no_grad():
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            self.text_features_all = text_features / text_features.norm(dim=-1, keepdim=True)
            print("Text prompt features loaded and normalized successfully.")
        
    def load_clip_text_features(self, classnames):
        self.num_pos = len(classnames)
        text_feat_file = f"datasets/text_features_{self.cfg.DATASET.NAME}_{self.cfg.DATASET.SUBSAMPLE_CLASSES}.pth"
        if os.path.exists(text_feat_file):
            print(f"Loading precomputed text features from {text_feat_file}")
            self.text_features = torch.load(text_feat_file).type(self.clip_model.dtype)
            self.text_features = self.text_features/self.text_features.norm(dim=-1, keepdim=True)
        else:
            print(f"Computing and saving text features to {text_feat_file}")
            with torch.no_grad():
                temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
                prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
                self.text_features = self.clip_model.encode_text(tokenized_prompts).type(self.clip_model.dtype)
                self.text_features = self.text_features/self.text_features.norm(dim=-1, keepdim=True)
                torch.save(self.text_features, text_feat_file)

    def forward(self, image, use_ori_clip=False):
        # ===== get image features =====
        self.image_encoder.eval()
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
        
        if self.cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER and (not use_ori_clip):
            image_features = self.image_adapter(image_features)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

        # ===== get text features =====
        if use_ori_clip: 
            text_features = torch.cat([self.text_features, self.text_features_neg], dim=0).to(image.device)
        elif self.cfg.TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT and (not use_ori_clip):
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        elif self.cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT and (not self.cfg.TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT):
            text_features = self.text_features_all.to(image.device)
        
        # ===== calculate logits =====
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features.to(text_features.device) @ text_features.t()
        return logits, image_features
    
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
            self.classnames_neg = [self.classnames_neg[i] for i in valid_indices.tolist()]
            
            # Filter negatives: Keep only those which are not similar to prototypes
            if proto_filter and hasattr(self, "prototypes"):
                sim_proto = self.prototypes.to(self.text_features.device) @ torch.cat([self.text_features, self.text_features_neg], dim=0).T
                _, id_max = torch.max(sim_proto, dim=-1)
                mask = id_max >= len(self.text_features)
                indices_to_drop = set((id_max[mask] - len(self.text_features)).tolist())
                indices_to_keep = torch.tensor([i for i in range(len(self.text_features_neg)) if i not in indices_to_drop], device=id_max.device)
                self.text_features_neg = self.text_features_neg[indices_to_keep]
                self.classnames_neg = [self.classnames_neg[i] for i in indices_to_keep.tolist()]
            
            num_neg_labels = min(4000, len(self.classnames_neg))
            sample_indices = random.sample(range(len(self.classnames_neg)), num_neg_labels)
            self.text_features_neg = self.text_features_neg[sample_indices].type(self.dtype)
            self.classnames_neg = [self.classnames_neg[i] for i in sample_indices]
            self.num_neg = len(self.classnames_neg)

            print(f"Number of negative samples after filtering: {self.num_neg}")
            torch.save({'neg_emb': self.text_features_neg.cpu(), 'neg_name': self.classnames_neg},
                    os.path.join('datasets/imagenet_neg', f'neg_embedding_l{args.pct_low}p{args.pct}.pth'))

            # Save selected negative words
            with open(os.path.join('datasets/imagenet_neg', f"selected_neg_l{args.pct_low}p{args.pct}.txt"), "w") as f:
                for item in self.classnames_neg:
                    f.write("{}\n".format(item))
                    
    def to_device(self, device):
        """Moves the model and related tensors to the specified device."""
        # self.to(device)
        self.image_encoder.to(device)
        self.text_encoder.to(device)
        # self.text_features = self.text_features.to(device)
        # self.text_features_neg = self.text_features_neg.to(device)
        
        if hasattr(self, "prompt_learner"):
            self.prompt_learner.to(device)
            self.prompt_learner.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)

        if hasattr(self, "image_adapter"):
            self.image_adapter.to(device)

        for name, buf in self.named_buffers():
            setattr(self, name, buf.to(device))

        print(f"Model and related tensors moved to {device}")


def is_label_in_topk(logits, labels, k):
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

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADAPTERS.PREC == "fp32" or cfg.TRAINER.ADAPTERS.PREC == "amp":
            self.clip_model = self.clip_model.float()  # CLIP's default precision is fp16
        elif cfg.TRAINER.ADAPTERS.PREC == "fp16":
            self.clip_model = self.clip_model.half()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, self.clip_model)
        self.model.to_device(self.device)
        
        for name, module in self.model.named_modules():
            if isinstance(module, LayerNorm):
                module.float()

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if ("prompt_learner" not in name) and ("adapter" not in name):
                param.requires_grad_(False)
        
        if cfg.MODEL.INIT_WEIGHTS:
            if cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT and not cfg.TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT:
                self.load_model(cfg.MODEL.INIT_WEIGHTS, epoch=cfg.MODEL.INIT_EPOCH, module_name='prompt_learner')
                self.model.load_text_prompt_features
                
        # ============ define optimzer ============
        if cfg.TRAINER.ADAPTERS.USE_IMAGE_ADAPTER:
            if cfg.TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER:
                self.optim_img = build_optimizer(self.model.image_adapter, cfg.OPTIM)
                self.sched_img = build_lr_scheduler(self.optim_img, cfg.OPTIM)
                self.register_model("image_adapter", self.model.image_adapter, self.optim_img, self.sched_img)
            else:
                self.register_model("image_adapter", self.model.image_adapter)
                
        if cfg.TRAINER.ADAPTERS.USE_TEXT_PROMPT:
            if cfg.TRAINER.ADAPTERS.TRAIN_TEXT_PROMPT:
                self.optim_txt = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
                self.sched_txt = build_lr_scheduler(self.optim_txt, cfg.OPTIM)
                self.register_model("prompt_leaner", self.model.prompt_learner, self.optim_txt, self.sched_txt)
            else:
                self.register_model("prompt_learner", self.model.prompt_learner)

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
                image = image.view(-1, *image.shape[2:])
                label = torch.repeat_interleave(label, self.cfg.INPUT.NUM_CROPS)

                with torch.no_grad():
                    logits, feature = self.model(image.type(self.model.dtype), use_ori_clip=True)
                    probs = F.softmax(logits, dim=-1)
                    pos_feat, pos_label, neg_feat, neg_label = filter_positive_negative(logits, feature, label, self.cfg.INPUT.NUM_CROPS, top_k_percent=0.2)
                    
                    # keep_prob_mask = probs[torch.arange(probs.size(0)), label] > 0.02
                    # topk_check = is_label_in_topk(logits, label, self.cfg.topk)
                    # keep_mask = keep_prob_mask & topk_check
                    # feature = feature[keep_mask]
                    # label = label[keep_mask]
                all_features.append(pos_feat/pos_feat.norm(dim=-1, keepdim=True))
                all_labels.append(pos_label)
                
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
        torch.save({"prototypes": self.model.prototypes, "class_covariances": self.model.class_covariances}, "datasets/model_prototypes_covariances.pth")

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        image, label = batch["img"], batch["label"]
        label = label.to(self.device)
        image = image.to(self.device)
        
        if self.cfg.INPUT.NUM_CROPS > 1: 
            image = image.view(-1, *image.shape[2:])
            label = torch.repeat_interleave(label, self.cfg.INPUT.NUM_CROPS)

            with torch.no_grad():
                output_fz, image_feats_fz = self.model(image, use_ori_clip=True)
                pos_img, pos_label, neg_img, neg_label = filter_positive_negative(output_fz, image, label, self.cfg.INPUT.NUM_CROPS, top_k_percent=0.1)
                del output_fz, image_feats_fz
                gc.collect()
                torch.cuda.empty_cache()
            
            for optim in self._optims.values():
                optim.zero_grad()

            if pos_img.numel() > 0:
                output_pos, image_feats_pos = self.model(pos_img)
                loss_pos = F.cross_entropy(output_pos, pos_label)
                loss_pos_pt = 1 - F.cosine_similarity(image_feats_pos, self.model.prototypes[pos_label], dim=-1).mean() \
                            if self.cfg.TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER else torch.tensor(0.0).to(loss_pos.device)
                (loss_pos + self.cfg.TRAINER.ADAPTERS.LAMBDA * loss_pos_pt).backward()

            if neg_img.numel() > 0:
                output_neg, image_feats_neg = self.model(neg_img)
                prob_neg = F.softmax(output_neg, dim=-1)
                
                loss_neg = -torch.log((prob_neg[:, -len(self.model.text_features_neg):].sum(dim=-1) + 1e-9) /
                                    (prob_neg.sum(dim=-1) + 1e-9)).mean()
                loss_neg_pt = F.cosine_similarity(image_feats_neg, self.model.prototypes[neg_label], dim=-1).mean() \
                            if self.cfg.TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER else torch.tensor(0.0).to(loss_neg.device)
                
                (self.cfg.TRAINER.ADAPTERS.LAMBDA_NEG*(loss_neg + self.cfg.TRAINER.ADAPTERS.LAMBDA * loss_neg_pt)).backward()
            
            for optim in self._optims.values():
                optim.step()

        loss_summary = {
            "loss_pos": loss_pos.item(),
            "loss_pos_pt": loss_pos_pt.item(),
            "loss_neg": loss_neg.item(),
            "loss_neg_pt": loss_neg_pt.item(),
            "acc": compute_accuracy(output_pos[:, :1000], pos_label)[0].item(),
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

        names = self.get_model_names() if module_name is None else [module_name]

        model_file = "model.pth.tar-" + str(epoch) if epoch is not None else "model-best.pth.tar"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def load_proto(self, args):
        # self.compute_class_prototypes()
        # torch.save({'proto': self.model.prototypes, 'var': self.model.class_covariances}, 'datasets/proto_var.pth')
        checkpoint = torch.load('datasets/proto_var.pth')
        self.model.prototypes = checkpoint['proto'].type(self.model.dtype)
        self.model.class_covariances = checkpoint['var'].type(self.model.dtype)
        
        # import pdb; pdb.set_trace()
        # self.model.refine_negative_samples(args)
        # torch.save({'neg_emb': self.model.text_features_neg, 'neg_name': self.model.classnames_neg}, 'datasets/imagenet_neg/neg_embedding_refined.pth')
        # print('Saved refined negative labels')
        
        # if self.cfg.TRAINER.LOCOOP.NEG:
        #     self.model.init_text_prompt_learner(self.model.classnames + self.model.classnames_neg)
        #     self.model.to_device(self.device)
        # del self.model.clip_model
        # torch.cuda.empty_cache()
        
    def train(self, args=None):
        if self.cfg.TRAINER.ADAPTERS.TRAIN_IMAGE_ADAPTER:
            self.load_proto(args)
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            
            if self.epoch==0 or (self.epoch + 1) % 5 == 0:
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
        # self.compute_class_prototypes()
        # torch.save({'proto': self.model.prototypes, 'var': self.model.class_covariances}, 'datasets/proto_var.pth') # prototypes for ID
        # self.model.refine_negative_samples(args)
        if not hasattr(self.model, "prototypes"):
            self.load_proto(args)

        def get_id_score(data_loader, T = 1):
            all_feats, all_labels, all_sim = get_feats(data_loader, self.model)
            
            maha = compute_mahalanobis_distance(all_feats.cuda(), self.model.prototypes.cuda(), self.model.class_covariances.cuda())
            maha = maha.cpu()
            maha, idx = maha.min(dim=-1)
            pred_probs = F.softmax(all_sim / T, dim=1)
            id_score = pred_probs[torch.arange(len(pred_probs)), idx]

            id_score1 = pred_probs[:, -len(self.model.text_features_neg):].sum(dim=-1)

            return maha, id_score, id_score1, all_labels
            
        # Load the model and preprocessing pipeline
        _, preprocess = clip_w_local.load(self.cfg.MODEL.BACKBONE.NAME)

        id_data_loader = set_val_loader(args, preprocess)
        id_maha, id_score, id_score1, all_labels = get_id_score(id_data_loader)
        id_score = id_score.cpu().numpy()    # The larger the better
        id_score1 = id_score1.cpu().numpy()  # The smaller the better
        
        auroc_list, aupr_list, fpr_list = [], [], []
        auroc_list1, aupr_list1, fpr_list1 = [], [], []
        
        if args.in_dataset in ['imagenet']:
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
        for idx, out_dataset in enumerate(out_datasets):
            print(f"Evaluting OOD dataset {out_dataset}...")
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
            od_maha, od_score, od_score1, _ = get_id_score(ood_loader)
            od_score = od_score.cpu().numpy()
            od_score1 = od_score1.cpu().numpy()
            # torch.save({'od_maha': od_maha, 'od_idx': od_idx, 'od_sim': od_sim}, f'datasets/od_maha_{out_dataset}.pth')
            
            # print(f"------ ID score: {stats.describe(id_score)}, {out_dataset} OOD score: {stats.describe(od_score)}")
            # print(f"------ ID score 1: {stats.describe(id_score1)}, {out_dataset} OOD score 1: {stats.describe(od_score1)}")

            fpr, auroc, aupr = get_and_print_results(-id_score, -od_score, auroc_list, aupr_list, fpr_list)
            fpr1, auroc1, aupr1 = get_and_print_results(id_score1, od_score1, auroc_list1, aupr_list1, fpr_list1)
            # print(f"=======> OOD data {out_dataset} Score@0, FPR:{fpr}, AUROC:{auroc}, AURPC:{aupr}")
            # print(f"=======> OOD data {out_dataset} Score@1, FPR:{fpr1}, AUROC:{auroc1}, AURPC:{aupr1}")
            # plot_distribution(args, -id_score, -od_score, out_dataset, score='new')

        print("OOD avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list), np.mean(auroc_list), np.mean(aupr_list)))
        print("OOD1 avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list1), np.mean(auroc_list1), np.mean(aupr_list1)))
        wandb.log({
            "ood/fpr": np.mean(fpr_list), 
            "ood/auroc": np.mean(auroc_list), 
            "ood/aupr": np.mean(aupr_list), 
            "ood/fpr1": np.mean(fpr_list1), 
            "ood/auroc1": np.mean(auroc_list1), 
            "ood/aupr1": np.mean(aupr_list1)
        }, step=(1 + self.epoch) * self.num_batches)
        
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


