import torch

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


def filter_positive_negative(
    logits: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, 
    num_crops: int, top_k_percent: float = 0.2
):
    num_samples = logits.shape[0] // num_crops  # Compute number of original samples

    logits = logits.view(num_samples, num_crops, -1)  # [num_samples, num_crops, num_classes]
    labels = labels.view(num_samples, num_crops)      # [num_samples, num_crops]
    images = images.view(num_samples, num_crops, *images.shape[1:])  # [num_samples, num_crops, C, H, W]

    # Get logits corresponding to ground truth labels
    ground_truth_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [num_samples, num_crops]

    # Sort the logits for each sample (descending order)
    sorted_logits, sorted_indices = torch.sort(ground_truth_logits, descending=True, dim=-1)  # [num_samples, num_crops]

    num_positive = max(1, int(num_crops * top_k_percent))
    pos_indices = sorted_indices[:, :num_positive]  # [num_samples, num_positive]
    neg_indices = sorted_indices[:, -num_positive:] # [num_samples, num_positive]

    pos_samples = images.gather(dim=1, index=pos_indices.unsqueeze(-1).expand(-1, -1, *images.shape[2:])).flatten(0, 1)
    pos_labels = labels.gather(dim=1, index=pos_indices).flatten()

    neg_samples = images.gather(dim=1, index=neg_indices.unsqueeze(-1).expand(-1, -1, *images.shape[2:])).flatten(0, 1)
    neg_labels = labels.gather(dim=1, index=neg_indices).flatten()

    return pos_samples, pos_labels, neg_samples, neg_labels



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