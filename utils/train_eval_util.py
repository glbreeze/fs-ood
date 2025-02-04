import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import clip_w_local
from collections import defaultdict


def set_model_clip(args):
    model, _ = clip_w_local.load(args.CLIP_ckpt)

    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return model, val_preprocess


def set_val_loader(args, preprocess=None, subset=False, subset_cnt=100, num_classes=None):
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
  
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "imagenet":
        dataset = datasets.ImageFolder('/imagenet/val', transform=preprocess)
        if subset or num_classes is not None:
            classwise_count = defaultdict(int)
            indices = []
            class_selected = set(range(1000-num_classes, 1000)) if num_classes is not None else None
            
            for i, label in enumerate(dataset.targets):
                if (class_selected is None or label in class_selected) and classwise_count[label] < subset_cnt:
                    indices.append(i)
                    classwise_count[label] += 1
            
            dataset = torch.utils.data.Subset(dataset, indices)
            
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError
    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess=None):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Places'), transform=preprocess)
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Texture', 'images'),
                                          transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
    return testloaderOut
