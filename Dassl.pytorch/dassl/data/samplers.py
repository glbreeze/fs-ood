import copy
import numpy as np
from itertools import cycle, islice
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler


class RandomDomainSampler(Sampler):
    """Randomly samples N domains each with K images
    to form a minibatch of size N*K.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_domain (int): number of domains to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain has equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class SeqDomainSampler(Sampler):
    """Sequential domain sampler, which randomly samples K
    images from each domain to form a minibatch.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        # Make sure each domain has equal number of images
        n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            for domain in self.domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomClassSampler(Sampler):
    """Randomly samples N classes each with K instances to
    form a minibatch of size N*K.

    Modified from https://github.com/KaiyangZhou/deep-person-reid.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_ins (int): number of instances per class to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_ins):
        if batch_size < n_ins:
            raise ValueError(
                "batch_size={} must be no less "
                "than n_ins={}".format(batch_size, n_ins)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.n_ins = n_ins
        self.ncls_per_batch = self.batch_size // self.n_ins
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            self.index_dic[item.label].append(index)
        self.labels = list(self.index_dic.keys())
        assert len(self.labels) >= self.ncls_per_batch

        # estimate number of images in an epoch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.index_dic[label])
            if len(idxs) < self.n_ins:
                idxs = np.random.choice(idxs, size=self.n_ins, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.n_ins:
                    batch_idxs_dict[label].append(batch_idxs)
                    batch_idxs = []

        avai_labels = copy.deepcopy(self.labels)
        final_idxs = []

        while len(avai_labels) >= self.ncls_per_batch:
            selected_labels = random.sample(avai_labels, self.ncls_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        return iter(final_idxs)

    def __len__(self):
        return self.length



class ClassAwareSampler(Sampler):
    def __init__(self, datasource, batch_size, num_samples_per_class=2):
        """
        Args:
            datasource: List of datum objects. Each datum must have a `label` attribute.
            num_samples_per_class: Number of samples per class in each batch.
            num_classes_in_batch: Number of classes to sample per batch.
        """
        self.datasource = datasource
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes_in_batch = self.batch_size // num_samples_per_class
        self.class_to_indices = self._get_class_to_indices()
        self.all_classes = list(self.class_to_indices.keys())  # List of all class labels
        self.num_samples = len(self.datasource)

        # Sanity check: Ensure we do not sample more classes than available
        if self.num_classes_in_batch > len(self.all_classes):
            raise ValueError(f"num_classes_in_batch ({self.num_classes_in_batch}) exceeds the number of classes ({len(self.all_classes)}).")

    def _get_class_to_indices(self):
        """
        Groups indices of datasource by class labels.
        Returns: A dictionary mapping class labels to a list of indices.
        """
        class_to_indices = {}
        for idx, datum in enumerate(self.datasource):
            label = datum.label  # Get the class label of the datum
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices

    def __iter__(self):
        """
        Dynamically yield batches of indices with equal sampling within each class for one epoch.
        """
        # Shuffle classes at the start of the epoch
        class_order = cycle(np.random.permutation(self.all_classes))

        # Shuffle indices for each class and create iterators
        class_iters = {
            cls: cycle(np.random.permutation(self.class_to_indices[cls]))
            for cls in self.all_classes
        }

        total_batches = self.__len__()
        for i in range(total_batches):
            # Select a subset of classes for the current batch
            selected_classes = list(islice(class_order, self.num_classes_in_batch))
            current_batch = []

            for cls in selected_classes:
                current_batch.extend(
                    islice(class_iters[cls], self.num_samples_per_class)
                )

            yield current_batch

    def __len__(self):
        """
        Returns the total number of batches.
        """
        batch_size = self.num_classes_in_batch * self.num_samples_per_class
        return (self.num_samples + batch_size - 1) // batch_size  # Round up to account for remainder



def build_sampler(
    sampler_type,
    cfg=None,
    data_source=None,
    batch_size=32,
    n_domain=0,
    n_ins=16
):
    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)

    elif sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)

    elif sampler_type == "RandomDomainSampler":
        return RandomDomainSampler(data_source, batch_size, n_domain)

    elif sampler_type == "SeqDomainSampler":
        return SeqDomainSampler(data_source, batch_size)

    elif sampler_type == "RandomClassSampler":
        return RandomClassSampler(data_source, batch_size, n_ins)
    
    elif sampler_type == 'ClassAwareSampler':
        return ClassAwareSampler(data_source, batch_size, num_samples_per_class=2)

    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))
