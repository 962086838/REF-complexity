import numpy as np
import torch
import os
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

mean = {
    'MNIST': np.array([0.1307]),
    'CIFAR10': np.array([0.4914, 0.4822, 0.4465]),
    'CIFAR100': np.array([0.5071, 0.4867, 0.4408]),
    'TinyImageNet': np.array([0.4802, 0.4481, 0.3975]),
}
std = {
    'MNIST': np.array([0.3081]),
    'CIFAR10': np.array([0.2023, 0.1994, 0.2010]),
    'CIFAR100': np.array([0.2675, 0.2565, 0.2761]),
    'TinyImageNet': np.array([0.2302, 0.2265, 0.2262]),
}
train_transforms = {
    'MNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
    'CIFAR10': [transforms.RandomCrop(32, padding=4, padding_mode='edge'), transforms.RandomHorizontalFlip()],  # origin
    # 'CIFAR10': [],
    # 'CIFAR10': [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #             transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    #             transforms.RandomHorizontalFlip()],
    # 'CIFAR10': [transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    #             transforms.RandomCrop(32, padding=4, padding_mode='edge'), transforms.RandomHorizontalFlip()],
    # 'CIFAR10': [transforms.RandomRotation(10),
    #             transforms.RandomCrop(32, padding=4, padding_mode='edge'), transforms.RandomHorizontalFlip()],
    'CIFAR100': [transforms.RandomCrop(32, padding=4, padding_mode='edge'), transforms.RandomHorizontalFlip()],
    'TinyImageNet': [transforms.RandomCrop(64, padding=6, padding_mode='edge'), transforms.RandomHorizontalFlip()],
}
test_transforms = {
    'MNIST': [],
    'CIFAR10': [],
    'CIFAR100': [],
    'TinyImageNet': [],
}
input_dim = {
    'MNIST': np.array([1, 28, 28]),
    'CIFAR10': np.array([3, 32, 32]),
    'CIFAR100': np.array([3, 32, 32]),
    'TinyImageNet': np.array([3, 64, 64]),
}


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, **kwargs):
        path = 'train' if train else 'val'
        self.data = ImageFolder(os.path.join('tiny-imagenet-200', path), transform=transform)
        self.classes = self.data.classes
        self.transform = transform

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rand_labels):
        self.dataset = dataset
        self.classes = dataset.classes
        self.labels = rand_labels

    def __getitem__(self, item):
        return self.dataset[item][0], self.labels[item]

    def __len__(self):
        return len(self.dataset)


class NoiseAddDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_scale=0.0):
        self.dataset = dataset
        self.classes = dataset.classes
        self.noisetobeadd = noise_scale * torch.randn((len(self.dataset), self.dataset[0][0].shape[0], self.dataset[0][0].shape[1], self.dataset[0][0].shape[2]))

    def __getitem__(self, item):
        return self.dataset[item][0] + self.noisetobeadd[item], self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)


def get_dataset(dataset, dataset_name, datadir, augmentation=True):
    default_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset_name], std[dataset_name])
    ]
    train_transform = train_transforms[dataset_name] if augmentation else test_transforms[dataset_name]
    train_transform = transforms.Compose(train_transform + default_transform)
    test_transform = transforms.Compose(test_transforms[dataset_name] + default_transform)
    Dataset = globals()[dataset]
    train_dataset = Dataset(root=datadir, train=True, download=True, transform=train_transform)
    train_no_noise_dataset = Dataset(root=datadir, train=True, download=True, transform=test_transform)
    test_dataset = Dataset(root=datadir, train=False, download=True, transform=test_transform)

    # varying sample size
    # sample_size = 4000
    # _full_index = np.arange(len(train_dataset))
    # np.random.shuffle(_full_index)
    # _full_index = _full_index[0:sample_size]
    # train_dataset = torch.utils.data.Subset(train_dataset, _full_index)
    # train_no_noise_dataset = torch.utils.data.Subset(train_no_noise_dataset, _full_index)
    # if not hasattr(train_dataset, 'classes'):
    #     setattr(train_dataset, 'classes', 10)
    # if not hasattr(train_no_noise_dataset, 'classes'):
    #     setattr(train_no_noise_dataset, 'classes', 10)

    # add noise to input image
    # noise_scale = 0.4
    # train_dataset = NoiseAddDataset(train_dataset, noise_scale=noise_scale)
    # train_no_noise_dataset = NoiseAddDataset(train_no_noise_dataset, noise_scale=noise_scale)

    return train_dataset, train_no_noise_dataset, test_dataset


def load_data(dataset, datadir, batch_size, noisy_generator=None, augmentation=True, parallel=False, workers=4, num_classes=None):
    train_dataset, train_no_noise_dataset, test_dataset = get_dataset(dataset, dataset, datadir, augmentation=augmentation)
    if noisy_generator is not None:
        if num_classes is not None:
            rand_labels = torch.randint(num_classes, (len(train_dataset),), generator=noisy_generator,
                                        dtype=torch.long)
        else:
            rand_labels = torch.randint(len(train_dataset.classes), (len(train_dataset),), generator=noisy_generator,
                                        dtype=torch.long)
        print(len(rand_labels), rand_labels)
        train_dataset = NoisyDataset(train_dataset, rand_labels)
        train_no_noise_dataset = NoisyDataset(train_no_noise_dataset, rand_labels)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=torch.seed()) if parallel else None
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not parallel,
                             num_workers=workers, sampler=train_sampler, pin_memory=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if parallel else None
    train_no_noise_loader = DataLoader(train_no_noise_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=workers, sampler=test_sampler, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, sampler=test_sampler, pin_memory=True)
    return trainloader, train_no_noise_loader, testloader

