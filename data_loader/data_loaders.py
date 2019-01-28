from torchvision import datasets, transforms
import numpy as np
from base import BaseDataLoader
from .transforms import RandomElasticDeformation, RandomCrop, Normalization
from .datasets import AcdcDataset3D

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_root, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, self.dataset, batch_size, shuffle, validation_split, validation_random, num_workers)
        
class AcdcDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, mode, data_root, batch_size, crop_size, target_resolution, elastic_deform, random_crop, 
                 shuffle, validation_split, validation_random, num_workers, training=True):
        train_transform_list = []
        valid_transform_list = []
        if elastic_deform:
            train_transform_list.append(RandomElasticDeformation(num_controlpoints=4, std_deformation_sigma=10, proportion_to_augment=0.95))
        if random_crop:
            train_transform_list.append(RandomCrop(crop_size))
        train_transform_list.append(Normalization())
        valid_transform_list.append(Normalization())
        train_trsfm = transforms.Compose(train_transform_list)
        valid_trsfm = transforms.Compose(valid_transform_list)
        
        self.dataset = AcdcDataset3D(mode, data_root, target_resolution, transform=train_trsfm, training=True)
        self.valid_dataset = AcdcDataset3D(mode, data_root, target_resolution, transform=valid_trsfm, training=True)
        super(AcdcDataLoader, self).__init__(self.dataset, self.valid_dataset, batch_size, shuffle, validation_split, validation_random, num_workers)