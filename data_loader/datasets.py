import os
import glob
import scipy
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import nibabel as nib

class AcdcDataset3D(data.Dataset):
    training_dir = 'training'
    testing_dir = 'testing'
    
    def __init__(self, mode, data_root, target_resolution, transform, training=True):
        super(AcdcDataset3D, self).__init__()
        self.mode = mode
        self.data_root = data_root
        self.training = training
        self.target_resolution = target_resolution
        self.transform = transform
        if training:
            self.data_path = os.path.join(data_root, self.training_dir)
        else:
            self.data_path = os.path.join(data_root, self.testing_dir)
            
        # get the image / label(if training == True) file paths
        image_paths, label_paths = [], []
        patient_paths = [os.path.join(self.data_path, name)
                         for name in sorted(os.listdir(self.data_path)) if os.path.isdir(os.path.join(self.data_path, name))]
        for patient_path in patient_paths:
            image_paths.extend(glob.glob(os.path.join(patient_path, '*frame??.nii.gz')))
            if self.training:
                label_paths.extend(glob.glob(os.path.join(patient_path, '*frame??_gt.nii.gz')))
        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        
    def __getitem__(self, index):
        metadata = nib.load(self.image_paths[index])
        raw_resolution = metadata.header.structarr['pixdim'][1:4]
        image = metadata.get_data()
        label = []
        if self.mode == '2D':
            image = self.resample(image, raw_resolution, self.target_resolution, interp_order=3)
            if self.training:
                label = nib.load(self.label_paths[index]).get_data()
                label = self.resample(label, raw_resolution, self.target_resolution, interp_order=0)
        elif self.mode == '3D':
            image = self.resample(image, raw_resolution, self.target_resolution, interp_order=3)
            if self.training:
                label = nib.load(self.label_paths[index]).get_data()
                label = self.resample(label, raw_resolution, self.target_resolution, interp_order=0)
            data = {'image': image, 'label': label}
            data = self.transform(data)
            image, label = data['image'], data['label']
            image = torch.from_numpy(image.transpose((2, 0, 1))[np.newaxis, :]).float() # (C, D, H, W) 
            label = torch.from_numpy(label.transpose((2, 0, 1))[np.newaxis, :]).float() # (C, D, H, W)
        return image, label
    
    def __len__(self):
        return len(self.image_paths)
    
    def resample(self, volume, in_resolution, out_resolution, interp_order):
        if self.mode == '2D':
            zoom = [in_resolution[0] / out_resolution[0], in_resolution[1] / out_resolution[1], 1]
        elif self.mode == '3D':
            zoom = [in_resolution[0] / out_resolution[0], in_resolution[1] / out_resolution[1], in_resolution[2] / out_resolution[2]]
        return scipy.ndimage.zoom(volume, zoom, order=interp_order)