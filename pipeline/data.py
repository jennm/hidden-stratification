import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class CelebA(Dataset):
    _norm_stats = {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
    }


    def __init__(self, root, split, target1='Blond_Hair', target2='Male'):
        self.root = root
        self.split = split
        self.target1 = target1
        self.target2 = target2
        self.X, self.Y, self.class_labels = self._load_samples(self.target_name)
        self.transform = get_transform()


    def get_num_classes(self):
        return torch.max(self.Y).item() + 1
    

    def _load_samples(self):
        attrs_df = pd.read_csv(os.path.join(self.root, 'celebA', 'list_attr_celeba.csv'),
                               delim_whitespace=True)
        self.data_dir = os.path.join(self.root, 'celebA', 'img_align_celeba')

        filenames = attrs_df['image_id'].values  # all img filenames
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')
        attrs_names = attrs_df.columns.copy()

        attrs_df = attrs_df.values
        attrs_df[attrs_df == -1] = 0

        def attr_idx(attr_name):
            return attrs_names.get_loc(attr_name)
    
        target1_idx = attr_idx(self.target1)
        Y = attrs_df[:, target1_idx]  # target values for all images
        self._num_supclasses = np.amax(Y).item() + 1

        split_df = pd.read_csv(os.path.join(self.root, 'celebA', 'list_eval_partition.csv'),
                               delim_whitespace=True)
        
        split_array = split_df['partition'].values  # partition id for each img
        split_dict = {'train':0, 'val':1, 'test':2}
        split_indices = split_array == split_dict[self.split]  # keeps only imgs in desired partition

        X = filenames[split_indices]  # gets files for imgs in desired partition
        Y = torch.tensor(Y[split_indices], dtype=torch.float)  # picks target values for imgs in partition

        self.len_dataset = Y.shape[0]
        target2_idx = attr_idx(self.target2)
        class_labels = (attrs_df[:, target1_idx] == 1) & (attrs_df[:, target2_idx] == 1)
        class_labels = torch.tensor(class_labels, dtype=torch.float)

        return X, Y, class_labels


    def __len__(self):
        return self.len_dataset


    def __getitem__(self, idx):
        img_fn = os.path.join(self.data_dir, self.X[idx])
        image = Image.open(img_fn)
        if self.transform is not None:
            image = self.transform(image)
        x = image
        x = x.to(torch.float)
        return {'image': x, 'label': self.Y[idx], 'class_labels': self.class_labels[idx]}


def get_transform():
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)
    transform = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize(**CelebA._norm_stats),
    ])

    return transform


def create_dataloader():
    pass