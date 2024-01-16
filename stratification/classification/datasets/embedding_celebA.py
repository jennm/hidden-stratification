import os
import torch
import pandas as pd
from PIL import Image
import logging
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from ...find_celeba_embeddings_class import FindEmbeddings


class CelebAEmbeddingDataset(Dataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    Adapted from https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
    """
    superclass_names = ['No Blond Hair', 'Blond Hair']
    true_subclass_names = [
        'Blond_Hair = 0, Male = 0', 'Blond_Hair = 0, Male = 1', 'Blond_Hair = 1, Male = 0',
        'Blond_Hair = 1, Male = 1'
    ]
    _channels = 3
    _resolution = 224
    _normalization_stats = {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}

    def __init__(self, root, split, model):
        logging.info(f'Loading {self.split} split of {self.name}')
        self.X, self.Y_Array = self._load_samples()
        self.transform = get_transform_celebA()
        self.model = model
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.find_embeddings = FindEmbeddings(self.model, criterion)

    def update_model(self, model):
        self.model = model
        self.find_embeddings.update_model(self.model)

    def _load_samples(self):
        self.target_name = 'Blond_Hair'
        # self.confounder_names = ['Male']

        attrs_df = pd.read_csv(os.path.join(self.root, 'celebA', 'list_attr_celeba.csv'),
                               delim_whitespace=True)

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root, 'celebA', 'img_align_celeba')
        filename_array = attrs_df['image_id'].values
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')
        attr_names = attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        attrs_df = attrs_df.values
        attrs_df[attrs_df == -1] = 0

        def attr_idx(attr_name):
            return attr_names.get_loc(attr_name)

        # Get the y values
        target_idx = attr_idx(self.target_name)
        y_array = attrs_df[:, target_idx]
        self._num_supclasses = np.amax(y_array).item() + 1

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        # self.confounder_idx = [attr_idx(a) for a in self.confounder_names]
        # confounders = attrs_df[:, self.confounder_idx]
        # confounder_array = confounders @ np.power(
        #     2, np.arange(len(self.confounder_idx)))

        # Map to groups
        # self._num_subclasses = self._num_supclasses * \
        #     pow(2, len(self.confounder_idx))
        # group_array = (y_array * (self._num_subclasses / 2) +
        #                confounder_array).astype('int')

        # Read in train/val/test splits
        split_df = pd.read_csv(os.path.join(self.root, 'celebA', 'list_eval_partition.csv'),
                               delim_whitespace=True)
        split_array = split_df['partition'].values
        split_dict = {'train': 0, 'val': 1, 'test': 2}
        split_indices = split_array == split_dict[self.split]

        X = filename_array[split_indices]
        # Y_dict = {
        #     'superclass': torch.tensor(y_array[split_indices]),
        #     'true_subclass': torch.tensor(group_array[split_indices])
        # }
        # return X, Y_dict
        return X, y_array

    def __getitem__(self, idx, embedding_num=1):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (x_dict, y_dict) where x_dict is a dictionary mapping all
                possible inputs and y_dict is a dictionary for all possible labels.
        """
        img_filename = os.path.join(self.data_dir, self.X[idx])
        image = Image.open(img_filename)
        if self.transform is not None:
            image = self.transform(image)
        x = image

        embeddings = self.find_embeddings.get_example_embeddings(x, self.Y_Array[idx], embedding_num)

        
        return x, embeddings


def get_transform_celebA():
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    transform = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize(**CelebAEmbeddingDataset._normalization_stats),
    ])
    return transform
