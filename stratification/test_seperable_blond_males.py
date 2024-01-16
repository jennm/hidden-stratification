import argparse
import gc
import json
import logging
import numpy as np
import os
import pandas as pd
import sklearn.metrics
import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from functools import partial
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from stratification.harness import GEORGEHarness
from stratification.intersectional_algorithm import ALGOClassification
from stratification.utils.parse_args import get_config


class CelebADataset(Dataset):
    _normalization_stats = {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}

    def __init__(self, root, split, transform=None, test_celeba=False):
        # Initialize your dataset
        # Load your dataset from the CSV file or any other source
        self.root = root
        self.split = split
        self.test_celeba = test_celeba
        logging.info(f'Loading {self.split} split of CelebA')
        if self.test_celeba:
            self.X, self.Y_Array, self.class_labels = self._load_samples()
        else:
            self.X, self.Y_Array
        self.transform = get_transform_celebA()
        
    def get_num_classes(self):
        return torch.max(self.Y_Array).item() + 1

    def _load_samples(self):
        self.target_name = 'Blond_Hair'
        
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

        # Read in train/val/test splits
        split_df = pd.read_csv(os.path.join(self.root, 'celebA', 'list_eval_partition.csv'),
                               delim_whitespace=True)
        split_array = split_df['partition'].values
        split_dict = {'train': 0, 'val': 1, 'test': 2}
        split_indices = split_array == split_dict[self.split]
        self.len_dataset = len(split_indices) 

        X = filename_array[split_indices]
        y_array = torch.tensor(y_array[split_indices], dtype=torch.float)
        self.len_dataset = y_array.shape[0] 

        if self.test_celeba:
            male_idx = attr_idx('Male')
            class_labels = list()
            for i in range(attrs_df.shape[0]):
                if attrs_df[i][target_idx] == 1 and attrs_df[i][male_idx] == 1:
                    class_labels.append(1)
                else:
                    class_labels.append(0)
            class_labels = torch.tensor(class_labels, dtype=torch.float)
            return X, y_array, class_labels

        return X, y_array
        

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
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
        x = x.to(torch.float)
        if self.test_celeba:
            return {'image': x, 'label': self.Y_Array[idx], 'class_labels': self.class_labels[idx]}
        return {'image': x, 'label': self.Y_Array[idx]}

def get_transform_celebA():
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    transform = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize(**CelebADataset._normalization_stats),
    ])
    return transform

tail_cache = dict()
def hook_fn(module, input, output):
    device = output.get_device()
    if device in tail_cache:
        tail_cache[device].append(input[0].clone().detach())
    else:
        tail_cache[device] = [input[0].clone().detach()]
    

def get_hooks(model):
    hooks = []
    num_layers = sum(1 for _ in model.modules())
    print('model num layers', num_layers)
    for i, module in enumerate(model.modules()):
        if i >= num_layers - 5:
            print(f"{i}: {num_layers - i}")
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    print('hooks_length: ', len(hooks))

def setup(config_fp):
    with open(config_fp, 'r') as f:
        config = json.dumps(json.load(f))
    config = get_config([config])
    harness = GEORGEHarness(config, use_cuda=torch.cuda.is_available(), log_format='simple')
    dataset = CelebADataset(root='data', split='train', test_celeba=True)
    num_classes = int(dataset.get_num_classes())
    print('num classes:', num_classes)
    with torch.no_grad():
        model = harness.get_nn_model(config, num_classes=num_classes, mode='erm')
    config = config['classification_config']
    shared_dl_args = {'batch_size': config['batch_size'], 'num_workers': config['workers']}
    return model, dataset, shared_dl_args, num_classes

def load_pretrained_model(model_path, model):
    state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    get_hooks(model)
    return model

def collate_func(batch, pretrained_model, criterion, layer_num, class_label=False):
    global tail_cache
    inputs = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    with torch.no_grad():
        # Forward pass through the 5th to last layer
        outputs = pretrained_model(inputs)
        device = outputs.get_device()
        pred = torch.argmax(outputs, 1)
        # loss per batch for now
        pred = pred.to(torch.float)
        labels = labels.to(device)
        loss = criterion(pred, labels)

    embeddings = torch.cat(tuple([tail_cache[i][layer_num + 1].to(device) for i in list(tail_cache.keys())]), dim=0)

    if class_label:
        class_labels = torch.stack([item['class_labels'] for item in batch])
        data = {'embeddings': embeddings, 'loss': loss, 'predicted_label': pred, 'actual_label': labels, 'class_label': class_labels}
    else:
        data = {'embeddings': embeddings, 'loss': loss, 'predicted_label': pred, 'actual_label': labels}
    tail_cache = dict()
    gc.collect()
    torch.cuda.empty_cache()

    return data

def create_dataloader(saved_model_path, model, datasets, shared_dl_args, layer_num=1, criterion=nn.CrossEntropyLoss(reduction='none')):
    pretrained_model = load_pretrained_model(saved_model_path, model)

    if type(datasets) is dict:
        dataloaders = dict()
        for dataset_type in datasets:
                collate_fn = partial(collate_func, pretrained_model=pretrained_model, criterion=criterion, layer_num=layer_num)
                dataloaders[dataset_type] = DataLoader(datasets[dataset_type], **shared_dl_args, collate_fn=collate_fn)
    else:
        # Create a DataLoader
        dataloaders = DataLoader(datasets, **shared_dl_args, collate_fn=collate_fn)

    return dataloaders, pretrained_model

# Define a simple logistic regression model for image classification
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return nn.functional.softmax(self.linear(x), dim=1)


def find_group(dataloaders, num_classes=2, loss_threshold=.5, acc_threshold=.9, actual_label=1):
    input_size = 25088  # MNIST images are 28x28 pixels
    num_classes = num_classes
    log_model = LogisticRegressionModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(log_model.parameters(), lr=0.01)
    all_class_labels = list()

    for batch in dataloaders['train']:
        device = torch.cuda.current_device()
        embeddings = batch['embeddings']
        losses = batch['loss']
        predicted_labels = batch['predicted_label']
        actual_labels = batch['actual_label']

        class_labels = list()
        if losses.item() >= loss_threshold:
            for i in range(embeddings.shape[0]):
                if predicted_labels[i] != actual_labels[i] and actual_labels[i] == actual_label:
                    class_labels.append(1)
                else:
                    class_labels.append(0)
            class_label = 1
        else:
            class_labels = [0 for i in range(embeddings.shape[0])]
                
        # for i in range(len(losses)):
        #     if losses[i] >= loss_threshold and predicted_labels[i] != actual_label[i] and actual_labels[i] == actual_label:
        #         class_labels.append(1)
        #     else:
        #         class_labels.append(0)

    # for batch in dataloaders['train']:
        # device = torch.cuda.current_device()
        log_model.to(device)
        embeddings = batch['embeddings']
        # loss = batch['loss']
        # class_labels = batch['class_label']

        # Flatten the images
        embeddings = embeddings.view(embeddings.size(0), -1)
        class_labels = torch.tensor(class_labels)
        class_labels = class_labels.to(device)
        # all_class_labels.extend(class_labels)

        # Forward pass
        outputs = log_model(embeddings)

        # Changing torch type
        class_labels = class_labels.to(torch.long)#float)

        # Calculate loss
        loss = criterion(outputs, class_labels)
        loss_mean = torch.mean(loss)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

    log_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloaders['val']:
            embeddings = batch['embeddings']
            loss = batch['loss']
            predicted_labels = batch['predicted_label']
            actual_labels = batch['actual_label']

            class_labels = list()
            for i in range(len(losses)):
                if losses[i] >= loss_threshold and predicted_labels[i] != actual_label[i] and actual_labels[i] == actual_label:
                    class_labels.append(1)
                else:
                    class_labels.append(0)

            embeddings = embeddings.view(embeddings.size(0), -1)
            outputs = log_model(embeddings)

            _, predicted = torch.max(outputs, 1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
    return log_model, accuracy


def train_test_classifier(config_fp, pretrained_fp, robust=True):
    _model, dataset, shared_dl_args, num_classes = setup(config_fp)
    


    # Set random seed for reproducibility
    torch.manual_seed(42)
    mp.set_start_method('spawn')

    # Define transformations and download MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CelebADataset(root='data', split='train', test_celeba=True)
    val_dataset = CelebADataset(root='data', split='val', test_celeba=True)
    test_dataset = CelebADataset(root='data', split='test', test_celeba=True)
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    layer_num = 3

    # find_group=True
    classify = ALGOClassification(config_fp)

    while True:
        groups = find_group(train_dataset)
        classify.train(model, train_dataloader, val_dataloader, groups, robust)

    # Initialize the model, loss function, and optimizer
    dataloaders, old_model = create_dataloader(pretrained_fp, _model, datasets, shared_dl_args, layer_num)
    log_models, acc = find_group(dataloaders, 2)
        
        #retrain
    return log_models, acc
    # return new_model

    for i in range(5):
        print(f'Layer {i}')
        dataloaders, old_model = create_dataloader(pretrained_fp, _model, datasets, shared_dl_args, i)

        # Initialize the model, loss function, and optimizer
        input_size = 25088  # MNIST images are 28x28 pixels
        num_classes = 2  # use a variable
        log_model = LogisticRegressionModel(input_size, num_classes)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(log_model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 5

        for epoch in range(num_epochs):
            log_model.train()
            for batch in dataloaders['train']:
                device = torch.cuda.current_device()
                log_model.to(device)
                embeddings = batch['embeddings']
                loss = batch['loss']
                class_labels = batch['class_label']

                # Flatten the images
                embeddings = embeddings.view(embeddings.size(0), -1)
                class_labels = class_labels.to(device)

                # Forward pass
                outputs = log_model(embeddings)

                # Changing torch type
                class_labels = class_labels.to(torch.long)#float)

                # Calculate loss
                loss = criterion(outputs, class_labels)
                loss_mean = torch.mean(loss)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

            # Print training loss after each epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_mean.item():.4f}')

        # Evaluation on the test set
        log_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in dataloaders['test']:
                embeddings = batch['embeddings']
                loss = batch['loss']
                class_labels = batch['class_label']
                class_labels = class_labels.to(device)
                class_labels = class_labels.to(torch.long)

                embeddings = embeddings.view(embeddings.size(0), -1)
                outputs = log_model(embeddings)

                _, predicted = torch.max(outputs, 1)
                total += class_labels.size(0)
                correct += (predicted == class_labels).sum().item()

            accuracy = correct / total
            print(f'Test Accuracy: {accuracy:.4f}')
        
            torch.cuda.empty_cache()

    return log_model, accuracy







def list_of_columns(arg):
    arg.split(' ')

def find_poor_performing_group_indices(groups, data_info_path):
    indices = list()
    data_info_df = pd.read_csv(data_info_path)
    cols = list(data_info_df.columns)[0].split(' ')
    relevant_cols = list()
    i = 0
    while len(relevant_cols) < len(groups) and i < len(cols):
        if cols[i] in groups:
            relevant_cols.append(i)
    
    for index, row in data_info_df.iterrows():
        all_relevant = True
        for j in range(relevant_cols):
            if row.iloc[j] != 1:
                all_relevant = False
                break
        if all_relevant:
            indices.append(index)

    return indices




def main():
    parser = argparse.ArgumentParser(description='tests that poor performinc class is separable by embeddings')
    parser.add_argument('--config_fp', type=str, help='relative file path to config json.')
    parser.add_argument('--pretrained_fp', type=str, help='relative file path to model checkpoint.')
    parser.add_argument('--data_info', type=str, help='relative file path to the data info')
    parser.add_argument('--poor_performing_group', type=list_of_columns, help='names in data_info of the poor performing group')
    args = parser.parse_args()

    log_model, acc = train_test_classifier(args.config_fp, args.pretrained_fp)
    print('Accuracy:', acc)
    # _model, dataset, shared_dl_args, num_classes = setup(args.config_fp)
    # create_dataloader()
    # dataloader, model = create_dataloader(args.pretrained_fp, _model, dataset, shared_dl_args, 1)
    

    



if __name__ == '__main__':
    main()
