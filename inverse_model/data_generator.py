import torch
from torch.utils import data
import os
import json

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, partition):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.base_path = '../data/processed_poke'
        self.data_path = os.path.join(self.base_path, partition)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.tensor(torch.load(os.path.join(self.data_path, ID + '.pt'))).float()
        y = torch.tensor(self.labels[ID])[2:5].float()

        return X, y

if __name__=='__main__':
  base_path = '../data/processed_poke'
  labels = {}
  ids = {}
  for data_partition_name in ['train', 'val', 'test']:
    label_path = os.path.join(base_path, data_partition_name, 'labels.json')
    ids_path = os.path.join(base_path, data_partition_name, 'ids.json')

    with open(label_path) as json_file:
      labels[data_partition_name] = json.load(json_file)

    with open(ids_path) as json_file:
      ids[data_partition_name] = json.load(json_file)


  params = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 6}
  max_epochs = 100

  train_dataset = Dataset(ids['train'], labels['train'], partition='train')
  training_generator = data.DataLoader(train_dataset, **params)

  test_dataset = Dataset(ids['test'], labels['test'], partition='test')
  test_generator = data.DataLoader(test_dataset, **params)

  val_dataset = Dataset(ids['val'], labels['val'], partition='val')
  val_generator = data.DataLoader(val_dataset, **params)


  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  # cudnn.benchmark = True

  # Parameters

  
  # Loop over epochs
  for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
      # Transfer to GPU
      local_batch, local_labels = local_batch.to(device), local_labels.to(device)
  
      # Model computations
      [...]
  
      # Validation
      with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
          # Transfer to GPU
          local_batch, local_labels = local_batch.to(device), local_labels.to(device)
