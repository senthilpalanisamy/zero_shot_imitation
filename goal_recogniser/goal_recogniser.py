import os
import time
import json
import sys
from datetime import date, datetime

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torchsummary import summary 
from torch.autograd import Variable


from data_generator import Dataset
from utils import *
import random


class Net(nn.Module):
  def __init__(self):
    super().__init__()

    self.__to_linear = 3600
    self.conv1 = nn.Conv2d(3, 64, 11, stride=4)
    self.conv2 = nn.Conv2d(64, 192, 5, stride=1, padding=2)
    self.conv3 = nn.Conv2d(192, 384, 3, stride=1, padding=2)
    self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=2)
    # self.fc1 = nn.Linear(self._to_linear, 100)
    self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=2)
    #self.conv6 = nn.Conv2d(256, 200, 3)
    self.conv6 = nn.Conv2d(256, 200, 3, stride=1, padding=2)
    self.bn0 = nn.BatchNorm1d(self.__to_linear)
    self.fc1 = nn.Linear(self.__to_linear, 1000)
    self.bn1 = nn.BatchNorm1d(1000)
    self.fc2 = nn.Linear(1000, 20) 
    self.bn2 = nn.BatchNorm1d(20)
    #self.fc3 = nn.Linear(200, 20)
    #self.fc4 = nn.Linear(20, 5)
    self.fc5 = nn.Linear(20, 1)

    self.sigmoid = nn.Sigmoid()
    self.relu = nn.LeakyReLU()



  def forward(self, image1, image2):
    latent_features = []
    image_pair=[image1, image2]
    for index in range(2):
      image = image_pair[index]
      x = F.max_pool2d(self.relu(self.conv1(image)), (3,3), stride=2)
      x = F.max_pool2d(self.relu(self.conv2(x)), (3, 3), stride=2)
      x = self.relu(self.conv3(x))
      x = self.relu(self.conv4(x))
      x = F.max_pool2d(self.relu(self.conv5(x)), (3, 3), stride=2)
      x = F.max_pool2d(self.relu(self.conv6(x)), (3, 3))
      latent_features.append(x)
    
    N = latent_features[0].shape[0]
    flatten_input1 = latent_features[0].reshape(N, -1)
    flatten_input2 = latent_features[1].reshape(N, -1)
    latent_2images = torch.cat((flatten_input1, flatten_input2), 1) 
    x = self.bn1(self.fc1(self.bn0(latent_2images.view(-1, self.__to_linear))))
    x = self.bn2(self.fc2(x))
    #x = self.fc3(x)
    #x = self.fc4(x)
    op = self.fc5(x)
    op = self.sigmoid(op)
    return op

  def calculate_loss(self, outputs, targets):

    criterion = nn.BCELoss()
    entropy_loss = criterion(outputs.reshape(-1), targets) 
    return entropy_loss

  def calculate_accuracy(self, data, labels):

     img1 = data[:,0,:,:,:]
     img2 = data[:,1,:,:,:]
     outputs = self.forward(img1, img2)
     self.zero_grad()
     loss = self.calculate_loss(outputs, labels)
     accuracies = []
     classifier_accuracies = {}

     predicted_labels = [int(torch.round(op)) for op in outputs]
     classification_results = []
     for gt, predicted in zip(labels, predicted_labels):
        result = 1 if gt==predicted else 0
        classification_results.append(result)

     del loss
     del outputs
     return sum(classification_results) / len(classification_results)



  
  def transfer_weigths_from_alexnet(self):

    alexnet = models.alexnet()
    transfer_model_state = alexnet.state_dict()
    similar_layer_names = [['conv1.weight', 'features.0.weight'],
                           ['conv1.bias', 'features.0.bias'],     
                           ['conv2.weight', 'features.3.weight'],
                           ['conv2.bias', 'features.3.bias'],     
                           ['conv3.weight', 'features.6.weight'],
                           ['conv3.bias', 'features.6.bias'],     
                           ['conv4.weight', 'features.8.weight'],
                           ['conv4.bias', 'features.8.bias'],     
                           ['conv5.weight', 'features.10.weight'],
                           ['conv5.bias', 'features.10.bias']]     
    model_state = self.state_dict()
    for my_layer_name, transfer_layer_name in similar_layer_names:
      model_state[my_layer_name] = transfer_model_state[transfer_layer_name]
      print(my_layer_name, model_state[my_layer_name].shape == transfer_model_state[transfer_layer_name].shape)
    self.load_state_dict(model_state)

class networkTrainer:
  def __init__(self, partitioned_datasets, EPOCHS=100, BATCH_SIZE=100, experiment_details={}):
    self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.__data_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = {'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 6}

    self.train_data_generator = data.DataLoader(partitioned_datasets['train'], **params)
    self.val_data_generator = data.DataLoader(partitioned_datasets['val'], **params)
    self.test_data_generator = data.DataLoader(partitioned_datasets['test'], **params)


    self.__net = Net()
    self.__net.transfer_weigths_from_alexnet()
    self.__net = self.__net.to(self.__device)
    for params in self.__net.named_parameters():
       params[1].retain_grad()

    self.sample_data = partitioned_datasets['train'][0]
    tensor_shape = self.sample_data[0].shape
    self.__IMG_HEIGHT = tensor_shape[2]
    self.__IMG_WIDTH = tensor_shape[3]
    self.__NO_OF_CHANNELS = tensor_shape[1]
    self.EPOCHS = EPOCHS
    self.BATCH_SIZE = BATCH_SIZE
    self.lr = experiment_details['lr']
    self.exp_name = experiment_details['exp_name']
    path_to_write = os.path.join('run', experiment_name)
    self.__writer = SummaryWriter(path_to_write)
    self.__accuracies = {}
    

  def __write_values_to_graph(self, dataset_name, accuracy, iter_count):

    self.__writer.add_scalar('accuracy/'+dataset_name+'goal_recog', accuracy, iter_count)


  def __plot_accuracy_graphs(self, dataset_name, iter_count): 

    if(dataset_name == 'val'):
        data_generator = self.val_data_generator
    elif(dataset_name == 'test'):
      data_generator = self.test_data_generator
    elif(dataset_name == 'train'):
      data_generator = self.train_data_generator


    full_accuracy = 0;
    count = 0
    for batch_x, labels in data_generator:
      batch_x = batch_x.to(self.__device)
      labels = labels.to(self.__device)

      batch_accuracy = self.__net.calculate_accuracy(batch_x, labels)
      full_accuracy += batch_accuracy
      torch.cuda.empty_cache() 
      count += 1

    full_accuracy = full_accuracy / count

    self.__write_values_to_graph(dataset_name, full_accuracy, iter_count)
    self.__accuracies[dataset_name] = full_accuracy
     




  def train_network(self):

    optimizer = optim.Adam(self.__net.parameters(), lr=self.lr)
    X, y  = self.sample_data
    y = y.to(self.__device).float().unsqueeze(0)
    img1 = X[0,:,:,:].to(self.__device)
    img2 = X[1,:,:,:].to(self.__device)
  
    self.__writer.add_image('goal recogniser_image1', img1 * 255.0)
    self.__writer.add_image('goal_recogniser_image2', img2 * 255.0)
    self.__writer.add_graph(self.__net, (img1.unsqueeze(0), img2.unsqueeze(0)))
    index =0
    self.__net.train()

    for epoch in range(self.EPOCHS):
      for batch_x, batch_y in tqdm(self.train_data_generator):
        index += 1

        # self.__net.zero_grad()
        # optimizer.zero_grad()
        batch_x = batch_x.to(self.__device)
        batch_y = batch_y.to(self.__device)


        batch_x_img1 = batch_x[:,0,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)
        batch_x_img2 = batch_x[:,1,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)

        outputs = self.__net(batch_x_img1, batch_x_img2)
        loss = self.__net.calculate_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print(loss)

        #plot_grad_flow_2(self.__net.named_parameters())


        batch_accuracy = self.__net.calculate_accuracy(batch_x, batch_y)
        self.__accuracies['train'] = batch_accuracy
        self.__write_values_to_graph(dataset_name = 'train', accuracy = batch_accuracy,
                                     iter_count = index)
        del loss
        del outputs

      print('validation')
      self.__plot_accuracy_graphs(dataset_name='val', iter_count=index)
        #self.__plot_accuracy_graphs(dataset_name='train', iter_count=epoch * self.BATCH_SIZE + i)
      print('end validation')


    self.__plot_accuracy_graphs(dataset_name='test', iter_count = index)
    torch.save(self.__net.state_dict(), os.path.join('./models', self.exp_name + '.pt'))
    results = [self.__accuracies['test'],
               self.__accuracies['train'],
               self.__accuracies['val']]
    return results

 

    # self.__writer.close()


if __name__=='__main__':

  base_path = '/senthil/data/goal_recog_data_1'
  labels = {}
  ids = {}
  experiment_name = sys.argv[1]
  no_of_epochs = int(sys.argv[2])
  seed_no = int(sys.argv[3])
  experiment_name = experiment_name + '_epoch_' + str(no_of_epochs) + '_'+\
                     datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
  learning_rate = float(sys.argv[4])
  if seed_no == -1:
    seed_no = random.randint(0, 10000) 
  torch.cuda.manual_seed(seed_no)
  partitioned_datasets = {}
  for data_partition_name in ['train', 'val', 'test']:
    label_path = os.path.join(base_path, data_partition_name, 'labels.json')
    ids_path = os.path.join(base_path, data_partition_name, 'ids.json')

    with open(label_path) as json_file:
      labels[data_partition_name] = json.load(json_file)

    with open(ids_path) as json_file:
      ids[data_partition_name] = json.load(json_file)
  exp_details = [date.today().strftime("%d/%m/%Y"), datetime.now().strftime("H:%M:%S"),
                 experiment_name, no_of_epochs, seed_no]

  partitioned_datasets['train']  = Dataset(ids['train'], labels['train'], partition='train', base_path=base_path)
  partitioned_datasets['test']  = Dataset(ids['test'], labels['test'], partition='test', base_path=base_path)
  partitioned_datasets['val'] = Dataset(ids['val'], labels['val'], partition='val', base_path=base_path)
  experiment_details = {}
  experiment_details['exp_name'] = experiment_name
  experiment_details['lr'] = learning_rate

  exp_details = [date.today().strftime("%d/%m/%Y"), datetime.now().strftime("%H:%M:%S"),
                 experiment_name, no_of_epochs, seed_no, learning_rate]

  dl_trainer = networkTrainer(partitioned_datasets, EPOCHS=no_of_epochs, experiment_details=experiment_details)
  results = dl_trainer.train_network()
  results = list(map(float, results))
  row_to_write = exp_details + results
  write_to_gsheet(row_to_write)
