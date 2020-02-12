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

from data_generator import Dataset
from config import *
from utils import *
import random


class Net(nn.Module):
  def __init__(self, lamda=0.5):
    super().__init__()

    self.__to_linear = 3600
    NO_OF_ACTIONS = 3
    self.conv1 = nn.Conv2d(3, 64, 11, stride=4)
    self.conv2 = nn.Conv2d(64, 192, 5, stride=1, padding=2)
    self.conv3 = nn.Conv2d(192, 384, 3, stride=1, padding=2)
    self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=2)
    # self.fc1 = nn.Linear(self._to_linear, 100)
    self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=2)
    #self.conv6 = nn.Conv2d(256, 200, 3)
    self.conv6 = nn.Conv2d(256, 200, 3, stride=1, padding=2)
    self.pre_xy_classifier = nn.Linear(self.__to_linear, 1000)
    #self.fc1 = nn.Linear(19009600, 200) 
    # Action_classifier
    self.xy_classifier = nn.Linear(1000, XYBIN_COUNT) 
    self.pre_angle_classifier = nn.Linear(self.__to_linear + XYBIN_COUNT, 200)
    self.angle_classifier = nn.Linear(200, ANGLE_BIN_COUNT)
    self.pre_length_classifier = nn.Linear(self.__to_linear + XYBIN_COUNT + ANGLE_BIN_COUNT,
                                    200)
    self.length_classifier = nn.Linear(200, LEN_ACTIONBIN_COUNT)
    self.forward_fc1 = nn.Linear(self.__to_linear//2 + NO_OF_ACTIONS, self.__to_linear//2)
    self.forward_fc2 = nn.Linear(self.__to_linear//2, self.__to_linear//2)
    self.forward_fc3 = nn.Linear(self.__to_linear//2, self.__to_linear//2)
    self._IMAGE_WIDTH = 224
    self._IMAGE_COL = 224
    self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.lamda = 0.5




  def forward(self, image1, image2, actions):
    latent_features = []
    image_pair=[image1, image2]
    for index in range(2):
      image = image_pair[index]
      x = F.max_pool2d(F.relu(self.conv1(image)), (3,3), stride=2)
      x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3), stride=2)
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.max_pool2d(F.relu(self.conv5(x)), (3, 3), stride=2)
      x = F.max_pool2d(F.relu(self.conv6(x)), (3, 3))
      latent_features.append(x)

      # if self._to_linear is None:
      #    self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
      #    self._to_linear = self._to_linear * 2
      #    print('to_linear', self._to_linear)
      #    self.fc1 = nn.Linear(self._to_linear, 200)
    
    N = latent_features[0].shape[0]
    flatten_input1 = latent_features[0].reshape(N, -1)
    flatten_input2 = latent_features[1].reshape(N, -1)
    latent_2images = torch.cat((flatten_input1, flatten_input2), 1) 
    x = self.pre_xy_classifier(latent_2images.view(-1, self.__to_linear))
    x = self.xy_classifier(x)
    # op1 = F.softmax(x, dim=1)
    op1 = torch.argmax(x, axis=1).to(self.__device)
    op1 = (torch.zeros(len(op1), XYBIN_COUNT)).to(self.__device).scatter_(1, op1.unsqueeze(1), 1.)
    #op1 = torch.FloatTensor(N, XYBIN_COUNT).to(self.__device).scatter_(1,op1, 1)
    angle_concat = torch.cat((latent_2images, op1), 1) 
    x = self.pre_angle_classifier(angle_concat)
    x = self.angle_classifier(x)
    op2 = torch.argmax(x, axis=1)
    op2 = (torch.zeros(len(op2), ANGLE_BIN_COUNT)).to(self.__device).scatter_(1, op2.unsqueeze(1), 1.)
    # op2 = torch.FloatTensor(N, ANGLE_BIN_COUNT).to(self.__device).scatter_(1,op2, 1)
    x = torch.cat((angle_concat, op2), 1)
    x = self.pre_length_classifier(x)
    x = self.length_classifier(x)

    op3 = torch.argmax(x, axis=1)
    op3 = (torch.zeros(len(op3), LEN_ACTIONBIN_COUNT)).to(self.__device).scatter_(1, op3.unsqueeze(1), 1.)
    #op3 = (torch.zeros(len(op3), LEN_ACTIONBIN_COUNT).scatter_(1, op3.unsqueeze(1), 1.)).to(self.__device)
    #op3 = torch.FloatTensor(N, LEN_ACTIONBIN_COUNT).to(self.__device).scatter_(1,op3, 1)

    forward_input = torch.cat((flatten_input1,actions), 1)
    x = self.forward_fc1(forward_input)
    x = self.forward_fc2(x)
    forward_output = self.forward_fc3(x)
    #final_op = torch.cat((op1.unsqueeze(0), op2.unsqueeze(0), op3.unsqueeze(0)), 0)   
    return (op1, op2, op3, forward_output, flatten_input1, flatten_input2)

  def inverse_loss(self, outputs, targets):
    op1, op2, op3, forward_op, latent_image, latent_predicition = outputs
    XY_BIN, THETA, LENGTH = 0, 1, 2
    target1, target2, target3 = targets[:,XY_BIN], targets[:,THETA], targets[:,LENGTH]
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss().cuda()
    # op = torch.LongTensor(target1)
    #op1 = op1.to(torch.int64)
    loss1 = criterion(op1.float(), target1.long()) 
    loss2 = criterion(op2.float(), target2.long()) 
    loss3 = criterion(op3.float(), target3.long()) 
    MSEloss = nn.MSELoss()
    forward_loss = MSEloss(latent_image, latent_predicition) 
    total_loss = loss1 + loss2 + loss3 + self.lamda * forward_loss             
    return total_loss

  def calculate_accuracy(self, data, labels):

     img1 = data[:,0,:,:,:]
     img2 = data[:,1,:,:,:]
     outputs = self.forward(img1, img2, labels)
     self.zero_grad()
     loss = self.inverse_loss(outputs, labels)
     accuracies = []
     classifier_accuracies = {}


     for op_idx, op in enumerate(outputs[:3]):
       op_labels = torch.argmax(op, axis=1)
       success = sum([1 for data_idx in range(len(op_labels))\
                      if op_labels[data_idx] == labels[data_idx, op_idx]])
       accuracies.append(success / len(op_labels))
     classifier_accuracies['loc_xy'] = accuracies[0]
     classifier_accuracies['angle'] = accuracies[1]
     classifier_accuracies['length'] = accuracies[2]
     classifier_accuracies['overall'] = sum(accuracies) / 3
     classifier_accuracies['loss'] = loss.data
     del loss
     del outputs
     return classifier_accuracies

  
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
    self.__device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    self.__data_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    params = {'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 6}

    self.train_data_generator = data.DataLoader(partitioned_datasets['train'], **params)
    self.val_data_generator = data.DataLoader(partitioned_datasets['val'], **params)
    self.test_data_generator = data.DataLoader(partitioned_datasets['test'], **params)


    self.__net = Net()
    # self.__net.apply(weights_init)
    self.__net.transfer_weigths_from_alexnet()
    self.__net = self.__net.to(self.__device)
    self.__net.lamda = experiment_details['lamda']
    #tensor_shape = next(self.train_data_generator)[0][0].shape
    self.sample_data = partitioned_datasets['train'][0]
    tensor_shape = self.sample_data[0].shape
    self.__IMG_HEIGHT = tensor_shape[2]
    self.__IMG_WIDTH = tensor_shape[3]
    self.__NO_OF_CHANNELS = tensor_shape[1]
    self.EPOCHS = EPOCHS
    self.BATCH_SIZE = BATCH_SIZE
    self.exp_name = experiment_details['exp_name']
    path_to_write = os.path.join('run', experiment_name)
    self.__writer = SummaryWriter(path_to_write)
    self.__accuracies = {}
    

  def __write_values_to_graph(self, dataset_name, accuracy, iter_count):

    self.__writer.add_scalar('accuracy/'+dataset_name+'loc_xy', accuracy['loc_xy'], iter_count)
    self.__writer.add_scalar('accuracy/'+dataset_name+'angle', accuracy['angle'], iter_count)
    self.__writer.add_scalar('accuracy/'+dataset_name+'length', accuracy['length'], iter_count) 
    self.__writer.add_scalar('accuracy/'+dataset_name+'overall', accuracy['overall'], iter_count)
    self.__writer.add_scalar('loss/' + dataset_name, accuracy['loss'], iter_count)


  def __plot_accuracy_graphs(self, dataset_name, iter_count): 

    if(dataset_name == 'val'):
        data_generator = self.val_data_generator
    elif(dataset_name == 'test'):
      data_generator = self.test_data_generator
    elif(dataset_name == 'train'):
      data_generator = self.train_data_generator
    #dataset_x = dataset_x.to(self.__device)
    #labels = lables.to(self.__device)


    full_accuracy = {'loc_xy':0, 'angle':0, 'length':0, 'overall':0, 'loss':0}
    count = 0
    for batch_x, labels in data_generator:
      batch_x = batch_x.to(self.__device)
      labels = labels.to(self.__device)

      batch_accuracy = self.__net.calculate_accuracy(batch_x, labels)
      count += 1
      for key, value in batch_accuracy.items():
        full_accuracy[key] += value
      torch.cuda.empty_cache() 

    for key, value in full_accuracy.items():
      full_accuracy[key] = value / count

    self.__write_values_to_graph(dataset_name, full_accuracy, iter_count)
    self.__accuracies[dataset_name] = full_accuracy
     




  def train_network(self):
  

    optimizer = optim.Adam(self.__net.parameters(), lr=0)
    X, y  = self.sample_data
    y = y.to(self.__device).float().unsqueeze(0)
    #loss_function = nn.MSELoss()

    # For visualisation
    img1 = X[0,:,:,:].to(self.__device)
    img2 = X[1,:,:,:].to(self.__device)
  
    self.__writer.add_image('baxter_poking_image', img1 * 255.0)
    self.__writer.add_image('baxter_poking_image', img2 * 255.0)
    self.__writer.add_graph(self.__net, (img1.unsqueeze(0), img2.unsqueeze(0), y))
   # TODO: Find how model with weights can be visualised
    # self.__writer.add_graph(self.__net, self.__train_x[0,:,:,:,:])



    index =0

    for epoch in range(self.EPOCHS):
      for batch_x, batch_y in tqdm(self.train_data_generator):
        index += 1

        self.__net.zero_grad()
        batch_x = batch_x.to(self.__device)
        batch_y = batch_y.to(self.__device)

        batch_x_img1 = batch_x[:,0,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)
        batch_x_img2 = batch_x[:,1,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)

        outputs = self.__net(batch_x_img1, batch_x_img2, batch_y)
        loss = self.__net.inverse_loss(outputs, batch_y)
        print(loss)
        # get_dot = register_hooks(loss)
        loss.backward()
        # dot = get_dot()
        # dot.save('tmp.dot')
        # plot_grad_flow(self.__net.named_parameters())


        optimizer.step()

        batch_accuracy = self.__net.calculate_accuracy(batch_x, batch_y)

        self.__accuracies['train'] = batch_accuracy
        self.__write_values_to_graph(dataset_name = 'train', accuracy = batch_accuracy,
                                     iter_count = index)
        del loss
        del outputs
        if index > 5000:
          optimizer = optim.Adam(self.__net.parameters(), lr=1e-4)

      print('validation')
      self.__plot_accuracy_graphs(dataset_name='val', iter_count=index)
        #self.__plot_accuracy_graphs(dataset_name='train', iter_count=epoch * self.BATCH_SIZE + i)
      print('end validation')


    self.__plot_accuracy_graphs(dataset_name='test', iter_count = index)
    torch.save(self.__net.state_dict(), os.path.join('./models', self.exp_name + '.pt'))
    results = [self.__accuracies['test']['overall'], self.__accuracies['val']['overall'],\
               self.__accuracies['train']['overall'], self.__accuracies['test']['loss'],\
               self.__accuracies['val']['loss'], self.__accuracies['train']['loss'],\
               self.__accuracies['test']['loc_xy'], self.__accuracies['test']['angle'],\
               self.__accuracies['test']['length'], self.__accuracies['val']['loc_xy'],\
               self.__accuracies['val']['angle'], self.__accuracies['val']['length'],\
               self.__accuracies['train']['loc_xy'], self.__accuracies['train']['angle'],\
               self.__accuracies['train']['length']]
    return results

 

    # self.__writer.close()


if __name__=='__main__':

  base_path = '../data/processed_poke_3'
  labels = {}
  ids = {}
  experiment_name = sys.argv[1]
  no_of_epochs = int(sys.argv[2])
  seed_no = int(sys.argv[3])
  experiment_name = experiment_name + '_epoch_' + str(no_of_epochs) + '_'+\
                     datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
  lamda = float(sys.argv[4])
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
  experiment_details['lamda'] = lamda

  exp_details = [date.today().strftime("%d/%m/%Y"), datetime.now().strftime("%H:%M:%S"),
                 experiment_name, no_of_epochs, seed_no, lamda]

  dl_trainer = networkTrainer(partitioned_datasets, EPOCHS=no_of_epochs, experiment_details=experiment_details)
  results = dl_trainer.train_network()
  results = list(map(float, results))
  row_to_write = exp_details + results
  write_to_gsheet(row_to_write)
