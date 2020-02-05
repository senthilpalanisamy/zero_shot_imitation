import os
import time
import json

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
    self.pre_xy_classifier = nn.Linear(self.__to_linear, 1000)
    #self.fc1 = nn.Linear(19009600, 200) 
    # Action_classifier
    self.xy_classifier = nn.Linear(1000, XYBIN_COUNT) 
    self.pre_angle_classifier = nn.Linear(self.__to_linear + XYBIN_COUNT, 200)
    self.angle_classifier = nn.Linear(200, ANGLE_BIN_COUNT)
    self.pre_length_classifier = nn.Linear(self.__to_linear + XYBIN_COUNT + ANGLE_BIN_COUNT,
                                    200)
    self.length_classifier = nn.Linear(200, LEN_ACTIONBIN_COUNT)
    self.EPOCHS = 100
    self.BATCH_SIZE = 8
    self._IMAGE_WIDTH = 224
    self._IMAGE_COL = 224
    self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



  def forward(self, image1, image2):
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
    op1 = F.softmax(x, dim=1)
    angle_concat = torch.cat((latent_2images, op1), 1) 
    x = self.pre_angle_classifier(angle_concat)
    x = self.angle_classifier(x)
    op2 = F.softmax(x, dim=1)
    x = torch.cat((angle_concat, op2), 1)
    x = self.pre_length_classifier(x)
    x = self.length_classifier(x)
    op3 = F.softmax(x, dim=1)
    #final_op = torch.cat((op1.unsqueeze(0), op2.unsqueeze(0), op3.unsqueeze(0)), 0)   
    return (op1, op2, op3)

  def inverse_loss(self, outputs, targets):
    op1, op2, op3 = outputs
    XY_BIN, THETA, LENGTH = 2, 3, 4
    target1, target2, target3 = targets[:,XY_BIN], targets[:,THETA], targets[:,LENGTH]
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss().cuda()
    # op = torch.LongTensor(target1)
    #op1 = op1.to(torch.int64)
    loss1 = criterion(op1.float(), target1.long()) 
    loss2 = criterion(op2.float(), target2.long()) 
    loss3 = criterion(op3.float(), target3.long()) 
    total_loss = loss1 + loss2 + loss3             
    return total_loss

  def calculate_accuracy(self, data, labels):

     img1 = data[:,0,:,:,:]
     img2 = data[:,1,:,:,:]
     outputs = self.forward(img1, img2)
     self.zero_grad()
     loss = self.inverse_loss(outputs, labels)
     accuracies = []
     classifier_accuracies = {}


     for op_idx, op in enumerate(outputs):
       op_labels = torch.argmax(op, axis=1)
       success = sum([1 for data_idx in range(len(op_labels))\
                      if op_labels[data_idx] == labels[data_idx, op_idx+2]])
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
  def __init__(self, partitioned_datasets, EPOCHS=100, BATCH_SIZE=100):
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
    #tensor_shape = next(self.train_data_generator)[0][0].shape
    self.sample_data = partitioned_datasets['train'][0]
    tensor_shape = self.sample_data[0].shape
    self.__IMG_HEIGHT = tensor_shape[2]
    self.__IMG_WIDTH = tensor_shape[3]
    self.__NO_OF_CHANNELS = tensor_shape[1]
    self.EPOCHS = EPOCHS
    self.BATCH_SIZE = BATCH_SIZE
    self.__writer = SummaryWriter()

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
     




  def train_network(self):
  

    optimizer = optim.Adam(self.__net.parameters(), lr=0.001)
    X, y  = self.sample_data
    #loss_function = nn.MSELoss()

    # For visualisation
    img1 = X[0,:,:,:].to(self.__device)
    img2 = X[1,:,:,:].to(self.__device)
  
    self.__writer.add_image('baxter_poking_image', img1)
    self.__writer.add_image('baxter_poking_image', img2)
    self.__writer.add_graph(self.__net, (img1.unsqueeze(0), img2.unsqueeze(0)))
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

        outputs = self.__net(batch_x_img1, batch_x_img2)
        loss = self.__net.inverse_loss(outputs, batch_y)
        print(loss)
        loss.backward()
        optimizer.step()

        batch_accuracy = self.__net.calculate_accuracy(batch_x, batch_y)
        self.__write_values_to_graph(dataset_name = 'train', accuracy = batch_accuracy,
                                     iter_count = index)
        del loss
        del outputs
      print('validation')
      self.__plot_accuracy_graphs(dataset_name='val', iter_count=index)
        #self.__plot_accuracy_graphs(dataset_name='train', iter_count=epoch * self.BATCH_SIZE + i)
      print('end validation')


    self.__plot_accuracy_graphs(dataset_name='test', iter_count = index)
    self.__net.save_state_dict('mytraining.pt')

 

    # self.__writer.close()


if __name__=='__main__':

  base_path = '../data/processed_poke'
  labels = {}
  ids = {}
  partitioned_datasets = {}
  for data_partition_name in ['train', 'val', 'test']:
    label_path = os.path.join(base_path, data_partition_name, 'labels.json')
    ids_path = os.path.join(base_path, data_partition_name, 'ids.json')

    with open(label_path) as json_file:
      labels[data_partition_name] = json.load(json_file)

    with open(ids_path) as json_file:
      ids[data_partition_name] = json.load(json_file)

  partitioned_datasets['train']  = Dataset(ids['train'], labels['train'], partition='train')
  # training_generator = data.DataLoader(train_dataset, **params)

  partitioned_datasets['test']  = Dataset(ids['test'], labels['test'], partition='test')
  # test_generator = data.DataLoader(test_dataset, **params)

  partitioned_datasets['val'] = Dataset(ids['val'], labels['val'], partition='val')

  dl_trainer = networkTrainer(partitioned_datasets, EPOCHS=100)
  dl_trainer.train_network()
  #model_state = net.state_dict()
  #net.tranfer_weigths_from(alexnet_state)

  # net = net.double()
  # image1 = torch.tensor(np.zeros((1, 3, 224, 224)))
  # image2 = torch.tensor(np.zeros((1, 3, 224, 224)))
  # joint_image = [image1.double(), image2.double()]
  # net.zero_grad()
  # outputs = net(joint_image)
  # print('finished')


