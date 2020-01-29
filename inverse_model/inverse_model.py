import os

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchsummary import summary

from dataset_reader import BaxterPokingDataReader


class Net(nn.Module):
  def __init__(self):
    super().__init__()

    self._to_linear = None
    self.conv1 = nn.Conv2d(3, 64, 11)
    self.conv2 = nn.Conv2d(64, 192, 5)
    self.conv3 = nn.Conv2d(192, 384, 3)
    self.conv4 = nn.Conv2d(384, 256, 3)
    # self.fc1 = nn.Linear(self._to_linear, 100)
    self.conv5 = nn.Conv2d(256, 256, 3)
    self.conv6 = nn.Conv2d(256, 200, 3)
    #self.fc1 = nn.Linear(self._to_linear, 100)
    self.fc2 = nn.Linear(200, 40)
    self.EPOCHS = 100
    self.BATCH_SIZE = 8
    self._IMAGE_WIDTH = 224
    self._IMAGE_COL = 224



  def forward(self, image_pair):
    latent_features = []
    for image in image_pair:
      x = F.relu(self.conv1(image))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      latent_features.append(x)

    if self._to_linear is None:
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        self._to_linear = self._to_linear * 2
        self.fc1 = nn.Linear(self._to_linear, 200)
    
    N = latent_features[0].shape[0]
    flatten_input1 = latent_features[0].reshape(N, -1)
    flatten_input2 = latent_features[1].reshape(N, -1)
    x = torch.cat((flatten_input1, flatten_input2), 1) 
    x = self.fc1(x.view(-1, self._to_linear))
    x = self.fc2(x)
    return F.softmax(x, dim=1)
  
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
  def __init__(self, dataset, EPOCHS=3, BATCH_SIZE=8):
    self.__net = Net()
    self.__net = self.__net.transfer_weigths_from_alexnet()
    self.__dataset = dataset
    self.__IMG_HEIGHT = dataset[0][0].shape[2]
    self.__IMG_WIDTH = dataset[0][0].shape[3]
    self.__NO_OF_CHANNELS = dataset[0][0].shape[1]
    self.__VAL_PERCENTAGE = 0.1
    self.__TEST_PERCENTAGE = 0.1
    self.EPOCHS = EPOCHS
    self.BATCH_SIZE = BATCH_SIZE
    self.__partition_dataset()


  def train_network(self, train_data):
  

    optimizer = optim.Adam(self.__net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(self._EPOCHS):
      for i in tqdm(range(0, len(self._train_x), self._BATCH_SIZE)):
        batch_x = train_x[i:i+BATCH_SIZE].view(-1, 2, self.__NO_OF_CHANNELS , 
                                              self.__IMG_HEIGHT, self.__IMG_WIDTH)
        batch_y = train_y[i:i+BATCH_SIZE]
        self.__net.zero_grad()
        outputs = self.__net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

  def __partition_dataset(self):

    # 2 because this is a simese type network
    X = torch.Tensor([i[0] for i in self.__dataset]).view(-1, 2, self.__NO_OF_CHANNELS , 
                                                         self.__IMG_HEIGHT, self.__IMG_WIDTH)
    X = X / 255.0
    # Only x is predicted as of now
    Y = torch.Tensor([i[1][0] for i in self.__dataset])

    val_size = int(len(X) * self.__VAL_PERCENTAGE)
    print(val_size)

    test_size = int(len(X) * self.__TEST_PERCENTAGE)

    self._train_x = X[:-val_size-test_size]
    self._train_y = y[:-val_size-test_size]
    self._val_x = X[-val_size:]
    self._val_y = X[-val_size:]
    self._test_x = X[-val_size - test_size:-val_size]
    self._test_y = X[-val_size - test_size:-val_size] 

    
if __name__=='__main__':
  dataset_path = '/home/senthilpalanisamy/work/winter_project/my_code/data/mini_data'
  poking_data = BaxterPokingDataReader(dataset_path)
  poking_data.read_and_process_data()
  dl_trainer = networkTrainer(poking_data.total_data[:100])
  #model_state = net.state_dict()
  #net.tranfer_weigths_from(alexnet_state)

  # net = net.double()
  # image1 = torch.tensor(np.zeros((1, 3, 224, 224)))
  # image2 = torch.tensor(np.zeros((1, 3, 224, 224)))
  # joint_image = [image1.double(), image2.double()]
  # net.zero_grad()
  # outputs = net(joint_image)
  # print('finished')


