import os
import time

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

from dataset_reader import BaxterPokingDataReader
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
    target1, target2, target3 = targets[:,0], targets[:,1], targets[:,2]
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
                      if op_labels[data_idx] == labels[data_idx, op_idx]])
       accuracies.append(success / len(op_labels))
     classifier_accuracies['loc_xy'] = accuracies[0]
     classifier_accuracies['angle'] = accuracies[1]
     classifier_accuracies['length'] = accuracies[2]
     classifier_accuracies['overall'] = sum(accuracies) / 3
     classifier_accuracies['loss'] = loss
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
  def __init__(self, dataset, EPOCHS=3, BATCH_SIZE=8):
    self.__device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    self.__data_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    self.__net = Net()
    self.__net.transfer_weigths_from_alexnet()
    # self.__net.to(self.__device)
    # self.__net = self.__net.cuda()
    self.__net = self.__net.to(self.__device)
    # self.__net = torch.nn.DataParallel(self.__net, device_ids=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    self.__dataset = dataset
    self.__IMG_HEIGHT = dataset[0][0].shape[2]
    self.__IMG_WIDTH = dataset[0][0].shape[3]
    self.__NO_OF_CHANNELS = dataset[0][0].shape[1]
    self.__VAL_PERCENTAGE = 0.1
    self.__TEST_PERCENTAGE = 0.1
    self.EPOCHS = EPOCHS
    self.BATCH_SIZE = BATCH_SIZE
    self.__partition_dataset()
    self.__writer = SummaryWriter()

  def __plot_accuracy_graphs(self, dataset_name, iter_count): 

    if(dataset_name == 'val'):
      dataset_x = self.__val_x
      labels = self.__val_y
    elif(dataset_name == 'test'):
      dataset_x = self.__test_x
      labels = self.__test_y
    elif(dataset_name == 'train'):
      dataset_x = self.__train_x
      labels = self.__train_y
      
    accuracy = self.__net.calculate_accuracy(dataset_x, labels)

    self.__writer.add_scalar('accuracy/'+dataset_name+'loc_xy', accuracy['loc_xy'], iter_count)
    self.__writer.add_scalar('accuracy/'+dataset_name+'angle', accuracy['angle'], iter_count)
    self.__writer.add_scalar('accuracy/'+dataset_name+'length', accuracy['length'], iter_count) 
    self.__writer.add_scalar('accuracy/'+dataset_name+'overall', accuracy['overall'], iter_count)
    self.__writer.add_scalar('loss/' + dataset_name, accuracy['loss'], iter_count)



  def train_network(self):
  

    optimizer = optim.Adam(self.__net.parameters(), lr=0.001)
    #loss_function = nn.MSELoss()

    # For visualisation
    img1 = self.__train_x[0,0,:,:,:]
    img2 = self.__train_x[0,1,:,:,:]
  
    self.__writer.add_image('baxter_poking_image', img1)
    self.__writer.add_image('baxter_poking_image', img2)
    self.__writer.add_graph(self.__net, (img1.unsqueeze(0), img2.unsqueeze(0)))
    # TODO: Find how model with weights can be visualised
    # self.__writer.add_graph(self.__net, self.__train_x[0,:,:,:,:])




    for epoch in range(self.EPOCHS):
      for i in tqdm(range(0, self.__train_x.shape[0], self.BATCH_SIZE)):
        batch_x = self.__train_x[i:i+self.BATCH_SIZE].to(self.__data_device)
        batch_y = self.__train_y[i:i+self.BATCH_SIZE].to(self.__data_device)
        # batch_x_img1 = batch_x[:,0,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)
        # batch_x_img2 = batch_x[:,1,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)

        batch_x_img1 = batch_x[:,0,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)
        batch_x_img2 = batch_x[:,1,:,:,:].reshape(-1,self.__NO_OF_CHANNELS, self.__IMG_HEIGHT, self.__IMG_WIDTH)

        # self.__net.zero_grad()
        outputs = self.__net(batch_x_img1, batch_x_img2)
        loss = self.__net.inverse_loss(outputs, batch_y)
        #loss = loss_function(outputs, batch_y)
        print(loss)
        loss.backward()
        optimizer.step()
        self.__plot_accuracy_graphs(dataset_name='val', iter_count=epoch * self.BATCH_SIZE + i)
        self.__plot_accuracy_graphs(dataset_name='train', iter_count=epoch * self.BATCH_SIZE + i)


    self.__plot_accuracy_graphs(dataset_name='test', iter_count=epoch * self.BATCH_SIZE + i)

 

    # self.__writer.close()


  def __partition_dataset(self):

    # 2 because this is a simese type network
    X = torch.Tensor([i[0] for i in self.__dataset]).view(-1, 2, self.__NO_OF_CHANNELS , 
                                                         self.__IMG_HEIGHT, self.__IMG_WIDTH).to(self.__data_device)
    X = X / 255.0
    # Only x is predicted as of now
    # Think on how to handle this better
    Y = torch.Tensor([i[2][2:6] for i in self.__dataset]).to(self.__data_device)

    val_size = int(len(X) * self.__VAL_PERCENTAGE)
    print(val_size)

    test_size = int(len(X) * self.__TEST_PERCENTAGE)

    self.__train_x = X[:-val_size-test_size]
    self.__train_y = Y[:-val_size-test_size]
    self.__val_x = X[-val_size:]
    self.__val_y = Y[-val_size:]
    self.__test_x = X[-val_size - test_size:-val_size]
    self.__test_y = Y[-val_size - test_size:-val_size] 


    
if __name__=='__main__':
  dataset_path = '../data/baxter_poke'
  poking_data = BaxterPokingDataReader(dataset_path)
  poking_data.read_and_process_data()
  dl_trainer = networkTrainer(poking_data.total_data[:100], EPOCHS=100)
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


