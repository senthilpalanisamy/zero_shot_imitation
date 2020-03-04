import os
import glob
import math

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2
import torch
import json
import random


class BaxterPokingDataReader:

  #TODO:Construct a visualization function to view the data
  #TODO:Check if one hot encoding is necessary for the data
  #TODO:Does storing as npy object make it faster to read data
  def __init__(self, base_path='./data'):
    self.__base_path = base_path
    self.total_data = []
    self.__image_width = -1
    self.__image_height = -1
    self.__labels = {}
    self.__ids = []
    self.__prefix_name = 'goal_'
    self.__write_path = '/senthil/data/goal_recog_data_1'
    self.__VAL_PERCENTAGE = 0.1
    self.__TEST_PERCENTAGE = 0.1
 
  def __load_data_from_directory(self):

    data_runs = os.listdir(self.__base_path)
    
    for folder in data_runs:

      folder_path = os.path.join(self.__base_path, folder)
      gt_file_path = os.path.join(folder_path, 'actions.npy')
      ground_truth = np.load(gt_file_path)
      image_names = os.listdir(folder_path)
      image_names = sorted([file_name for file_name in image_names if file_name.endswith('.jpg')])
      previous_image = cv2.imread(os.path.join(folder_path, image_names[0]))
      indices_to_remove = []
      
      for index, img_name in enumerate(image_names[1:-1]):
        negative_samples = list(image_names)
        #print(index, len(image_names))
        for index_to_remove in [index+2, index+1, index]:
          negative_samples.pop(index_to_remove)
        neg_image_name = random.choice(negative_samples)

          
        this_data = []
        current_image = cv2.imread(os.path.join(folder_path, img_name))
        positive_image = cv2.imread(os.path.join(folder_path, image_names[index+2]))
        negative_image = cv2.imread(os.path.join(folder_path, neg_image_name))
        
        positive_pair = np.concatenate((current_image[np.newaxis,: ], 
                                       positive_image[np.newaxis, :]), axis=0)

        negative_pair = np.concatenate((current_image[np.newaxis,: ], 
                                       negative_image[np.newaxis, :]), axis=0)
        if(ground_truth[index][-1]== 1):
          positive_pair = positive_pair.reshape((2, 3, 240, 240))
          negative_pair = negative_pair.reshape((2, 3, 240, 240))
          this_data.append(positive_pair)
          this_data.append(1)
          self.total_data.append(this_data)

          this_data = []
          this_data.append(negative_pair)
          this_data.append(0)
          self.total_data.append(this_data)

          previous_image = current_image
    self.__image_height = current_image.shape[0]
    self.__image_width = current_image.shape[1]
    print(len(self.total_data))


  def __write_processed_data(self, partition_name, start_index, end_index):

    DATA_INDEX = 0
    GT_INDEX = 1
    partiion_labels = {}
    ids = []
    write_path = os.path.join(self.__write_path, partition_name)
    if not os.path.exists(write_path):
      os.makedirs(write_path)

    for index in range(start_index, end_index):
      this_label = self.__prefix_name + partition_name+ str(index)
      partiion_labels[this_label] = self.total_data[index][GT_INDEX]
      ids.append(this_label)
      torch.save(self.total_data[index][DATA_INDEX], 
                 os.path.join(write_path, this_label + '.pt'))
   
    with open(os.path.join(write_path, 'labels.json'), 'w') as labels_file:
      json.dump(partiion_labels, labels_file)

    with open(os.path.join(write_path, 'ids.json'), 'w') as ids_file:
      json.dump(ids, ids_file)


  def read_and_process_data(self):
    self.__load_data_from_directory()
    random.shuffle(self.total_data)
    data_len = len(self.total_data) 
    test_index_start = 0
    test_index_end = int(self.__TEST_PERCENTAGE * data_len)
    val_index_start = test_index_end 
    val_index_end = val_index_start + int(self.__VAL_PERCENTAGE * data_len)
    train_index_start = val_index_end
    train_index_end = data_len
    self.__write_processed_data(partition_name = 'train', start_index = train_index_start, 
                                end_index = train_index_end)
    self.__write_processed_data(partition_name='test', start_index = test_index_start, 
                                end_index = test_index_end)
    self.__write_processed_data(partition_name='val', start_index = val_index_start, 
                                end_index = val_index_end)



      
  

if __name__=='__main__':
  data_directory  = '../data/baxter_poke'
  pokedata = BaxterPokingDataReader(data_directory)
  pokedata.read_and_process_data()


    
