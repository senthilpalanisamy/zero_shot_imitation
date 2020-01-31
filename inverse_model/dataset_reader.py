import os
import glob
import math

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2

from config import *


class BaxterPokingDataReader:

  #TODO:Construct a visualization function to view the data
  #TODO:Check if one hot encoding is necessary for the data
  #TODO:Does storing as npy object make it faster to read data
  def __init__(self, base_path='./data'):
    self.__base_path = base_path
    self.total_data = []
    self.__image_width = -1
    self.__image_height = -1
    self.__xbin_count = XBIN_COUNT
    self.__ybin_count = YBIN_COUNT
    self.__xybin_count = XYBIN_COUNT
    self.__anglebin_count = ANGLE_BIN_COUNT
    self.__len_actionbin_count = LEN_ACTIONBIN_COUNT
    self.__class_counts = [self.__xbin_count, self.__ybin_count,
                           self.__xybin_count, self.__anglebin_count, 
                           self.__len_actionbin_count]
 
  def __load_data_from_directory(self):
    data_runs = os.listdir(self.__base_path)
    
    for folder in data_runs:
      folder_path = os.path.join(self.__base_path, folder)
      gt_file_path = os.path.join(folder_path, 'actions.npy')
      ground_truth = np.load(gt_file_path)
      N = len(ground_truth)
      ground_truth = np.hstack((ground_truth[:,:2], np.ones((N,1))*-1, ground_truth[:, 2:]))
      image_names = os.listdir(folder_path)
      image_names = [file_name for file_name in image_names if file_name.endswith('.jpg')]
      previous_image = cv2.imread(os.path.join(folder_path, image_names[0]))
      
      for index, img_name in enumerate(image_names[1:]):
        this_data = []
        current_image = cv2.imread(os.path.join(folder_path, img_name))
        #print(ground_truth[index])
        concat_image = np.concatenate((current_image[np.newaxis,: ], 
                                       previous_image[np.newaxis, :]), axis=0)
        concat_image = np.moveaxis(concat_image, -1, 1)
        this_data.append(concat_image)
        this_data.append(ground_truth[index])
        self.total_data.append(this_data)
        previous_image = current_image
    self.__image_height = current_image.shape[0]
    self.__image_width = current_image.shape[1]
    print(len(self.total_data))

  def __remove_invalid_data(self):
    GT_INDEX=1
    IS_VALID_INDEX=-1
    self.total_data = [data for data in self.total_data if 
                        data[GT_INDEX][IS_VALID_INDEX]]

  def __dicretise_gt_actions(self):
    discretised_data = []
    ACTION_COORD_X=0
    ACTION_COORD_Y=1
    XY_ACTIONBIN = 2

    ANGLE=3
    ACTIONLEN=4
    TOTAL_ANGLE = 2 * np.pi
    GT_TOTAL_NO = 4
    for data in self.total_data:
      images, groundtruth = data
      groundtruth[ACTION_COORD_X] =  round(groundtruth[ACTION_COORD_X] / self.__image_width
                                    * (self.__xbin_count-1))
      groundtruth[ACTION_COORD_Y] =  round(groundtruth[ACTION_COORD_Y] / self.__image_height
                                    * (self.__ybin_count-1))
      groundtruth[XY_ACTIONBIN] = (groundtruth[ACTION_COORD_Y]+1) * self.__ybin_count +\
                                  groundtruth[ACTION_COORD_X]
                                  
                                  

      groundtruth[ANGLE] = round(groundtruth[ANGLE] / TOTAL_ANGLE * (self.__anglebin_count-1))
      groundtruth[ACTIONLEN] = round(groundtruth[ACTIONLEN] * 100)
      groundtruth = groundtruth.astype(np.int16)
      gt_onehot_vectors=[]
      print(groundtruth)
      for index, gt_value in enumerate(groundtruth[:-1]):
        #Construct a one-hot vector for ground truth
        gt_onehot_vectors.append(np.eye(self.__class_counts[index])[gt_value])
      discretised_data.append([images, gt_onehot_vectors, groundtruth])
    self.total_data = discretised_data

  def read_and_process_data(self):
    self.__load_data_from_directory()
    self.__remove_invalid_data()
    self.__dicretise_gt_actions()

  def get_stored_data(self):
    return self.total_data
   # How costly is having two copies of the same data
      
      
  

if __name__=='__main__':
  data_directory  = '/home/senthilpalanisamy/work/winter_project/my_code/data/mini_data'
  pokedata = BaxterPokingDataReader(data_directory)
  pokedata.read_and_process_data()


    
