import os
import glob
import math

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2


class BaxterPokingDataReader:
  def __init__(self, base_path='./data'):
    self._base_path = base_path
    self._total_data = []
    self._image_width = -1
    self._image_height = -1
    self._xbin_count = 20
    self._ybin_count = 20
    self._anglebin_count = 20
    self._len_actionbin_count = 15
 
  def load_data_from_directory(self):
    data_runs = os.listdir(self._base_path)
    
    for folder in data_runs:
      folder_path = os.path.join(self._base_path, folder)
      gt_file_path = os.path.join(folder_path, 'actions.npy')
      ground_truth = np.load(gt_file_path)
      images_names = os.listdir(os.path.join(folder_path))
      previous_image = cv2.imread(os.path.join(folder_path, images_names[0]))
      for index, img_name in enumerate(images_names[1:]):
        this_data = []
        current_image = cv2.imread(os.path.join(folder_path, img_name))
        this_data.append([previous_image, current_image])
        this_data.append(ground_truth[index])
        self._total_data.append(this_data)
        previous_image = current_image
    self._image_height = current_image.shape[0]
    self._image_width = current_image.shape[1]
    print(len(self._total_data))

  def remove_invalid_data(self):
    GT_INDEX=1
    IS_VALID_INDEX=-1
    self._total_data = [data for data in self._total_data if 
                        data[GT_INDEX][IS_VALID_INDEX]]

  def dicretise_gt_actions(self):
    discretised_data = []
    ACTION_COORD_X=0
    ACTION_COORD_Y=1
    ANGLE=2
    ACTIONLEN=3
    TOTAL_ANGLE = 2 * np.pi
    for data in self._total_data:
      images, groundtruth = data
      groundtruth[ACTION_COORD_X] =  round(groundtruth[ACTION_COORD_X] / self._image_width
                                    * self._xbin_count)
      groundtruth[ACTION_COORD_Y] =  round(groundtruth[ACTION_COORD_Y] / self._image_height
                                    * self._ybin_count)
      groundtruth[ANGLE] = round(groundtruth[ANGLE] / TOTAL_ANGLE * self._anglebin_count)
      groundtruth[ACTIONLEN] = round(groundtruth[ACTIONLEN] * 100)
      discretised_data.append([images, groundtruth.astype(np.int8)])
    self._total_data = discretised_data

  def read_and_process_data(self):
    self.load_data_from_directory()
    self.remove_invalid_data()
    self.dicretise_gt_actions()
      
      
  
  #TODO:Construct a visualization function to view the data

if __name__=='__main__':
  data_directory  = '/home/senthilpalanisamy/work/winter_project/my_code/data/mini_data'
  pokedata = BaxterPokingDataReader(data_directory)
  pokedata.read_and_process_data()


    
