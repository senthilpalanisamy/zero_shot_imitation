import os
import glob
import math

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2

class BaxterPokingDataReader:
  def __init__(self, base_path='./data'):
    self.base_path = base_path
    self.total_data = []
 
  def load_data_from_directory(self):
    data_runs = os.listdir(self.base_path)
    
    for folder in data_runs:
      folder_path = os.path.join(self.base_path, folder)
      gt_file_path = os.path.join(folder_path, 'actions.npy')
      ground_truth = np.load(gt_file_path)
      images_names = os.listdir(os.path.join(folder_path))
      previous_image = cv2.imread(os.path.join(folder_path, images_names[0]))
      for index, img_name in enumerate(images_names[1:]):
        this_data = []
        current_image = cv2.imread(os.path.join(folder_path, img_name))
        this_data.append([previous_image, current_image])
        this_data.append(ground_truth[index])
        self.total_data.append(this_data)
        previous_image = current_image
    print(len(self.total_data))
  
  #TODO:Construct a visualization function to view the data

if __name__=='__main__':
  data_directory  = '/home/senthilpalanisamy/work/winter_project/my_code/data/mini_data'
  pokedata = BaxterPokingDataReader(data_directory)
  pokedata.load_data_from_directory()


    
