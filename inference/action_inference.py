import os
import json

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_generator import Dataset
from torch.utils import data
from inverse_model import *



class RecurrentActionPredictor:
  def __init__(self, model_paths):
    goal_recognier_path, joint_model_path = model_paths
       
    self.joint_model = Net()
    self.joint_model.load_state_dict(torch.load(joint_model_path))
    self.image_sequence = []
    self.current_goal_image = None

  def is_goal_reached(self):
    return False
   
  def predict_next_action(self, current_image):
    if(self.is_goal_reached()):
      if(not image_sequence):
        return True, None
      self.current_goal_image = self.image_sequence.pop(0)
    # Need when running on actual images
    #current_image_tensor = torch.from_numpy(current_image) / 255.0
    dummy_action = torch.tensor([1.,1.,1.]).unsqueeze(0)
    current_image_tensor = current_image
    all_outputs = self.joint_model(current_image_tensor, self.current_goal_image, dummy_action)[:3]
    next_action = [torch.argmax(op, axis=1) for op in all_outputs]
    return False, next_action

def return_data_set_generator():

  base_path = '../data/processed_poke_3'
  data_partition_name = 'test'
  label_path = os.path.join(base_path, data_partition_name, 'labels.json')
  ids_path = os.path.join(base_path, data_partition_name, 'ids.json')

  with open(label_path) as json_file:
    labels = json.load(json_file)

  with open(ids_path) as json_file:
    ids = json.load(json_file)
  test_dataset = Dataset(ids, labels, partition='test', base_path=base_path)

  params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 1}

  test_data_generator = data.DataLoader(test_dataset, **params)
  return test_data_generator


if __name__=='__main__':
    model_paths = [ None,
                    '../inverse_model_alternate/models/joint_model_7_epoch_1000_01_03_2020_00:28:18.pt'] 
    baxter_poke_predictor = RecurrentActionPredictor(model_paths)
    test_dataset = return_data_set_generator()
    base_path = '/senthil/results/test_data_results'

    scaling_factor = 1.0
    X_BINS = 20
    Y_BINS = 20
    XYBIN_COUNT = XBIN_COUNT * YBIN_COUNT
    ANGLE_BIN_COUNT = 20
    LEN_ACTIONBIN_COUNT = 16
    IMAGE_WIDTH = 240
    IMAGE_HEIGHT = 240
    index = 0

    for images, action in test_dataset:
        image1=  images[:,0,:,:,:]
        goal_image=  images[:,1,:,:,:]
        baxter_poke_predictor.current_goal_image = goal_image
        is_finished, next_action = baxter_poke_predictor.predict_next_action(image1)
        xy, theta, length = [next_action[0], next_action[1], next_action[2]]
        image1 = image1.reshape(240, 240, 3)
        image2 = goal_image.reshape(240, 240, 3)


        y = (xy // X_BINS + 0.5) * (IMAGE_HEIGHT-1) / (Y_BINS - 1 ) * scaling_factor
        x = (xy % int(X_BINS) + 0.5) * (IMAGE_WIDTH-1) / (X_BINS -1 ) * scaling_factor
        angle = (theta+ 0.5) * 2 * np.pi / (ANGLE_BIN_COUNT - 1 )
        length = length + 0.5
        length = length * scaling_factor
        point1 = (int(x), int(y))
        point2 = (int(x + length * np.cos(angle)), int(y + length * np.sin(angle)))

        final_image = np.concatenate((image2, image1), axis=0)

        cv2.arrowedLine(final_image, point1, point2, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(base_path,str(index)+'.png'), final_image * 255)
        index += 1
        #cv2.imshow('result', final_image)
        #cv2.waitKey(0)


