import os
import json

from torch.utils import data
import cv2
import numpy as np

from config import *
from data_generator import Dataset


X_BINS = 20
Y_BINS = 20
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240
NO_OF_CHANNELS = 3


def visualise_dataset(dataset_generator, write_path):

  if not os.path.exists(train_image_path):
    os.makedirs(train_image_path)

  index = 0
  scaling_factor = 2
  
  for index, (data, label) in enumerate(train_data_generator):
     #index += 1
     image1 = np.array(data[0,0, :,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
     #image1 = data[0,0, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
     image2 = np.array(data[0,1,:,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
     
     #image2 = data[0,1, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
     # vis2 = cv.CreateMat(240, 240, cv2.CV_8UC3)
  
     xy = int(label[0][0]) 
     y = (xy // X_BINS + 0.5) * (IMAGE_HEIGHT-1) / (Y_BINS - 1 ) * scaling_factor
     x = (xy % int(X_BINS) + 0.5) * (IMAGE_WIDTH-1) / (X_BINS -1 ) * scaling_factor
     angle = (label[0][1]+ 0.5) * 2 * np.pi / (ANGLE_BIN_COUNT - 1 )
     length = label[0][2] + 0.5
     length = length * scaling_factor
     point1 = (int(x), int(y))
     point2 = (int(x + length * np.cos(angle)), int(y + length * np.sin(angle)))
     image1 = cv2.resize(image1, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))
     image2 = cv2.resize(image2, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))
  
     cv2.arrowedLine(image2, point1, point2, (255, 255, 255), scaling_factor)
     # print(image1.shape)
     #cv2.line(image1,(0,0),(150,150),(255,255,255),15)
  
     final_image = np.concatenate((image2, image1), axis=0)
     # cv2.imshow('result', final_image)
     # cv2.imshow('image1', image1)
     # cv2.imshow('image2', image2)
     # cv2.waitKey(0)
  
     full_path = os.path.join(train_image_path, str(index)+'.jpeg')
     cv2.imwrite(full_path, final_image)

base_path = '../data/processed_poke_3'
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

partitioned_datasets['train']  = Dataset(ids['train'][:10000], labels['train'], partition='train', base_path = base_path)
partitioned_datasets['test']  = Dataset(ids['test'], labels['test'], partition='test', base_path = base_path)
partitioned_datasets['val'] = Dataset(ids['val'], labels['val'], partition='val', base_path = base_path)


params = {'batch_size': 1,
         'shuffle': True,
         'num_workers': 6}

train_data_generator = data.DataLoader(partitioned_datasets['train'], **params)
val_data_generator = data.DataLoader(partitioned_datasets['val'], **params)
test_data_generator = data.DataLoader(partitioned_datasets['test'], **params)

write_path = '../results/visualise_images'

if not os.path.exists(write_path):
    os.makedirs(write_path)

train_prefix = 'train'
val_prefix = 'val'
test_prefix = 'test'
train_image_path = os.path.join(write_path, train_prefix)
val_image_path = os.path.join(write_path, val_prefix)
test_image_path = os.path.join(write_path, test_prefix)

if not os.path.exists(train_image_path):
    os.makedirs(train_image_path)


index = 0
scaling_factor = 2
visualise_dataset(train_data_generator, train_image_path)
visualise_dataset(val_data_generator, val_image_path)
visualise_dataset(test_data_generator, test_image_path)

for index, (data, label) in enumerate(train_data_generator):
   #index += 1
   image1 = np.array(data[0,0, :,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
   #image1 = data[0,0, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
   image2 = np.array(data[0,1,:,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
   
   #image2 = data[0,1, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
   # vis2 = cv.CreateMat(240, 240, cv2.CV_8UC3)

   xy = int(label[0][0]) 
   y = (xy // X_BINS + 0.5) * (IMAGE_HEIGHT-1) / (Y_BINS - 1 ) * scaling_factor
   x = (xy % int(X_BINS) + 0.5) * (IMAGE_WIDTH-1) / (X_BINS -1 ) * scaling_factor
   angle = (label[0][1]+ 0.5) * 2 * np.pi / (ANGLE_BIN_COUNT - 1 )
   length = label[0][2] + 0.5
   length = length * scaling_factor
   point1 = (int(x), int(y))
   point2 = (int(x + length * np.cos(angle)), int(y + length * np.sin(angle)))
   image1 = cv2.resize(image1, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))
   image2 = cv2.resize(image2, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))

   cv2.arrowedLine(image2, point1, point2, (255, 255, 255), scaling_factor)
   # print(image1.shape)
   #cv2.line(image1,(0,0),(150,150),(255,255,255),15)

   final_image = np.concatenate((image2, image1), axis=0)
   # cv2.imshow('result', final_image / 255)
   # cv2.imshow('image1', image1 / 255)
   # cv2.imshow('image2', image2 / 255)
   # cv2.waitKey(0)

   full_path = os.path.join(train_image_path, str(index)+'.jpeg')
   cv2.imwrite(full_path, final_image)


for index, (data, label) in enumerate(val_data_generator):
   #index += 1
   image1 = np.array(data[0,0, :,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
   #image1 = data[0,0, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
   image2 = np.array(data[0,1,:,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
   
   #image2 = data[0,1, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
   # vis2 = cv.CreateMat(240, 240, cv2.CV_8UC3)

   xy = int(label[0][0]) 
   y = (xy // X_BINS + 0.5) * (IMAGE_HEIGHT-1) / (Y_BINS - 1 ) * scaling_factor
   x = (xy % int(X_BINS) + 0.5) * (IMAGE_WIDTH-1) / (X_BINS -1 ) * scaling_factor
   angle = (label[0][1]+ 0.5) * 2 * np.pi / (ANGLE_BIN_COUNT - 1 )
   length = label[0][2] + 0.5
   length = length * scaling_factor
   point1 = (int(x), int(y))
   point2 = (int(x + length * np.cos(angle)), int(y + length * np.sin(angle)))
   image1 = cv2.resize(image1, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))
   image2 = cv2.resize(image2, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))

   cv2.arrowedLine(image2, point1, point2, (255, 255, 255), scaling_factor)
   # print(image1.shape)
   #cv2.line(image1,(0,0),(150,150),(255,255,255),15)

   final_image = np.concatenate((image2, image1), axis=0)
   # cv2.imshow('result', final_image / 255)
   # cv2.imshow('image1', image1 / 255)
   # cv2.imshow('image2', image2 / 255)
   # cv2.waitKey(0)

   full_path = os.path.join(train_image_path, str(index)+'.jpeg')
   cv2.imwrite(full_path, final_image)


for index, (data, label) in enumerate(test_data_generator):
   #index += 1
   image1 = np.array(data[0,0, :,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
   #image1 = data[0,0, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
   image2 = np.array(data[0,1,:,:,:]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
   
   #image2 = data[0,1, :,:,:].permute(1, 2, 0).numpy().astype(np.uint8).copy()
   # vis2 = cv.CreateMat(240, 240, cv2.CV_8UC3)

   xy = int(label[0][0]) 
   y = (xy // X_BINS + 0.5) * (IMAGE_HEIGHT-1) / (Y_BINS - 1 ) * scaling_factor
   x = (xy % int(X_BINS) + 0.5) * (IMAGE_WIDTH-1) / (X_BINS -1 ) * scaling_factor
   angle = (label[0][1]+ 0.5) * 2 * np.pi / (ANGLE_BIN_COUNT - 1 )
   length = label[0][2] + 0.5
   length = length * scaling_factor
   point1 = (int(x), int(y))
   point2 = (int(x + length * np.cos(angle)), int(y + length * np.sin(angle)))
   image1 = cv2.resize(image1, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))
   image2 = cv2.resize(image2, (IMAGE_WIDTH * scaling_factor, IMAGE_HEIGHT * scaling_factor))

   cv2.arrowedLine(image2, point1, point2, (255, 255, 255), scaling_factor)
   # print(image1.shape)
   #cv2.line(image1,(0,0),(150,150),(255,255,255),15)

   final_image = np.concatenate((image2, image1), axis=0)
   # cv2.imshow('result', final_image / 255)
   # cv2.imshow('image1', image1 / 255)
   # cv2.imshow('image2', image2 / 255)
   # cv2.waitKey(0)

   full_path = os.path.join(train_image_path, str(index)+'.jpeg')
   cv2.imwrite(full_path, final_image)
