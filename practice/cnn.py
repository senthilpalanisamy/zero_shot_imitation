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



class DataserReader():
  IMG_SIZE = 50
  CATS = "./PetImages/Cat"
  DOGS = "./PetImages/Dog"
  LABELS = {CATS:0, DOGS:1}
  training_data = []
  catcount = 0
  dogcount = 0

  def process_training_data(self):
    for label in self.LABELS:
      all_image_names = os.listdir(label)
      all_image_paths = [os.path.join(label, file_name) for file_name in all_image_names]
      for image_path in tqdm(all_image_paths):
        try:
          image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
          image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
          self.training_data.append([np.array(image), np.eye(2)[self.LABELS[label]]])
          if label == self.CATS:
            self.catcount += 1
          elif label == self.DOGS:
            self.dogcount += 1
        except Exception as e:
          pass
      np.random.shuffle(self.training_data)
      np.save("training_data.npy", self.training_data)
      print("Cats count", self.catcount)
      print("Dogs count", self.dogcount)

# cat_and_dogs = DataserReader()
# cat_and_dogs.process_training_data()
# plt.imshow(cat_and_dogs.training_data[1][0])
# plt.show()

class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 8, 5)
      self.conv2 = nn.Conv2d(8, 16, 5)
      self.conv3 = nn.Conv2d(16, 64, 5)
      self.conv4 = nn.Conv2d(64, 128, 1)
      # self.fc1 = nn.Linear(128 * 3 * 3, 100)
      # self.fc1 = nn.Linear(self._to_linear, 100)
      self.fc2 = nn.Linear(100, 10)
      self.fc3 = nn.Linear(10, 2)
      self._to_linear = None


    def forward(self, x):
      x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
      x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
      x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
      x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
      if self._to_linear is None:
          self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
          self.fc1 = nn.Linear(self._to_linear, 100)
      x = self.fc1(x.view(-1, self._to_linear))
      x = self.fc2(x)
      x = self.fc3(x)
      return F.softmax(x, dim=1)

training_data = np.load("./training_data.npy", allow_pickle=True)
X = torch.Tensor([i[0] for i in training_data]).view(-1, 1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PERCENTAGE = 0.1
val_size = int(len(X) * VAL_PERCENTAGE)
print(val_size)

TEST_PERCENTAGE = 0.1
test_size = int(len(X) * TEST_PERCENTAGE)

train_x = X[:-val_size-test_size]
train_y = y[:-val_size-test_size]
val_x = X[-val_size:]
val_y = X[-val_size:]
test_x = X[-val_size - test_size:-val_size]
test_y = X[-val_size - test_size:-val_size] 


net = Net()
print(net)
print(summary(net, (1, 50, 50)))
#nt = net.forward()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()
EPOCHS =  3
BATCH_SIZE = 8

for epoch in range(EPOCHS):
  for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
    batch_x = train_x[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
    batch_y = train_y[i:i+BATCH_SIZE]
    net.zero_grad()
    outputs = net(batch_x)
    loss = loss_function(outputs, batch_y)
    loss.backward()
    optimizer.step()

print(loss)


