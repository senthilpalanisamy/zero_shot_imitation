import torch

class RecurrentActionPredictor:
  def __init__(self, model_paths):
    goal_recognier_path, joint_model_path = model_paths
    #self.goal_recognier = torch.load(goal_recognier_path)
    self.joint_model = torch.load(joint_model_path)
    self.image_sequence = []
    self.current_goal_image = None

  def is_goal_reached(self):
    return False
   
  def predict_next_action(self, current_image):
    if(is_goal_reached):
      if(not image_sequence):
        return True, None
      self.current_goal_image = self.image_sequence.pop(0)
    current_image_tensor = torch.from_numpy(current_image) / 255.0
    next_action = self.joint_model(current_image_tensor, current_goal_image)
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
    for images, action in test_dataset:
        image1=  images[0,0,:,:,:]
        goal_image=  images[0,0,:,:,:]
        self.current_goal_image = goal_image
        is_finished, next_action = predict_next_action(image1)

