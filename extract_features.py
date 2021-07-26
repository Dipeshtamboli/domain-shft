import pdb
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
# from analyse_data import make_directory
# Load the pretrained model
model = models.resnet50(pretrained=True)

# Use the model object to select the desired layer
# layer = model._modules.get('avgpool')
layer = model._modules.get('fc')

# Set model to evaluation mode
model.eval()

# Image transforms
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_vector(image_name):
    img = Image.open(image_name)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    my_embedding = torch.zeros((1,1000))
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding.numpy()

# abc = get_vector("datasets/test_office_31_data/amazon/back_pack/frame_0001.jpg")

relative_path = 'datasets/office_31/'
all_jpgs = glob.glob(os.path.dirname(os.path.realpath(__file__))+'/'+relative_path+"**/*.jpg" , recursive=True)
for image_path in tqdm(all_jpgs):
    feat_vec = get_vector(image_path)
    path_splits = image_path.split('/')
    path_splits.remove('images')
    img_id = path_splits[-1].split('.')[0]
    path_splits[-4] = "all_features_office31"
    # pdb.set_trace()
    save_path = '/' + os.path.join(*path_splits[:-1]) +'/'
    make_directory(save_path)
    with open(save_path + img_id + ".npy", 'wb') as f:
        np.save(f, feat_vec)