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

relative_path = 'datasets/resnet_features_subset_office31/'
# relative_path = 'datasets/office-31_10_class_subset/'

all_npys = glob.glob(os.path.dirname(os.path.realpath(__file__))+'/'+relative_path+"**/*.npy" , recursive=True)

num_plot_classes = 31
all_features = np.zeros((num_plot_classes*3*5,1000))
all_feat = {
    "amazon": np.zeros((num_plot_classes*5,1000)),
    "dslr": np.zeros((num_plot_classes*5,1000)),
    "webcam": np.zeros((num_plot_classes*5,1000)),
}
domain_names =[]
class_names = []
counter = 0
for i, npy_loc in enumerate(all_npys):
    unique_labels, unique_counts = np.unique(class_names, return_counts=True)
    domain = npy_loc.split('/')[-3]
    class_name = npy_loc.split('/')[-2]

    if len(np.unique(class_names)) < num_plot_classes or class_name in class_names:
        all_features[counter] = np.load(npy_loc)
        counter += 1
        domain_names.append(domain)
        class_names.append(class_name)