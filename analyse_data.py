# import cv2
import shutil
import json
import glob
import pdb
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import os
def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
def plot_dict_graph(myDict, savename):
    pd.DataFrame(myDict).plot(kind='bar')
    plt.savefig(savename)

def dump_dict_as_json(myDict, savename):
    with open(savename, 'w') as file:
        json_string = json.dumps(myDict, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        file.write(json_string)

def read_all_jpgs(relative_path, create_test_data=False, test_save_path="test_data"):
    all_jpgs = glob.glob(os.path.dirname(os.path.realpath(__file__))+'/'+relative_path+"**/*.jpg" , recursive=True)
    img_dict = {}
    for image in all_jpgs:
        # pdb.set_trace()
        img_id = image.split('/')[-1]
        domain_name = image.split('/')[-3]
        if domain_name=="images":
            domain_name = image.split('/')[-4]
        class_name = image.split('/')[-2]
        if not domain_name in img_dict.keys():
            img_dict[domain_name] = {}
        if not class_name in img_dict[domain_name]:
            img_dict[domain_name][class_name] = 1
        else:
            img_dict[domain_name][class_name] += 1
        
        if create_test_data and img_dict[domain_name][class_name] <=5:
            make_directory(f"{test_save_path}/{domain_name}/{class_name}")
            shutil.copy(image, f"{test_save_path}/{domain_name}/{class_name}/{img_id}")

    pprint.pprint(img_dict)
    return img_dict
# img_dict = read_all_jpgs('datasets/office_31/', create_test_data=True, test_save_path="datasets/test_office_31_data")
# img_dict = read_all_jpgs('datasets/office_31/')
img_dict = read_all_jpgs('datasets/test_office_31_data/')
plot_dict_graph(img_dict, "DomainWiseImageCount.jpg")
dump_dict_as_json(img_dict, "DomainWiseImageCount.txt")
