# import cv2
import json
import glob
import pdb
import pprint
import matplotlib.pyplot as plt
import pandas as pd

all_jpgs = glob.glob('datasets/office_31/**/*.jpg' , recursive=True)

img_dict = {}
for image in all_jpgs:
    img_splits = image.split('/')
    domain_name = image.split('/')[-4]
    class_name = image.split('/')[-2]
    if not domain_name in img_dict.keys():
        img_dict[domain_name] = {}
    if not class_name in img_dict[domain_name]:
        img_dict[domain_name][class_name] = 1
    
    img_dict[domain_name][class_name] += 1

pprint.pprint(img_dict)

with open("DomainWiseImageCount.txt", 'w') as file:
    json_string = json.dumps(img_dict, default=lambda o: o.__dict__, sort_keys=True, indent=2)
    file.write(json_string)

pd.DataFrame(img_dict).plot(kind='bar')
plt.savefig("img.jpg")
# pdb.set_trace()