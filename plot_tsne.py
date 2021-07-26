import os
import pdb
import numpy as np
from scipy import io
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time
import glob

start_time = time.time()
# load all the npy feature vectors

relative_path = 'datasets/resnet_features_complete_office31/'
# relative_path = 'datasets/resnet_features_subset_office31/'
# relative_path = 'datasets/office-31_10_class_subset/'

all_npys = glob.glob(os.path.dirname(os.path.realpath(__file__))+'/'+relative_path+"**/*.npy" , recursive=True)

num_plot_classes = 31
all_features = np.zeros((num_plot_classes*3*5,1000))
domain_names =[]
class_names = []
counter = 0
for i, npy_loc in enumerate(all_npys):
    unique_labels, unique_counts = np.unique(class_names, return_counts=True)
    domain = npy_loc.split('/')[-3]
    if not domain == "dslr":
        continue
    class_name = npy_loc.split('/')[-2]

    if len(np.unique(class_names)) < num_plot_classes or class_name in class_names:
        if counter>= len(all_features):
            # np.insert(all_features, counter, np.load(npy_loc))
            all_features = np.concatenate((all_features, np.load(npy_loc)), axis=0)
        else:
            all_features[counter] = np.load(npy_loc)
        counter += 1
        domain_names.append(domain)
        class_names.append(class_name)

tsne = TSNE(n_jobs=16)
embeddings = tsne.fit_transform(all_features)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

sns.set(rc={'figure.figsize':(11.7,8.27)})

palette = sns.color_palette("bright", num_plot_classes)
# palette = sns.color_palette("RdPu", 31)

# pdb.set_trace()
# plot = sns.scatterplot(vis_x, vis_y, hue=class_names, style = domain_names, markers=['P', 'o', 'X'], palette=palette)
plot = sns.scatterplot(vis_x, vis_y, hue=class_names, style = domain_names, markers=['o'], palette=palette)
plot.get_legend().set_title("Classes")

# handles, labels = plot.get_legend_handles_labels()
# labels[-1] = "gen"
# labels[-2] = "conv"
# plot.legend(handles, labels) 
plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
plt.tight_layout()
plt.savefig(f"TSNE_plots/office-31-dslr-{num_plot_classes}_classes_complete_dataset.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))
# pdb.set_trace()