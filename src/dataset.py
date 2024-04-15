"""
Custom dataset

Author: DI PIAZZA Theo
"""

import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class DatasetUAV(Dataset):
    def __init__(self, path_images, path_labels):

        # folder with all images
        self.path_images = path_images

        # path of csv with coordinates
        self.path_labels = path_labels

        # dataframe with coordinates
        self.labels = pd.read_csv(self.path_labels)

        # names of tiles
        self.names = list(self.labels['name'])

    def __len__(self):
        return len(self.names)
  
    def __getitem__(self, index):

        # get name of the tile to load
        self.name = self.names[index]

        # read image
        path_image  = os.path.join(self.path_images, self.name)
        image_array = np.load(path_image)

        # apply canny filter
        edges = cv2.Canny(image_array, 50, 200)

        # convert to binary values
        edges = 1*(edges > 0).astype(np.float32)

        return edges, self.name
