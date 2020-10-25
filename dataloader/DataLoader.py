import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

import pathlib
import tensorflow as tf


path = "/Users/kevin/Desktop/Abroad/GaTech/master/Study/3rd_Semester/big/project/masked-face-recognition/"
data_path = path + "../data/"
face_path = data_path + "AFDB_face_dataset"
masked_face_path = data_path + "AFDB_masked_face_dataset"

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

ppath = "/Users/kevin/Desktop/Abroad/GaTech/master/Study/3rd_Semester/big/project/data/AFDB_face_dataset/aidai/0_0_aidai_0014.jpg"

load_and_preprocess_image(ppath)

img_path = ppath
label = "aidai"

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)

plt.title(label)
print()