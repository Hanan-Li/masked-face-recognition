from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tripletdataset import TripletImageLoader
from PIL import Image
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff))
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)
resnet = InceptionResnetV1(
    classify=False,
    pretrained='vggface2'
).eval()

# resnet = InceptionResnetV1(classify=False)
# resnet.load_state_dict(torch.load("saved_models\mixed_mask_triplet", map_location=device), strict=False)
mtcnn = MTCNN(
    image_size=160,
    margin=14,
    selection_method='center_weighted_size'
)
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

resnet.eval()
path1 = "C:\\Users\\david\\Documents\\masked-face-recognition\\mixed_face_dataset_subset\\nini\\0_0_nini_0006.jpg"
path2 = "C:\\Users\\david\\Documents\\masked-face-recognition\\mixed_face_dataset_subset\\sunyaowei\\0_0_2.jpg"
img1 = Image.open(path1)
img2 = Image.open(path2)

img1 = img1.convert('RGB')
img1 = trans(img1).to(device)
img2 = img2.convert('RGB')
img2 = trans(img2).to(device)
img1_embedding = resnet(img1.unsqueeze(0)).detach().numpy()
img2_embedding = resnet(img2.unsqueeze(0)).detach().numpy()
if distance(img1_embedding, img2_embedding) < 0.6:
    print(distance(img1_embedding, img2_embedding))
    print("TRUE")
else:
    print(distance(img1_embedding, img2_embedding))
    print("FALSE")



# if distance(img1_embedding, img2_embedding) < 0.6:
#     print(distance(img1_embedding, img2_embedding))
#     print("TRUE")
# else:
#     print(distance(img1_embedding, img2_embedding))
#     print("FALSE")