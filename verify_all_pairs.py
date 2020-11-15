from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def get_paths(directory, batch_size, device, epochs, workers):
    orig_img_ds = datasets.ImageFolder(directory, transform = None)
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]
    loader = DataLoader(
        orig_img_ds,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )
    crop_paths = []
    box_probs = []

    for i, (x, b_paths) in enumerate(loader):        
        crop_paths.extend(b_paths)
    return crop_paths

def read_data(directory, batch_size, device, epochs, workers):
    orig_img_ds = datasets.ImageFolder(directory, transform = None)
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )
    # img = Image.open("C:\\Users\\david\\Documents\\masked-face-recognition\\team_pictures\\hanan_pics\\me_5.jpg")
    # boxes, probs = mtcnn.detect(img)
    # print(boxes)
    loader = DataLoader(
        orig_img_ds,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )
    crop_paths = []
    box_probs = []

    for i, (x, b_paths) in enumerate(loader):
        crops = [p.replace(directory, directory + '_cropped') for p in b_paths]
    
        # mtcnn(x, save_path=crops)
        
        crop_paths.extend(crops)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
    del mtcnn
    torch.cuda.empty_cache()

def get_embeddings(directory, batch_size, device, epochs, workers, saved_model):
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(directory, transform=trans)

    embed_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )
    resnet = InceptionResnetV1(
        classify=False,
        pretrained='vggface2'
    ).to(device)
    # resnet = InceptionResnetV1(classify=False)
    # resnet.load_state_dict(torch.load(saved_model, map_location=device), strict=False)
    # resnet.to(device)

    classes = []
    embeddings = []
    crop_paths = []
    resnet.eval()
    with torch.no_grad():
        for xb, yb in embed_loader:
            xb = xb.to(device)
            b_embeddings = resnet(xb)
            b_embeddings = b_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)
    return classes, embeddings

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

def get_stats(classes, embeddings, threshold):
    total_pairs = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tp_pairs = []
    fp_pairs = []
    tn_pairs = []
    fn_pairs = []
    for idx1, embedding1 in enumerate(embeddings):
        for idx2, embedding2 in enumerate(embeddings):
            if idx1 != idx2:
                total_pairs += 1
                dist = distance(embedding1, embedding2)
                if dist < threshold and classes[idx1] == classes[idx2]:
                    tp += 1
                    tp_pairs.append((idx1, idx2))
                elif dist < threshold and classes[idx1] != classes[idx2]:
                    fp += 1
                    fp_pairs.append((idx1, idx2))
                elif dist > threshold and classes[idx1] != classes[idx2]:
                    tn += 1
                    tn_pairs.append((idx1, idx2))
                else:
                    fn += 1
                    fn_pairs.append((idx1, idx2))
    return tp, fp, tn, fn, tp_pairs, fp_pairs, tn_pairs, fn_pairs

def sample(pair_list, crop_paths, classes, num_examples):
    fig = plt.figure()
    f, axarr = plt.subplots(num_examples,2) 

    for i in range(num_examples):
        pairs = random.choice(pair_list)
        print(crop_paths[pairs[0]])
        print(crop_paths[pairs[1]])
        im1 = mpimg.imread(crop_paths[pairs[0]])
        im2  = mpimg.imread(crop_paths[pairs[1]])
        axarr[i, 0].imshow(im1) 
        axarr[i, 1].imshow(im2) 

    plt.show()


if __name__ == "__main__":
    directory = sys.argv[1]
    batch_size = 1
    epochs = 15
    thresholds = np.arange(0.1, 2, 0.1)
    
    saved_model = "saved_models/AFDB_subset_non_triplet"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    workers = 0 if os.name == 'nt' else 8
    print('Running on device: {}'.format(device))
    print(directory)
    # read_data(directory, batch_size, device, epochs, workers)
    crop_paths = get_paths(directory, batch_size, device, epochs, workers)
    classes, embeddings = get_embeddings(directory, batch_size, device, epochs, workers, saved_model)
    
    tp, fp, tn, fn, tp_pairs, fp_pairs, tn_pairs, fn_pairs = get_stats(classes, embeddings, 0.6)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("threshold: ", 0.6)
    print("precision: ", precision)
    print("recall: ", recall)
    print("accuracy: ", accuracy)
    print("\n")

    print("True Positive")
    sample(tp_pairs, crop_paths, classes, 5)
    
    print("False Positive")
    sample(fp_pairs, crop_paths, classes, 5)

    print("True Negative")
    sample(tn_pairs, crop_paths, classes, 5)

    print("False Negative")
    sample(fn_pairs, crop_paths, classes, 5)
    # for threshold in thresholds:
    #     tp, fp, tn, fn, tp_pairs, fp_pairs, tn_pairs, fn_pairs = get_stats(classes, embeddings, threshold)
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     accuracy = (tp + tn) / (tp + fp + tn + fn)
    #     print("threshold: ", threshold)
    #     print("precision: ", precision)
    #     print("recall: ", recall)
    #     print("accuracy: ", accuracy)
    #     print("\n")