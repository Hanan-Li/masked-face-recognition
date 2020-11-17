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
        boxes, probs = mtcnn.detect(x)
        # print(boxes)
        # mtcnn(x, save_path=crops)
        
        crop_paths.extend(crops)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
    del mtcnn
    torch.cuda.empty_cache()

def get_embeddings(directory, batch_size, device, epochs, workers, saved_model, dataset):
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])


    embed_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    resnet = InceptionResnetV1(classify=False)
    if saved_model == "":
        resnet = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        ).to(device)
    else:
        resnet.load_state_dict(torch.load(saved_model, map_location=device), strict=False)
        resnet.to(device)
    classification = []
    embeddings = []
    crop_paths = []
    resnet.eval()
    with torch.no_grad():
        for idx, (anchor, positive, negative) in enumerate(embed_loader):
            # print(idx)
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_embeddings = resnet(anchor)
            anchor_embeddings = anchor_embeddings.to('cpu').numpy()
            positive_embeddings = resnet(positive)
            positive_embeddings = positive_embeddings.to('cpu').numpy()
            negative_embeddings = resnet(negative)
            negative_embeddings = negative_embeddings.to('cpu').numpy()
            embeddings.append([anchor_embeddings, positive_embeddings])
            embeddings.append([anchor_embeddings, negative_embeddings])
            classification.append(1)
            classification.append(0)
    return classification, embeddings

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

def get_stats(classification, embeddings, threshold):
    total_pairs = len(classification)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tp_pairs = []
    fp_pairs = []
    tn_pairs = []
    fn_pairs = []
    for idx, embed_pairs in enumerate(embeddings):
        embeddings1 = embed_pairs[0]
        embeddings2 = embed_pairs[1]
        dist = distance(embeddings1, embeddings2)
        if dist <= threshold and classification[idx] == 1:
            tp += 1
        elif dist > threshold and classification[idx] == 0:
            tn += 1
        elif dist <= threshold and classification[idx] == 0:
            fp += 1
        elif dist > threshold and classification[idx] == 1:
            fn += 1
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

def run_test(directory, batch_size, device, epochs, workers, saved_model, dataset):
    print(saved_model)
    classes, embeddings = get_embeddings(directory, batch_size, device, epochs, workers, saved_model, dataset)
    precision_list = []
    recall_list = []
    accuracy_list = []
    for threshold in thresholds:
        tp, fp, tn, fn, tp_pairs, fp_pairs, tn_pairs, fn_pairs = get_stats(classes, embeddings, threshold)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        print("threshold: ", threshold)
        print("precision: ", precision)
        print("recall: ", recall)
        print("accuracy: ", accuracy)
        print("\n")
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
    model_name = "Base Pretrained"
    if saved_model == "saved_models/lfw_mask_triplet":
        model_name = "Transfer Learning Model on LFW"
    elif saved_model == "saved_models/AFDB_subset_non_triplet":
        model_name = "Transfer Learning Model on Real World Masked"
    elif saved_model == "saved_models/AFDB_subset_triplet":
        model_name = "Transfer Learning Model on Real World Masked with Triplet"
    plt.plot(recall_list, precision_list, 'r')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve for " + model_name)
    plt.show()

    plt.plot(thresholds, accuracy_list, 'r', label="Base Pretrained Trained Model")
    plt.xlabel("thresholds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for " + model_name)
    plt.show()
    print("\n")
    print("\n")

if __name__ == "__main__":
    directory = sys.argv[1]
    batch_size = 1
    epochs = 15
    thresholds = np.arange(0.1, 1.1, 0.1)
    
    saved_model = "saved_models/lfw_mask_triplet"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    workers = 0 if os.name == 'nt' else 8
    print('Running on device: {}'.format(device))
    print(directory)
    # read_data(directory, batch_size, device, epochs, workers)
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    orig_dataset = datasets.ImageFolder(directory, transform = trans)
    dataset = TripletImageLoader(orig_dataset, train = False)
    run_test(directory, batch_size, device, epochs, workers, "", dataset)
    run_test(directory, batch_size, device, epochs, workers, "saved_models/lfw_mask_triplet", dataset)
    run_test(directory, batch_size, device, epochs, workers, "saved_models/AFDB_subset_non_triplet", dataset)
    run_test(directory, batch_size, device, epochs, workers, "saved_models/AFDB_subset_triplet", dataset)
    