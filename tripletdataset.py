import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TripletImageLoader(Dataset):
    def __init__(self, image_dataset, train = False):
        self.image_dataset = image_dataset
        self.classes = image_dataset.classes
        self.class_to_idx = image_dataset.class_to_idx
        self.imgs = image_dataset.imgs
        self.targets = image_dataset.targets
        self.loader = image_dataset.loader
        self.transform = image_dataset.transform

        #Preprocessing
        self.train = train
        self.class_idx = [self.class_to_idx[key] for key in self.class_to_idx]
        self.class_idx_to_idx = {idx : [] for idx in self.class_idx}
        
        for idx, val in enumerate(self.imgs):
            class_idx = val[1]
            self.class_idx_to_idx[class_idx].append(idx)
        if not self.train:
            self.test_triplets = []
            for i in range(len(self.imgs)):
                _ , class_idx = self.imgs[i]
                pos_index = i
                while pos_index == i:
                    pos_index = np.random.choice(self.class_idx_to_idx[class_idx])

                neg_label = class_idx
                while neg_label == class_idx:
                    neg_label = np.random.choice(self.class_idx)
                neg_index = np.random.choice(self.class_idx_to_idx[neg_label])
                self.test_triplets.append([pos_index, neg_index])


    def __getitem__(self, index):
        if self.train:
            anchor_path, anchor_label = self.imgs[index]
            anchor_img = self.loader(anchor_path)
            pos_index = index
            while pos_index == index:
                pos_index = np.random.choice(self.class_idx_to_idx[anchor_label])
            pos_img_path, _ = self.imgs[pos_index]
            pos_img = self.loader(pos_img_path)

            neg_label = anchor_label
            while neg_label == anchor_label:
                neg_label = np.random.choice(self.class_idx)
            neg_index = np.random.choice(self.class_idx_to_idx[neg_label])
            neg_img_path, _ = self.imgs[neg_index]
            neg_img = self.loader(neg_img_path)
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                pos_img = self.transform(pos_img)
                neg_img = self.transform(neg_img)
            return anchor_img, pos_img, neg_img
        else:
            anchor_path, anchor_label = self.imgs[index]
            anchor_img = self.loader(anchor_path)
            pos_idx = self.test_triplets[index][0]
            pos_img_path, _ = self.imgs[pos_idx]
            pos_img = self.loader(pos_img_path)

            neg_idx = self.test_triplets[index][1]
            neg_img_path, _ = self.imgs[neg_idx]
            neg_img = self.loader(neg_img_path)
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                pos_img = self.transform(pos_img)
                neg_img = self.transform(neg_img)
            return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.imgs)


class TripletLFWImageLoader(Dataset):
    def __init__(self, image_dataset, train = False):
        self.image_dataset = image_dataset
        self.classes = image_dataset.classes
        self.class_to_idx = image_dataset.class_to_idx
        self.imgs = image_dataset.imgs
        self.targets = image_dataset.targets
        self.loader = image_dataset.loader
        self.transform = image_dataset.transform

        #Preprocessing
        self.train = train
        self.class_idx = [self.class_to_idx[key] for key in self.class_to_idx]
        self.class_idx_to_idx = {idx : [] for idx in self.class_idx}
        
        for idx, val in enumerate(self.imgs):
            class_idx = val[1]
            self.class_idx_to_idx[class_idx].append(idx)
        if not self.train:
            self.test_triplets = []
            for i in range(len(self.imgs)):
                _ , class_idx = self.imgs[i]
                pos_index = i
                choose_opposite = np.random.choice([0,1])
                if choice_opposite == 1:
                    if class_idx % 2 == 0:
                        class_idx += 1
                    else:
                        class_idx -= 1
                while pos_index == i:
                    pos_index = np.random.choice(self.class_idx_to_idx[class_idx])

                neg_label = class_idx
                opp_neg_label = class_idx
                if class_idx % 2 == 0:
                    opp_neg_label += 1
                else:
                    opp_neg_label -= 1
                while neg_label == class_idx or neg_label == opp_neg_label:
                    neg_label = np.random.choice(self.class_idx)
                neg_index = np.random.choice(self.class_idx_to_idx[neg_label])
                self.test_triplets.append([pos_index, neg_index])

    def __getitem__(self, index):
        if self.train:
            anchor_path, anchor_label = self.imgs[index]
            anchor_img = self.loader(anchor_path)
            pos_index = index
            choose_opposite = np.random.choice([0,1])
            pos_label = anchor_label
            if choice_opposite == 1:
                if anchor_label % 2 == 0:
                    pos_label += 1
                else:
                    pos_label -= 1
            while pos_index == index:
                pos_index = np.random.choice(self.class_idx_to_idx[pos_label])
            pos_img_path, _ = self.imgs[pos_index]
            pos_img = self.loader(pos_img_path)

            neg_label = anchor_label
            opp_neg_label = anchor_label
            if anchor_label % 2 == 0:
                opp_neg_label = anchor_label + 1
            else:
                opp_neg_label = anchor_label - 1
            while neg_label == anchor_label or neg_label == opp_neg_label:
                neg_label = np.random.choice(self.class_idx)
            neg_index = np.random.choice(self.class_idx_to_idx[neg_label])
            neg_img_path, _ = self.imgs[neg_index]
            neg_img = self.loader(neg_img_path)
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                pos_img = self.transform(pos_img)
                neg_img = self.transform(neg_img)
            return anchor_img, pos_img, neg_img
        else:
            anchor_path, anchor_label = self.imgs[index]
            anchor_img = self.loader(anchor_path)
            pos_idx = self.test_triplets[index][0]
            pos_img_path, _ = self.imgs[pos_idx]
            pos_img = self.loader(pos_img_path)

            neg_idx = self.test_triplets[index][1]
            neg_img_path, _ = self.imgs[neg_idx]
            neg_img = self.loader(neg_img_path)
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                pos_img = self.transform(pos_img)
                neg_img = self.transform(neg_img)
            return anchor_img, pos_img, neg_img


    def __len__(self):
        return len(self.imgs)