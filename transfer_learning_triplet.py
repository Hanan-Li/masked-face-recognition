from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tripledataset import TripletImageLoader
import numpy as np
import os

def train_epoch(tr_loader, resnet, device, loss_fn, optimizer, scheduler):
    loss = 0
    for i_batch, (anchor, pos, neg) in enumerate(tr_loader):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor_embedding = resnet(anchor)
        pos_embedding = resnet(pos)
        neg_embedding = resnet(neg)

        loss_batch = loss_fn(anchor_embedding, pos_embedding, neg_embedding)

        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
    scheduler.step()
    loss = loss / (i_batch + 1)
    return loss

def val_epoch(val_loader, resnet, device, loss_fn, optimizer, scheduler):
    loss = 0
    for i_batch, (anchor, pos, neg) in enumerate(val_loader):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor_embedding = resnet(anchor)
        pos_embedding = resnet(pos)
        neg_embedding = resnet(neg)

        loss_batch = loss_fn(anchor_embedding, pos_embedding, neg_embedding)
        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
    loss = loss / (i_batch + 1)
    return loss

if __name__ == "__main__":
    data_dir = 'mixed_face_dataset_subset/'
    dataset = datasets.ImageFolder(data_dir, transform=trans)
    TripletImageLoader(dataset, train = True)
    # batch_size = 32
    # epochs = 20
    # workers = 0 if os.name == 'nt' else 8

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('Running on device: {}'.format(device))

    # resnet = InceptionResnetV1(
    #     classify=False,
    #     pretrained='vggface2',
    # ).to(device)

    # optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    # scheduler = MultiStepLR(optimizer, [5, 10])

    # trans = transforms.Compose([
    #     np.float32,
    #     transforms.ToTensor(),
    #     fixed_image_standardization
    # ])
    # dataset = datasets.ImageFolder(data_dir, transform=trans)

    # img_inds = np.arange(len(dataset))
    # np.random.shuffle(img_inds)
    # train_inds = img_inds[:int(0.8 * len(img_inds))]
    # val_inds = img_inds[int(0.8 * len(img_inds)):]

    # train_loader = DataLoader(
    #     TripletImageLoader(dataset, train = True),
    #     num_workers=workers,
    #     batch_size=batch_size,
    #     sampler=SubsetRandomSampler(train_inds)
    # )
    # val_loader = DataLoader(
    #     TripletImageLoader(dataset, train = False),
    #     num_workers=workers,
    #     batch_size=batch_size,
    #     sampler=SubsetRandomSampler(val_inds)
    # )

    # loss_fn = torch.nn.TripletMarginLoss()
    # train_loss = []
    # test_loss = []
    # for epoch in range(epochs):
    #     print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    #     print('-' * 10)

    #     resnet.train()
    #     train_loss_epoch = train_epoch(tr_loader, resnet, device, loss_fn, optimizer, scheduler)
    #     train_loss.append(train_loss_epoch)

    #     resnet.eval()
    #     test_loss_epoch = val_epoch(val_loader, resnet, device, loss_fn, optimizer, scheduler)
    #     test_loss.append(test_loss_epoch)
    

    # PATH = "saved_models/AFDB_subset_non_triplet"
    # torch.save(resnet.state_dict(), PATH)