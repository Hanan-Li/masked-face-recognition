from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tripletdataset import TripletImageLoader
import numpy as np
import os

def crop_images(directory, batch_size, device, workers):
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
        mtcnn(x, save_path=crops)
        crop_paths.extend(crops)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
    del mtcnn
    torch.cuda.empty_cache()

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
    print("LOSS: ", loss)
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
    data_dir = 'mixed_face_dataset_subset'

    batch_size = 20
    epochs = 100
    workers = 0 if os.name == 'nt' else 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # crop_images(data_dir, batch_size, device, workers)

    resnet = InceptionResnetV1(
        classify=False,
        pretrained='vggface2',
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_dir, transform=trans)

    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        TripletImageLoader(dataset, train = True),
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        TripletImageLoader(dataset, train = False),
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    loss_fn = torch.nn.TripletMarginLoss()
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        PATH = 'saved_models/mixed_checkpoint_' + str(epoch)
        resnet.train()
        train_loss_epoch = train_epoch(train_loader, resnet, device, loss_fn, optimizer, scheduler)
        train_loss.append(train_loss_epoch)

        resnet.eval()
        test_loss_epoch = val_epoch(val_loader, resnet, device, loss_fn, optimizer, scheduler)
        test_loss.append(test_loss_epoch)
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': resnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_epoch,
                'test_loss': test_loss_epoch
            }, PATH)
    print(train_loss)
    print(test_loss)
    PATH = "saved_models/mixed_mask_triplet"
    torch.save(resnet.state_dict(), PATH)

  