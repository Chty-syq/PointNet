import time

import torch
from torch.utils.data import DataLoader

from classifier.dataset import PointCloudData
from segmentor.args import data_dir, bsize, lr, epochs
from segmentor.dataset import AirplanesData
from src import model


def segmentor_loss(outputs, labels, matrix1, matrix2, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bsize = outputs.shape[0]
    id1 = torch.eye(3, requires_grad=True).repeat(bsize, 1, 1)
    id2 = torch.eye(64, requires_grad=True).repeat(bsize, 1, 1)
    if outputs.is_cuda:
        id1 = id1.cuda()
        id2 = id2.cuda()
    diff1 = id1 - torch.bmm(matrix1, matrix1.transpose(1, 2))
    diff2 = id2 - torch.bmm(matrix2, matrix2.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff1) + torch.norm(diff2)) / float(bsize)


def segmentor_train(device, segmentor, train_loader, valid_loader):
    segmentor.to(device)
    segmentor.train()
    optimizer = torch.optim.Adam(segmentor.parameters(), lr=lr)

    running_loss = 0.0
    batch_start = time.time()

    print("train start")
    for epoch in range(epochs):

        for i, data in enumerate(train_loader, 0):
            inputs = data["points"].float().to(device)
            labels = data["labels"].to(device)
            optimizer.zero_grad()
            outputs, m1, m2 = segmentor(inputs)

            loss = segmentor_loss(outputs, labels, m1, m2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_end = time.time()

            if i % 10 == 9:
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f, time: %.3fs' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 10, batch_end - batch_start))
                running_loss = 0.0
                batch_start = batch_end


if __name__ == '__main__':
    train_ds = AirplanesData(data_dir, valid=False)
    valid_ds = AirplanesData(data_dir, valid=True)
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))

    train_loader = DataLoader(dataset=train_ds, batch_size=bsize, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=bsize, num_workers=4)

    for i, data in enumerate(train_loader):
        print(i)
        print(data)

    # segmentor = model.Segmentor()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Using Device: ", device)
    #
    # segmentor_train(device, segmentor, train_loader, valid_loader)
