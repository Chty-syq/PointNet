import os
import time

import torch
from torch.utils.data import DataLoader

from src import model
from classifier.args import *
from classifier.dataset import PointCloudData


def classifier_loss(outputs, labels, matrix1, matrix2, alpha=0.0001):
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


def classifier_accuracy(device, classifier, valid_loader):
    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs = data['points'].float().to(device)
            labels = data['category'].to(device)
            outputs, _, _ = classifier(torch.transpose(inputs, 1, 2))
            _, pred = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += (pred == labels).sum().item()

    return correct / total


def classifier_train(device, classifier, train_loader, valid_loader):
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print("Start training")

    accuracy_best = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        classifier.train()
        running_loss = 0.0
        batch_start = time.time()

        for i, data in enumerate(train_loader, 0):
            inputs = data['points'].float().to(device)
            labels = data['category'].to(device)
            optimizer.zero_grad()
            outputs, m1, m2 = classifier(torch.transpose(inputs, 1, 2))

            loss = classifier_loss(outputs, labels, m1, m2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_end = time.time()

            if i % 10 == 9:
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f, time: %.3fs' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 10, batch_end - batch_start))
                running_loss = 0.0
                batch_start = batch_end

        accuracy = classifier_accuracy(device, classifier, valid_loader)
        print('Valid accuracy: %.2f %%, Time: %.3fs' % (100.0 * accuracy, time.time() - epoch_start))

        if accuracy > accuracy_best:
            accuracy_best = accuracy
            checkpoint = save_dir / "classifier_best.pth"
            torch.save(classifier.state_dict(), checkpoint)
            print("Model save to ", checkpoint)


if __name__ == "__main__":
    train_ds = PointCloudData(data_dir, valid=False)
    valid_ds = PointCloudData(data_dir, valid=True)
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))

    train_loader = DataLoader(dataset=train_ds, batch_size=bsize, shuffle=True, num_workers=10)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=bsize, num_workers=10)

    classifier = model.Classifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)

    classifier_train(device, classifier, train_loader, valid_loader)
