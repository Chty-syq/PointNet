import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from classifier.args import data_dir
from src import utils
from sklearn.metrics import confusion_matrix
from classifier.dataset import PointCloudData
from src.model import Classifier


def test(valid_loader):
    classifier = Classifier()
    classifier.load_state_dict(torch.load(Path("checkpoints/classifier_best.pth")))
    classifier.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs = data['points'].float()
            labels = data['category']
            outputs, m1, m2 = classifier(torch.transpose(inputs, 1, 2))
            _, pred = torch.max(outputs, 1)

            print('Batch [%4d / %4d]' % (i + 1, len(valid_loader)))

            all_preds += list(pred.numpy())
            all_labels += list(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    folders = [dir for dir in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir / dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    utils.plot_confusion_matrix(cm, list(classes.keys()), normalize=True)


if __name__ == "__main__":

    valid_ds = PointCloudData(data_dir, valid=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=64, num_workers=4)

    test(valid_loader)
