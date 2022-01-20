import numpy as np
import torch
from torch.utils.data import Dataset
import os
from path import Path
from segmentor.args import data_dir


def Normalize(mesh):
    mesh_norm = mesh - np.mean(mesh, axis=0)
    mesh_norm /= np.max(np.linalg.norm(mesh_norm, axis=1))
    return mesh_norm


class AirplanesData(Dataset):
    def __init__(self, root_dir, valid=False):
        self.root_dir = root_dir
        self.valid = valid
        self.files = []
        point_files = [dir[:-len(".pts")] for dir in sorted(os.listdir(root_dir / Path("points"))) if
                       dir.endswith(".pts")]
        label_files = [dir[:-len(".pts")] for dir in sorted(os.listdir(root_dir / Path("points_label"))) if
                       dir.endswith(".seg")]
        common_files = [dir for dir in point_files if dir in label_files]
        for file_name in common_files:
            self.files.append({
                "points": file_name + ".pts",
                "labels": file_name + ".seg"
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        points_path = self.root_dir / Path("points") / self.files[idx]["points"]
        labels_path = self.root_dir / Path("points_label") / self.files[idx]["labels"]
        points = np.loadtxt(points_path)
        labels = np.loadtxt(labels_path)
        points = torch.from_numpy(points)
        labels = torch.from_numpy(labels)

        return {
            "points": points,
            "labels": labels
        }


if __name__ == '__main__':
    data = AirplanesData(data_dir)
    print(data[0])
    # x, y, z = np.array(data[0][0]).T
    # pcshow(x, y, z)
