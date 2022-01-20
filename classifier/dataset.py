import numpy as np
import torch
import trimesh
from trimesh import sample
from torch.utils.data import Dataset
import os
from path import Path

from classifier.args import sample_points

def Normalize(mesh):
    mesh_norm = mesh - np.mean(mesh, axis=0)
    mesh_norm /= np.max(np.linalg.norm(mesh_norm, axis=1))
    return mesh_norm


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.valid = valid
        self.files = []

        folder = "train" if not valid else "test"
        for category in self.classes.keys():
            for file in os.listdir(root_dir / Path(category) / folder):
                if file.endswith(".off"):
                    self.files.append({
                        'path': root_dir / Path(category) / folder / file,
                        'category': category
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]["path"]
        category = self.files[idx]["category"]

        mesh = trimesh.load_mesh(path)
        mesh = trimesh.sample.sample_surface(mesh, sample_points)[0]
        mesh = Normalize(mesh)
        points = torch.from_numpy(np.array(mesh))

        return {
            'points': points,
            'category': self.classes[category]
        }
