from path import Path

from src.utils import get_project_root

data_dir = Path(get_project_root()) / Path("data/ModelNet10")
save_dir = Path(get_project_root()) / Path("checkpoints")
sample_points = 1024
bsize = 64
epochs = 15
lr = 1e-4
