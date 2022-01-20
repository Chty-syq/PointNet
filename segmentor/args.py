from path import Path
from src.utils import get_project_root

data_dir = Path(get_project_root()) / Path("data/Airplanes")
save_dir = Path(get_project_root()) / Path("checkpoints")
bsize = 64
epochs = 15
lr = 1e-4
