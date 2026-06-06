try:
    from .trainer import Trainer, find_all_linear_names
except Exception:
    Trainer = None

    def find_all_linear_names(*args, **kwargs):
        raise RuntimeError("InstructSAM trainer dependencies are unavailable in this environment.")

try:
    from .dataset import SFTDataset, DataCollator
except Exception:
    SFTDataset = None
    DataCollator = None

from .utils import *
