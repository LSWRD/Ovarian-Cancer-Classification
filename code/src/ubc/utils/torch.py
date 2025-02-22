import numpy as np
import os, random
import torch
import json
import pickle


def seed_everything_now(seed):
    """
    Seeds basic parameters for reproducibility of results.
    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEEDS = [42, 50, 2023, 2024, 20, 1973]
FOLDS = 4

CLASSES_MAP = {
    'HGSC': 0,
    'LGSC': 1,
    'EC': 2,
    'CC': 3,
    'MC': 4,
    'Other': 5,
}
CLASSES_INVERSE_MAP = {v:k for k,v in CLASSES_MAP.items()}

TUMOR_RATIO_MAP = {
    # CC
    1952: 100.,
    4877: 100.,
    6359: 100.,
    13526: 100.,
    15470: 100.,
    15486: 100.,
    22489: 100.,
    25923: 100.,
    25928: 100.,
    26219: 100.,
    26533: 100.,
    30369: 100.,
    38349: 100.,
    # EC
    286: 100.,
    4608: 100.,
    6281: 100.,
    7490: 100.,
    15912: 100.,
    16064: 100.,
    17487: 100.,
    18138: 100.,
    27851: 100.,
    28736: 100.,
    32042: 100.,
    38118: 100.,
    # HGSC
    1295: 100.,
    4963: 100.,
    5251: 100.,
    6140: 100.,
    6175: 100.,
    8713: 100.,
    14127: 100.,
    17174: 100.,
    18547: 100.,
    19157: 100.,
    20316: 100.,
    24507: 100.,
    # LGSC
    66: 100.,
    6898: 100.,
    21260: 100.,
    33708: 100.,
    33976: 100.,
    38585: 100.,
    38849: 100.,
    39466: 100.,
    52612: 100.,
    65022: 100.,
    # MC
    2227: 100.,
    4797: 100.,
    23523: 100.,
    28562: 100.,
    30868: 100.,
    31297: 100.,
    35792: 100.,
    36678: 100.,
    38019: 100.,
    42260: 100.,
    51893: 100.,
    56993: 100.,
}


def save_dict(tmp_dict, filename):
    pickle.dump(tmp_dict, open(filename, 'wb'))


def load_dict(filename):
    return pickle.load(open(filename, 'rb'))


class Config:
    """
    Placeholder to load a config from a saved json
    """

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def save_config(config, path):
    """
    Saves a config as a json
    Args:
        config (Config): Config.
        path (str): Path to save at.
    """
    dic = config.__dict__.copy()
    if dic.get("__doc__") is not None:
        del dic["__doc__"]
    if dic.get("__module__") is not None:
        del dic["__module__"]
    if dic.get("__dict__") is not None:
        del dic["__dict__"]
    if dic.get("__weakref__") is not None:
        del dic["__weakref__"]

    with open(path, "w") as f:
        json.dump(dic, f)

    return dic


def load_config(config_path):
    config = Config(json.load(open(config_path, "r")))
    return config
