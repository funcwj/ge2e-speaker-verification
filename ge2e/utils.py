# wujian@2018

import os 
import json

import os.path as op


def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not op.exists(fdir):
        os.makedirs(fdir)
    with open(op.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    """
    Load json as python object
    """
    path = op.join(fdir, name)
    if not op.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj