#!/usr/bin/env python

# wujian@2018

import os
import argparse

import torch as th
import numpy as np

from utils import load_json
from nnet import Nnet
from trainer import get_logger

from kaldi_python_io import Reader, ScriptReader

logger = get_logger(__name__)


class NnetComputer(object):
    """
    Compute output of networks
    """

    def __init__(self, cpt_dir, gpuid):
        # chunk size when inference
        loader_conf = load_json(cpt_dir, "loader.json")
        self.chunk_size = sum(loader_conf["chunk_size"]) // 2
        logger.info("Using chunk size {:d}".format(self.chunk_size))
        # GPU or CPU
        self.device = "cuda:{}".format(gpuid) if gpuid >= 0 else "cpu"
        # load nnet
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device)

    def _load_nnet(self, cpt_dir):
        # nnet config
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = Nnet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        nnet.eval()
        return nnet

    def _make_chunk(self, feats):
        T, F = feats.shape
        # step: half chunk
        S = self.chunk_size // 2
        N = (T - self.chunk_size) // S + 1
        if N <= 0:
            return feats
        elif N == 1:
            return feats[:self.chunk_size]
        else:
            chunks = th.zeros([N, self.chunk_size, F],
                              device=feats.device,
                              dtype=feats.dtype)
            for n in range(N):
                chunks[n] = feats[n * S:n * S + self.chunk_size]
            return chunks

    def compute(self, feats):
        feats = th.tensor(feats, device=self.device)
        with th.no_grad():
            chunks = self._make_chunk(feats)  # N x C x F
            dvector = self.nnet(chunks)  # N x D
            dvector = th.mean(dvector, dim=0).detach()
            return dvector.cpu().numpy()


def run(args):
    feats_reader = ScriptReader(args.feats)
    computer = NnetComputer(args.checkpoint, args.gpu)
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)
    for key, feats in feats_reader:
        logger.info("Compute dvector on utterance {}...".format(key))
        dvector = computer.compute(feats)
        np.save(os.path.join(args.dump_dir, key), dvector)
    logger.info("Compute over {:d} utterances".format(len(feats_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute dvector from SpeakerNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--feats",
        type=str,
        required=True,
        help="Rspecifier for input features")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU-id to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="dvector",
        help="Directory to dump dvector out")
    args = parser.parse_args()
    run(args)