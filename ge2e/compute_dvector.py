#!/usr/bin/env python

# wujian@2018

import os
import argparse

import torch as th
import numpy as np

from utils import load_json
from speaker_net import SpeakerNet
from libs.data_handler import ScriptReader, NumpyWriter, parse_scps
from libs.utils import get_logger

logger = get_logger(__name__)


class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid):
        # nnet config
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = SpeakerNet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))

        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        # chunk size when inference
        loader_conf = load_json(cpt_dir, "loader.json")
        self.chunk_size = sum(loader_conf["chunk_size"]) // 2
        logger.info("Using chunk size {:d}".format(self.chunk_size))
        self.nnet = nnet.to(self.device)
        # set eval model
        self.nnet.eval()

    def make_chunk(self, feats):
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
        feats = th.tensor(feats, dtype=th.float32, device=self.device)
        with th.no_grad():
            chunks = self.make_chunk(feats)  # N x C x F
            dvector = self.nnet(chunks)  # N x D
            dvector = th.mean(dvector, dim=0).detach()
            return dvector.cpu().numpy()


def run(args):
    feats_reader = ScriptReader(args.feats_scp)
    computer = NnetComputer(args.checkpoint, args.gpu)
    with NumpyWriter(args.dump_dir) as writer:
        for key, feats in feats_reader:
            logger.info("Compute dvector on utterance {}...".format(key))
            dvector = computer.compute(feats)
            writer.write(key, dvector)
    logger.info("Compute over {:d} utterances".format(len(feats_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute dvector from SpeakerNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--feats-scp",
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