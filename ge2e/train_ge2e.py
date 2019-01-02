#!/usr/bin/env python

# wujian@2018

import os
import argparse
import random

from speaker_net import SpeakerNet
from trainer import GE2ETrainer
from dataset import SpeakerLoader
from utils import dump_json
from conf import nnet_conf, trainer_conf, train_steps, dev_steps, chunk_size


def run(args):
    gpuids = tuple(map(int, args.gpu.split(",")))
    nnet = SpeakerNet(**nnet_conf)

    trainer = GE2ETrainer(
        nnet, gpuid=gpuids, checkpoint=args.checkpoint, **trainer_conf)

    loader_conf = {"M": args.M, "N": args.N, "chunk_size": chunk_size}
    for conf, fname in zip([nnet_conf, trainer_conf, loader_conf],
                           ["mdl.json", "trainer.json", "loader.json"]):
        dump_json(conf, args.checkpoint, fname)

    train_loader = SpeakerLoader(
        args.train, **loader_conf, num_steps=train_steps)
    dev_loader = SpeakerLoader(args.dev, **loader_conf, num_steps=dev_steps)

    trainer.run(train_loader, dev_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to train speaker embedding model using GE2E loss, "
        "auto configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--gpu", type=str, default=0, help="Training on which GPUs")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Directory to dump models")
    parser.add_argument(
        "--N", type=int, default=64, help="Number of speakers in each batch")
    parser.add_argument(
        "--M",
        type=int,
        default=10,
        help="Number of utterances for each speaker")
    parser.add_argument(
        "--train", type=str, required=True, help="Data directory for training")
    parser.add_argument(
        "--dev", type=str, required=True, help="Data directory for evaluation")
    args = parser.parse_args()
    run(args)
