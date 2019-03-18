#!/usr/bin/env python

# wujian@2018

import os
import pprint
import argparse
import random

from nnet import Nnet
from trainer import GE2ETrainer, get_logger
from dataset import SpeakerLoader
from utils import dump_json
from conf import nnet_conf, trainer_conf, train_dir, dev_dir

logger = get_logger(__name__)


def run(args):
    parse_str = lambda s: tuple(map(int, s.split(",")))
    nnet = Nnet(**nnet_conf)

    trainer = GE2ETrainer(
        nnet,
        gpuid=parse_str(args.gpu),
        checkpoint=args.checkpoint,
        resume=args.resume,
        **trainer_conf)

    loader_conf = {
        "M": args.M,
        "N": args.N,
        "chunk_size": parse_str(args.chunk_size)
    }
    for conf, fname in zip([nnet_conf, trainer_conf, loader_conf],
                           ["mdl.json", "trainer.json", "loader.json"]):
        dump_json(conf, args.checkpoint, fname)

    train_loader = SpeakerLoader(
        train_dir, **loader_conf, num_steps=args.train_steps)
    dev_loader = SpeakerLoader(
        dev_dir, **loader_conf, num_steps=args.dev_steps)

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
        "--train-steps",
        type=int,
        default=5000,
        help="Number of training steps in one epoch")
    parser.add_argument(
        "--dev-steps",
        type=int,
        default=800,
        help="Number of validation steps in one epoch")
    parser.add_argument(
        "--chunk-size",
        type=str,
        default="140,180",
        help="Range of chunk size, eg: 140,180")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint to resume training process")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))
    run(args)
