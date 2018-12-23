# wujian@2018

import os
import random

import numpy as np
import torch as th

from libs.data_handler import ScriptReader, parse_scps


class SpeakerSampler(object):
    """
    Remember to filter speakers which utterance number lower than M
    """
    def __init__(self, data_dir):
        depends = [os.path.join(data_dir, x) for x in ["feats.scp", "spk2utt"]]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError("Missing {}!".format(depend))
        self.reader = ScriptReader(depends[0])
        self.spk2utt = parse_scps(depends[1], num_tokens=-1)

    def sample(self, N=64, M=10, chunk_size=(140, 180)):
        """
        N: number of spks
        M: number of utts
        """
        spks = random.sample(list(self.spk2utt), N)
        chunks = []
        eg = dict()
        eg["N"] = N
        eg["M"] = M
        C = random.randint(*chunk_size)
        for spk in spks:
            utt_sets = self.spk2utt[spk]
            if len(utt_sets) < M:
                raise RuntimeError(
                    "Speaker {} can not got enough utterance with M = {:d}".
                    format(spk, M))
            samp_utts = random.sample(utt_sets, M)
            for uttid in samp_utts:
                feats = self.reader[uttid]
                T, F = feats.shape
                if T >= C:
                    start = random.randint(0, T - C)
                    chunks.append(feats[start:start + C])
                else:
                    chunk = np.zeros([C, F])
                    for i in range(0, C - T):
                        chunk[i] = feats[0]
                    chunk[C - T: ] = feats
                    chunks.append(chunk)
        feats = np.stack(chunks)
        eg["feats"] = th.tensor(feats, dtype=th.float32)
        return eg


class SpeakerLoader(object):
    def __init__(self,
                 data_dir,
                 N=64,
                 M=10,
                 chunk_size=(140, 180),
                 num_steps=10000):
        self.sampler = SpeakerSampler(data_dir)
        self.N, self.M, self.C = N, M, chunk_size
        self.num_steps = num_steps

    def _sample(self):
        return self.sampler.sample(self.N, self.M, self.C)

    def __iter__(self):
        for _ in range(self.num_steps):
            yield self._sample()
