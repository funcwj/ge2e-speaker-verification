# wujian@2018

import random
import os.path as op

import numpy as np
import torch as th

from kaldi_python_io import Reader, ScriptReader


class SpeakerSampler(object):
    """
    Remember to filter speakers which utterance number lower than M
    """

    def __init__(self, data_dir):
        depends = [op.join(data_dir, x) for x in ["feats.scp", "spk2utt"]]
        for depend in depends:
            if not op.exists(depend):
                raise RuntimeError("Missing {}!".format(depend))
        self.reader = ScriptReader(depends[0])
        self.spk2utt = Reader(depends[1], num_tokens=-1)

    def sample(self, N=64, M=10, chunk_size=(140, 180)):
        """
        N: number of spks
        M: number of utts
        """
        spks = random.sample(self.spk2utt.index_keys, N)
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
                utt = self.reader[uttid]
                pad = C - utt.shape[0]
                if pad < 0:
                    start = random.randint(0, -pad)
                    chunks.append(utt[start:start + C])
                else:
                    chunk = np.pad(utt, ((pad, 0), (0, 0)), "edge")
                    chunks.append(chunk)
        eg["feats"] = th.from_numpy(np.stack(chunks))
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
