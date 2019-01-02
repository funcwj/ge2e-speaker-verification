# wujian@2018

import math
import torch as th
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class TorchRNN(nn.Module):
    def __init__(self,
                 feature_dim,
                 rnn="lstm",
                 num_layers=2,
                 hidden_size=512,
                 dropout=0.0,
                 bidirectional=False):
        super(TorchRNN, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}
        if RNN not in supported_rnn:
            raise RuntimeError("unknown RNN type: {}".format(RNN))
        self.rnn = supported_rnn[RNN](
            feature_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)
        self.output_dim = hidden_size if not bidirectional else hidden_size * 2

    def forward(self, x, squeeze=False, total_length=None):
        """
        Accept tensor([N]xTxF) or PackedSequence Object
        """
        is_packed = isinstance(x, PackedSequence)
        # extend dim when inference
        if not is_packed:
            if x.dim() not in [2, 3]:
                raise RuntimeError(
                    "RNN expect input dim as 2 or 3, got {:d}".format(x.dim()))
            if x.dim() != 3:
                x = th.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        # using unpacked sequence
        # x: NxTxD
        if is_packed:
            x, _ = pad_packed_sequence(
                x, batch_first=True, total_length=total_length)
        if squeeze:
            x = th.squeeze(x)
        return x


class SpeakerNet(nn.Module):
    def __init__(self, feature_dim=40, embedding_dim=256, lstm_conf=None):
        super(SpeakerNet, self).__init__()
        self.encoder = TorchRNN(feature_dim, **lstm_conf)
        self.linear = nn.Linear(self.encoder.output_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        if x.dim() == 3:
            x = self.linear(x[:, -1, :])
        else:
            x = self.linear(x[-1, :])
        return x / th.norm(x, dim=-1, keepdim=True)


def foo_lstm():
    lstm_conf = {"num_layers": 3, "hidden_size": 738, "dropout": 0.5}
    nnet_conf = {
        "feature_dim": 40,
        "embedding_dim": 256,
        "lstm_conf": lstm_conf
    }
    nnet = SpeakerNet(**nnet_conf)
    x = th.rand(100, 40)
    x = nnet(x)
    print(x.shape)


if __name__ == "__main__":
    foo_lstm()