## Speaker Verification with GE2E Loss

Pytorch implement of "Generalized End-to-End Loss for Speaker Verification"

### Data Processing

1. Vad (recommend [py-webrtcvad](https://github.com/wiseman/py-webrtcvad))
2. Features (recommend [librosa](https://github.com/librosa/librosa))
3. Prepare data as `data/{train,dev}/{feats.scp,spk2utt}`

### Usage

see [train.sh](train.sh) and [compute_dvector.py](compute_dvector.py)

### Reference

Wan L, Wang Q, Papir A, et al. Generalized end-to-end loss for speaker verification[C]//2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018: 4879-4883.