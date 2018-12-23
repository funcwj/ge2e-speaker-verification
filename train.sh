#!/usr/bin/env bash
# wujian@2018

set -eu

train_dir=data/train
dev_dir=data/dev
cpt_dir=exp/ge2e
epochs=50

echo "$0 $@"

. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <gpuid> <cpt-id>" && exit 1

./ge2e/train_ge2e.py \
  --M 10 \
  --N 64 \
  --gpu $1 \
  --epochs $epochs \
  --train $train_dir \
  --dev $dev_dir \
  --checkpoint $cpt_dir/$2 \
  > $2.train.log 2>&1 