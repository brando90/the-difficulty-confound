#!/bin/bash
# Download the raw files of Experiment 22 into $EXP22_RAW (hashes are checked by expt_v1/prep.py).
# Usage: EXP22_RAW=$HOME/exp22/raw bash launch/fetch_data.sh
set -euo pipefail
RAW=${EXP22_RAW:?set EXP22_RAW}
mkdir -p "$RAW"
get() { [ -s "$RAW/$1" ] && return 0; curl -sfL --retry 3 -o "$RAW/$1.part" "$2"; mv "$RAW/$1.part" "$RAW/$1"; }
get cifar100_train.parquet https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet
get cifar100_test.parquet https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet
get cifar10_train.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
get cifar10_test.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
get 20news-bydate.tar.gz https://ndownloader.figshare.com/files/5975967
ls -la "$RAW"
