#!/bin/bash
# Download the raw MNIST, Fashion-MNIST, KMNIST and CIFAR-10 files into $VISION_RAW and build the npz files that
# expt_v1/run.py reads ($VISION_DATA). Usage: VISION_RAW=$SCRATCH/exp18_raw VISION_DATA=$SCRATCH/exp18_npz bash launch/fetch_data.sh
set -euo pipefail
RAW=${VISION_RAW:?set VISION_RAW}; OUT=${VISION_DATA:?set VISION_DATA}
mkdir -p "$RAW"/{mnist,fashion,kmnist,cifar_hf}
for s in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sfL -o "$RAW/mnist/$s.gz" "https://ossci-datasets.s3.amazonaws.com/mnist/$s.gz"
  curl -sfL -o "$RAW/fashion/$s.gz" "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/$s.gz"
  curl -sfL -o "$RAW/kmnist/$s.gz" "http://codh.rois.ac.jp/kmnist/dataset/kmnist/$s.gz"
done
for f in train-00000-of-00001 test-00000-of-00001; do
  curl -sfL -o "$RAW/cifar_hf/$f.parquet" "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/$f.parquet"
done
python "$(dirname "$0")/../expt_v1/prep.py" --raw "$RAW" --out "$OUT"
# the npz hashes must equal expt_v1/data_sha256.json (checked by verify.py)
