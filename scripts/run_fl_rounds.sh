#!/usr/bin/env bash
set -e
python train.py --config configs/experiment.yaml --use_fl --fast
