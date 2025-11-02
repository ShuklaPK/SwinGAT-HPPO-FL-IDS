#!/usr/bin/env bash
set -e
python data_pipeline.py --config configs/data_sdniot.yaml --build_cache
