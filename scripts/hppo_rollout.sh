#!/usr/bin/env bash
set -e
python mitigation/hppo_agent.py --config configs/hppo.yaml --rollout_steps 128
