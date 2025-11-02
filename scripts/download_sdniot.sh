#!/usr/bin/env bash
set -e
cat <<'EOT'
This dataset is hosted on Kaggle: https://www.kaggle.com/datasets/ymirsky/network-traffic-classification-sdn-iot (or similar).
To download:

1) Install Kaggle CLI: pip install kaggle
2) Create API token from Kaggle account; place kaggle.json in ~/.kaggle/ with 600 perms.
3) Example (adjust dataset slug):
   kaggle datasets download -d <owner>/<slug> -p datasets/sdniot --unzip

Then update configs/data_sdniot.yaml:data_path accordingly.
EOT
