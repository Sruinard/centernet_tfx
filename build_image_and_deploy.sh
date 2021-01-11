#!/bin/bash

docker build -t centernet_tfx_training .
docker tag centernet_tfx_training gcr.io/essential-aleph-300415/centernet_tfx_training
docker push gcr.io/essential-aleph-300415/centernet_tfx_training

python beam_dag_runner.py
