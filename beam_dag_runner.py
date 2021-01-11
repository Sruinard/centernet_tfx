# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define BeamDagRunner to run the pipeline using Apache Beam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ml_metadata.proto import metadata_store_pb2
import ml_metadata as mlmd

import os
from absl import logging

from pipeline import tfx_pipeline as pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import trainer_pb2
from config import TFXPipelineConfig


PIPELINE_ROOT = os.path.join(TFXPipelineConfig.OUTPUT_DIR, 'tfx_pipeline_output',
                             TFXPipelineConfig.PIPELINE_NAME)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

METADATA_PATH = "metadata.db"
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = METADATA_PATH
connection_config.sqlite.connection_mode = 3  # READWRITE_OPENCREATE


def run():
    """Define a beam pipeline."""

    BeamDagRunner().run(
        pipeline.create_pipeline(
            pipeline_name=TFXPipelineConfig.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=TFXPipelineConfig._BASE_INPUT,
            preprocessing_fn=TFXPipelineConfig._TRANSFROM_PREPROCESSING_FN,
            run_fn=TFXPipelineConfig._TASK_RUN_FN,
            train_args=trainer_pb2.TrainArgs(
                num_steps=TFXPipelineConfig.TRAIN_NUM_STEPS),
            eval_args=trainer_pb2.EvalArgs(num_steps=TFXPipelineConfig.EVAL_NUM_STEPS),
            eval_accuracy_threshold=TFXPipelineConfig.EVAL_ACCURACY_THRESHOLD,
            serving_model_dir=SERVING_MODEL_DIR,
            beam_pipeline_args=TFXPipelineConfig.DATAFLOW_BEAM_PIPELINE_ARGS,
            ai_platform_training_args=TFXPipelineConfig.GCP_AI_PLATFORM_TRAINING_ARGS,
            metadata_connection_config=connection_config

        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
