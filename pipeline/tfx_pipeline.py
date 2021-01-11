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
"""TFX taxi template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text, List, Dict, Any
import tensorflow_model_analysis as tfma

from ml_metadata.proto import metadata_store_pb2
import tfx
from tfx.components import ImporterNode
from tfx.components import ImportExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component  # pylint: disable=unused-import
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2, pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input
import tensorflow_data_validation as tfdv



CURATED_SCHEMA_GEN = "gs://raw_data_layer/schema/schema.pbtxt"
STATS_OPTIONS = tfdv.StatsOptions(
    feature_whitelist=[
        'image/height',
        'image/width',
        'image/filename',
        'image/source_id',
        'image/format'
        ]
)

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
    # query: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:
    """Implements the Centernet pipeline with TFX."""
    components = []

    output_config = example_gen_pb2.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
                 example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
             ],
             partition_feature_name='image/filename'))

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = ImportExampleGen(input=external_input(data_path), output_config=output_config)
    components.append(example_gen)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'], stats_options=STATS_OPTIONS)
    components.append(statistics_gen)

    
    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)
    components.append(schema_gen)
    
    # Import manually crafted schema
    importer_node = ImporterNode(
        instance_name='import_user_schema',
        source_uri="gs://raw_data_layer/schema/",
        artifact_type=tfx.types.standard_artifacts.Schema
    )
    components.append(importer_node)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(  # pylint: disable=unused-variable
        statistics=statistics_gen.outputs['statistics'],
        schema=importer_node.outputs['result'])
    components.append(example_validator)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=importer_node.outputs['result'],
        preprocessing_fn=preprocessing_fn)
    components.append(transform)

    # update training_args per once use.
    trainer_args = {
        'run_fn': run_fn,
        'transformed_examples': transform.outputs['transformed_examples'],
        'schema': importer_node.outputs['result'],
        'transform_graph': transform.outputs['transform_graph'],
        'train_args': train_args,
        'eval_args': eval_args,
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
    }
    if ai_platform_training_args is not None:
        trainer_args.update({
            'custom_executor_spec':
                executor_spec.ExecutorClassSpec(
                    ai_platform_trainer_executor.GenericExecutor
                ),
            'custom_config': {
                ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                    ai_platform_training_args,
                }
        })
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        # Change this value to control caching of execution results. Default value
        # is `False`.
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )
