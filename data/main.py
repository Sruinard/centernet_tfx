
import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from config import Config

from data.utils import dataloader

if __name__ == "__main__":
    options = PipelineOptions(
        runner=Config.RUNNER,
        project=Config.PROJECT_ID,
        job_name='generate-tfrecords' +
        datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S"),
        temp_location='gs://raw_data_layer/temp',
        region='us-central1',
        setup_file="./setup.py"
    )

    with beam.Pipeline(options=options) as pipeline:
        content = (pipeline
                   | "create data" >> beam.io.ReadFromTFRecord(file_pattern="/Volumes/STEF-EXT/object_detection/kitti/kitti/3.2.0/kitti-train.tfrecord*")
                   | "parse tfds examples" >> beam.ParDo(dataloader.ParseExample())
                   | "create tf examples" >> beam.ParDo(dataloader.ConvertToExample())
                   | "write to TFRecords" >> beam.io.WriteToTFRecord(file_path_prefix=Config.LABELS_TFRECORD)
                   )
