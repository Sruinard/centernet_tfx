
import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from config import Config

from data.utils import dataloader

if __name__ == "__main__":
    data_loader = dataloader.DataLoader()

    if Config.FINE_TUNE:
        train_df = data_loader._get_dutch_license_dateframe()
    else:
        train_df = data_loader._get_license_dataframe()
    labelmap = data_loader._get_labelmap(Config.LABELMAP_TXT)
    grouped_dfs = data_loader._get_groups(train_df, "image_id")
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
                   | "create data" >> beam.Create(grouped_dfs)
                   | "create tf examples" >> beam.ParDo(dataloader.ConvertToExample(Config.IMAGE_DIR, labelmap))
                   | "write to TFRecords" >> beam.io.WriteToTFRecord(file_path_prefix=Config.LABELS_TFRECORD)
                   )
