
import os

class Config:

    RUN_CLOUD = False
    FINE_TUNE = False
    BASE_DIR = "gs://raw_data_layer/license_plate_detection/CCPD2019" if RUN_CLOUD else "/Volumes/STEF-EXT/object_detection/kitti"
    DATASET = "CCPD2019"
    if FINE_TUNE:
        BASE_DIR = "gs://raw_data_layer/license_plate_detection/dutch_dataset" if RUN_CLOUD else "/Volumes/STEF-EXT/object_detection/kitti"
        DATASET = "dutch_dataset"

    LABELS_TFRECORD = os.path.join(BASE_DIR, "tfrecords/license_plate.tfrecord")

    RUNNER = "DataflowRunner" if RUN_CLOUD else "DirectRunner"
    PROJECT_ID = "essential-aleph-300415"
    REGION = "us-central1"
    

class MLConfig:
    DOWNSAMPLING_RATE = 4 # 512 / 4 --> 128
    
    INPUT_WIDTH = 512
    INPUT_HEIGHT = 512
    OUTPUT_WIDTH = 128
    OUTPUT_HEIGHT = 128
    
    CLASS_TO_INDEX = {
        "Car":0,
        "Van":1,
        "Truck":2,
        "Pedestrian":3,
        "Person_sitting":4,
        "Cyclist":5,
        "Tram":6,
        "Misc":7,
    }
    INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}

    N_CLASSES = len(CLASS_TO_INDEX) # background + license_plate
    N_OFFSETS = 2
    N_REGRESSIONS = 2

    K = 100
    MIN_CONFIDENCE = 0.3

    LEARNING_RATE_DECAY_EPOCHS = 10

    # efficientnet or resnet
    BACKBONE = "efficientnet"
    PATH_TO_TRANSFER_LEARNING_WEIGHTS = "gs://serving_models_tfx/centernet/efficientnet/checkpoints/20210104-201456/model_checkpoint" #"gs://serving_models_tfx/centernet/efficientnet/checkpoints/20210104-201456/model_checkpoint"

    META_BUCKET = f"gs://serving_models_tfx/centernet/{BACKBONE}/{Config.DATASET}/"
    EARLY_STOPPING_N = 10

class TFXPipelineConfig:
    
    _BASE_INPUT = os.path.join(Config.BASE_DIR, "tfrecords")

    PIPELINE_NAME = "tfx_centernet_license_plate_detection"
    OUTPUT_DIR = "gs://raw_data_layer/artifacts"
    
    _TRANSFROM_PREPROCESSING_FN = "trainer.transform_task.preprocessing_fn"
    _TASK_RUN_FN = "trainer.task.run_fn"

    _BUCKET_NAME = "gs://raw_data_layer"

    TRAIN_NUM_STEPS = 1001
    EVAL_NUM_STEPS = 1
    EVAL_ACCURACY_THRESHOLD = 0.5

    GCP_AI_PLATFORM_TRAINING_ARGS = {
        'project': Config.PROJECT_ID,
        'region': Config.REGION,
        'scaleTier': 'BASIC_GPU',
        'masterConfig': {
            'imageUri': 'gcr.io/' + Config.PROJECT_ID + '/centernet_tfx_training',
        },
        'jobDir': 'gs://raw_data_layer/training_job/',
        'pythonVersion': '3.7'
    }

    DATAFLOW_BEAM_PIPELINE_ARGS = [
        '--project=' + Config.PROJECT_ID,
        '--runner=DataflowRunner',
        '--temp_location=' + os.path.join('gs://', _BUCKET_NAME, 'tmp'),
        '--region=' + Config.REGION,
        '--disk_size_gb=50',
        '--experiments=shuffle_mode=auto',
        '--machine_type=n1-standard-8',
        '--max_num_workers=8'
    ]
