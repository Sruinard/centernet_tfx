
import datetime

import os
from config import Config, MLConfig
from core import centernet
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tfx.components.trainer.executor import TrainerFnArgs
from tqdm import tqdm
from core.loss import Loss

_LABELS_TO_POP = ['image/filename', 'image/format', 'image/object/heatmap',
                  'image/object/offset', 'image/object/regression', 'image/object/indices', 'image/source_id']


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        [feature_spec.pop(label_key) for label_key in _LABELS_TO_POP]

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)
        heatmap, offsets, regressions = model(transformed_features)
        return {
            "heatmap": heatmap,
            "offsets": offsets,
            "regressions": regressions
        }

    return serve_tf_examples_fn

def _get_serve_function(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(images):
        """Returns the output to be used in the serving signature."""
        images = tf.map_fn(tf.image.encode_jpeg, tf.cast(images, tf.uint8), tf.string)
        transformed_features = model.tft_layer({"image/encoded": images})
        heatmap, offsets, regressions = model(transformed_features)
        return {
            "heatmap": heatmap,
            "offsets": offsets,
            "regressions": regressions
        }

    return serve_tf_examples_fn

def _get_serve_function_inference(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()


    @tf.function
    def serve_tf_examples_fn(images):
        """Returns the output to be used in the serving signature."""
        decoder = centernet.Decoder(MLConfig.MIN_CONFIDENCE)
        images = tf.map_fn(tf.image.encode_jpeg, tf.cast(images, tf.uint8), tf.string)
        transformed_features = model.tft_layer({"image/encoded": images})
        heatmap, offsets, regressions = model(transformed_features)
        bboxes, confidence, classes = decoder([heatmap, offsets, regressions])
        bboxes *= MLConfig.DOWNSAMPLING_RATE
        return {
            "image": transformed_features["image/encoded_xf"],
            "heatmap": heatmap,
            "offsets": offsets,
            "regressions": regressions,
            "bboxes": bboxes,
            "confidence": confidence,
            "classes": classes
        }

    return serve_tf_examples_fn

def apply_random_image_hue(transformed_data, hue_delta=0.4):
    transformed_data.update({"image/encoded_xf": tf.image.random_hue(transformed_data["image/encoded_xf"], hue_delta)})
    return transformed_data

def _input_fn(file_pattern, tf_transform_output, batch_size, is_training):
    def parse_example(x, feature_spec=tf_transform_output.transformed_feature_spec()):
        return tf.io.parse_single_example(x, feature_spec)

    ds = tf.data.TFRecordDataset(tf.io.gfile.glob(
        file_pattern), compression_type="GZIP")
    ds = ds.map(parse_example)
    # ds = ds.shuffle(buffer_size=400, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if is_training:
        ds = ds.map(lambda x: apply_random_image_hue(x))
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def unpack_batch(batch):
    images = batch['image/encoded_xf']
    y_true_heatmap = batch["image/object/heatmap_xf"]
    y_true_offset = batch["image/object/offset_xf"]
    y_true_regression = batch["image/object/regression_xf"]
    indices = batch["image/object/indices_xf"]

    return images, [y_true_heatmap, y_true_offset, y_true_regression, indices]


def train_step(model, optimizer, x_train, y_train):

    with tf.GradientTape() as tape:
        y_pred = model(x_train, training=True)
        loss = Loss()(y_train, y_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

    
def eval_step(model, x_test, y_test):
    y_pred = model(x_test)
    loss = Loss()(y_test, y_pred)
    return loss


def virtual_epoch(num_checkpoints, batch_size):
    """Recommended approach for epochs by ML Design Patterns book.
    """
    num_training_examples = 100000
    stop_point = 10
    total_training_examples = int(num_training_examples * stop_point) # 1 million
    steps_per_epoch = total_training_examples // (num_checkpoints * batch_size)
    return steps_per_epoch

def run_fn(fn_args: TrainerFnArgs):
    loss_not_improved_counter = 0
    batch_size = 4

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files,
                              tf_transform_output, batch_size, is_training=True)

    eval_dataset = _input_fn(fn_args.eval_files,
                             tf_transform_output, batch_size, is_training=False)

    centernet_model = centernet.get_centernet_model()
    if MLConfig.PATH_TO_TRANSFER_LEARNING_WEIGHTS:
        centernet_model = centernet.get_transfer_learning_model(MLConfig.PATH_TO_TRANSFER_LEARNING_WEIGHTS)

    # path related settings
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(MLConfig.META_BUCKET, "logs", current_time)
    checkpoint_path = os.path.join(MLConfig.META_BUCKET, "checkpoints", current_time, "model_checkpoint")
    train_log_dir = os.path.join(log_dir, 'train')
    eval_log_dir = os.path.join(log_dir, 'eval')
    
    # metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    best_loss = 10000
    
    steps_per_epoch = virtual_epoch(fn_args.train_steps, batch_size)
    eval_steps = 1000 
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
                                                                decay_steps=steps_per_epoch * MLConfig.LEARNING_RATE_DECAY_EPOCHS,
                                                                decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_loss_metric = tf.keras.metrics.Mean()
    eval_loss_metric = tf.keras.metrics.Mean()

    epoch = 0

    #.repeat(fn_args.train_steps))
    for index, batch in enumerate(tqdm(train_dataset.repeat(fn_args.train_steps))):
        x_train, y_train = unpack_batch(batch)

        loss_value = train_step(model=centernet_model, optimizer=optimizer,
                                x_train=x_train, y_train=y_train)

        # Track progress
        train_loss_metric(loss_value)
        
        if index % steps_per_epoch == 0:
            
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'loss', train_loss_metric.result(), step=epoch)

            
            for batch in tqdm(eval_dataset.take(eval_steps)):
                x_test, y_test = unpack_batch(batch)
                loss_eval = eval_step(centernet_model, x_test, y_test)

                # Track progress
                eval_loss_metric(loss_eval)
                

            if eval_loss_metric.result() < best_loss:
                loss_not_improved_counter = 0
                best_loss = eval_loss_metric.result()
                centernet_model.save_weights(checkpoint_path)

            with eval_summary_writer.as_default():
                tf.summary.scalar(
                    'loss', eval_loss_metric.result(), step=epoch)

            # Reset metrics every epoch
            train_loss_metric.reset_states()
            eval_loss_metric.reset_states()
            loss_not_improved_counter +=1
            epoch += 1
            
            if loss_not_improved_counter > MLConfig.EARLY_STOPPING_N:
                break
    
    # Restore weights
    centernet_model.load_weights(checkpoint_path)

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(centernet_model,
                                tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples')),
        
        "serving_numpy":
        _get_serve_function(centernet_model,
                                tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None, None, None, 3],
                dtype=tf.float32,
                name='examples')),
        
        "serving_numpy_inference":
        _get_serve_function_inference(centernet_model,
                                tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None, None, None, 3],
                dtype=tf.float32,
                name='examples'))
    }

    centernet_model.save(os.path.join(MLConfig.META_BUCKET, "serving_models_tfx", current_time),
                         save_format='tf', signatures=signatures)
    

    centernet_model.save(fn_args.serving_model_dir,
                         save_format='tf', signatures=signatures)
