from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import efficientnet
from config import MLConfig

def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
    Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)

def preprocessing_fn(inputs):
    def parse_array(array, out_type=tf.float32):
        return tf.cast(tf.io.parse_tensor(array, tf.float64), out_type)

    def map_decode_fn(images):
        images = tf.io.decode_jpeg(images)
        images = tf.image.resize(images, (MLConfig.INPUT_WIDTH, MLConfig.INPUT_HEIGHT),
                                 method="nearest")
        images.set_shape((MLConfig.INPUT_WIDTH, MLConfig.INPUT_HEIGHT, 3))
        images = tf.cast(images, tf.float32)
        if MLConfig.BACKBONE == "resnet":
            images = resnet_v2.preprocess_input(images)
        elif MLConfig.BACKBONE == "efficientnet":
            images = efficientnet.preprocess_input(images)
        else:
            raise ValueError()
        return tf.cast(images, tf.float32)

    
    images = inputs["image/encoded"]
    heatmaps = inputs["image/object/heatmap"]
    offsets = inputs["image/object/offset"]
    regressions = inputs["image/object/regression"]
    indices = inputs["image/object/indices"]


    images = tf.map_fn(map_decode_fn, images, fn_output_signature=tf.float32)
    heatmaps = tf.map_fn(parse_array, heatmaps, fn_output_signature=tf.TensorSpec(
        shape=[MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_CLASSES], dtype=tf.float32))
    offsets = tf.map_fn(parse_array, offsets, fn_output_signature=tf.TensorSpec(
        shape=[MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_OFFSETS], dtype=tf.float32))
    regressions = tf.map_fn(parse_array, regressions,
                            fn_output_signature=tf.TensorSpec(shape=[MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_REGRESSIONS], dtype=tf.float32))
    indices = tf.map_fn(parse_array, indices,
                            fn_output_signature=tf.TensorSpec(shape=[MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT], dtype=tf.float32))

    outputs = {}
    outputs["image/encoded_xf"] = images
    outputs["image/object/heatmap_xf"] = heatmaps
    outputs["image/object/offset_xf"] = offsets
    outputs["image/object/regression_xf"] = regressions
    outputs["image/object/indices_xf"] = indices
    return outputs



def preprocessing_fn_visual(inputs):
    def parse_array(array, out_type=tf.float32):
        return tf.cast(tf.io.parse_tensor(array, tf.float64), out_type)

    def map_decode_fn(images):
        images = tf.io.decode_jpeg(images)
        images = tf.image.resize(images, (MLConfig.INPUT_WIDTH, MLConfig.INPUT_HEIGHT),
                                 method="nearest")
        images.set_shape((MLConfig.INPUT_WIDTH, MLConfig.INPUT_HEIGHT, 3))
        images = tf.cast(images, tf.float32)
        if MLConfig.BACKBONE == "resnet":
            images = resnet_v2.preprocess_input(images)
        elif MLConfig.BACKBONE == "efficientnet":
            images = efficientnet.preprocess_input(images)
        else:
            raise ValueError()
        return tf.cast(images, tf.float32)

    outputs = {}

    images = inputs["image/encoded"]
    heatmaps = inputs["image/object/heatmap"]
    offsets = inputs["image/object/offset"]
    regressions = inputs["image/object/regression"]
    indices = inputs["image/object/indices"]

    images = tf.map_fn(map_decode_fn, images, fn_output_signature=tf.float32)

    outputs["image/encoded_xf"] = images
    outputs["image/object/heatmap_xf"] = heatmaps
    outputs["image/object/offset_xf"] = offsets
    outputs["image/object/regression_xf"] = regressions
    outputs["image/object/indices_xf"] = indices
    return outputs