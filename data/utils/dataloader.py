import io
import os

import apache_beam as beam
import numpy as np
import pandas as pd
import tensorflow as tf
from config import MLConfig
from data.utils.gaussian import ObjectCoords, gaussian_radius
from PIL import Image
from tfx_bsl.coders.example_coder import ExampleToNumpyDict

class TFRecordExampleBase:
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature_list(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature_list(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature_list(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def string_encode(self, s, encoder="utf-8"):
        return s.encode(encoder)

    def string_list_encode(self, s, encoder="utf-8"):
        s = [item.encode(encoder) for item in s]
        return s

    def string_decode(self, s, decoder="utf-8"):
        return s.decode(decoder)

    def string_list_decoder(self, s, decoder="utf-8"):
        s = [item.decode(decoder) for item in s]
        return s

    def serialize_array(self, array):
        return tf.io.serialize_tensor(array)

    def parse_array(self, array, out_type=tf.float64):
        return tf.io.parse_tensor(array, out_type)


class TFRecordParser(TFRecordExampleBase):

    _array_columns = [('image/object/regression', tf.float64),
                      ('image/object/heatmap', tf.float64),
                      ('image/object/offset', tf.float64),
                      ('image/object/indices', tf.float64)]

    def parse_single_example(self, example):
        feature_description = {
            'image/height':
            tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image/width':
            tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image/filename':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/source_id':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/encoded':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/format':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/heatmap':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/offset':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/regression':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/indices':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/class/text':
            tf.io.VarLenFeature(tf.string),
            'image/object/class/label':
            tf.io.VarLenFeature(tf.int64)
        }

        example = tf.io.parse_single_example(example, feature_description)

        for array_column, dtype in self._array_columns:
            example[array_column] = self.parse_array(example[array_column],
                                                     dtype)

        return example


class ObjectTargetConstructor:
    def __init__(self) -> None:
        self.downsampling_rate = MLConfig.DOWNSAMPLING_RATE

    def generate_target(self, raw_labels):
        heatmap_target = np.zeros(
            (MLConfig.OUTPUT_WIDTH,
             MLConfig.OUTPUT_HEIGHT,
             MLConfig.N_CLASSES))
        offset_target = np.zeros(
            (MLConfig.OUTPUT_WIDTH,
             MLConfig.OUTPUT_HEIGHT,
             MLConfig.N_OFFSETS))
        regression_target = np.zeros(
            (MLConfig.OUTPUT_WIDTH,
             MLConfig.OUTPUT_HEIGHT,
             MLConfig.N_REGRESSIONS))
        indices = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT))

        classes_text = []
        classes = []

        for _, row in raw_labels.iterrows():
            object_coord = ObjectCoords(
                row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            label_text = MLConfig.INDEX_TO_CLASS[row["label"]]
            label = row["label"]
            classes_text.append(label_text)
            classes.append(label)

            xcenter = object_coord.xc / self.downsampling_rate
            xcenter_int = int(object_coord.xc // self.downsampling_rate)
            ycenter = object_coord.yc // self.downsampling_rate
            ycenter_int = int(object_coord.yc / self.downsampling_rate)
            width = object_coord.w // self.downsampling_rate
            height = object_coord.h // self.downsampling_rate
            if width == 0 or height == 0:
                continue
            heatmap = gaussian_radius(
                MLConfig.OUTPUT_WIDTH,
                MLConfig.OUTPUT_HEIGHT,
                xcenter_int,
                ycenter_int,
                width,
                height)

            heatmap_target[:, :, label] = np.maximum(
                heatmap_target[:, :, label], heatmap)

            offset_target[xcenter_int, ycenter_int, 0] = xcenter - xcenter_int
            offset_target[xcenter_int, ycenter_int, 1] = ycenter - ycenter_int

            regression_target[xcenter_int, ycenter_int, 0] = width
            regression_target[xcenter_int, ycenter_int, 1] = height

            indices[xcenter_int, ycenter_int] = 1

        return (
            heatmap_target,
            offset_target,
            regression_target,
            indices,
            classes_text,
            classes,
        )


class CsvExample(TFRecordExampleBase):

    @staticmethod
    def to_example(example):

        filename = example["image/file_name"].numpy().decode('utf-8')
        image = Image.fromarray(example["image"].numpy())
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        encoded_jpg = buf.getvalue()
        image_width, image_height = image.size

        filename = example["image/file_name"].numpy().decode('utf-8')
        data = []
        for box, int_label in zip(
                example["objects"]["bbox"], example["objects"]["type"]):
            ymax, xmin, ymin, xmax = box.numpy()
            ymin = int(ymin * image_height - image_height) * -1
            ymax = int(ymax * image_height - image_height) * -1
            xmin = int(xmin * image_width)
            xmax = int(xmax * image_width)
            label = int_label.numpy()
            data.append([filename, xmin, ymin, xmax, ymax, label])

        sample_dataframe = pd.DataFrame(
            data,
            columns=[
                "image_id",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "label"])

        width_resize_factor = image_width / MLConfig.INPUT_WIDTH
        height_resize_factor = image_height / MLConfig.INPUT_HEIGHT
        sample_dataframe[["xmin", "xmax"]] = sample_dataframe[[
            "xmin", "xmax"]] / width_resize_factor
        sample_dataframe[["ymin", "ymax"]] = sample_dataframe[[
            "ymin", "ymax"]] / height_resize_factor

        CenterNetTargetConstructor = ObjectTargetConstructor()

        (
            heatmap_target,
            offset_target,
            regression_target,
            indices,
            classes_text,
            classes,
        ) = CenterNetTargetConstructor.generate_target(
            sample_dataframe
        )

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": TFRecordExampleBase()._int64_feature(image_height),
                    "image/width": TFRecordExampleBase()._int64_feature(image_width),
                    "image/filename": TFRecordExampleBase()._bytes_feature(filename.encode('utf-8')),
                    "image/source_id": TFRecordExampleBase()._bytes_feature(filename.encode('utf-8')),
                    "image/encoded": TFRecordExampleBase()._bytes_feature(encoded_jpg),
                    "image/format": TFRecordExampleBase()._bytes_feature("jpg".encode("utf-8")),
                    "image/object/heatmap": TFRecordExampleBase()._bytes_feature(
                        TFRecordExampleBase().serialize_array(heatmap_target)
                    ),
                    "image/object/offset": TFRecordExampleBase()._bytes_feature(
                        TFRecordExampleBase().serialize_array(offset_target)
                    ),
                    "image/object/regression": TFRecordExampleBase()._bytes_feature(
                        TFRecordExampleBase().serialize_array(regression_target)
                    ),
                    "image/object/indices": TFRecordExampleBase()._bytes_feature(
                        TFRecordExampleBase().serialize_array(indices)
                    ),
                    "image/object/class/text": TFRecordExampleBase()._bytes_feature_list(
                        TFRecordExampleBase().string_list_encode(classes_text)
                    ),
                    "image/object/class/label": TFRecordExampleBase()._int64_feature_list(classes),
                }
            )
        )
        return tf_example


class ParseExample(beam.DoFn):
    def process(self, element, *args, **kwargs):
        feature_dict = ExampleToNumpyDict(element)
        parsed_example = {
            "image": tf.io.decode_image(feature_dict["image"][0]),
            "image/file_name": tf.convert_to_tensor(feature_dict["image/file_name"][0]),
            "objects": {
                    "bbox": tf.convert_to_tensor(feature_dict["objects/bbox"].reshape(-1, 4)),
                    "type": tf.convert_to_tensor(feature_dict["objects/type"].reshape(-1))
                }
            }
        yield parsed_example

class ConvertToExample(beam.DoFn):
    def process(self, element, *args, **kwargs):
        example_proto = CsvExample.to_example(element)
        yield example_proto.SerializeToString()
