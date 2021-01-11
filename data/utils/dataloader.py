import io
import os

import apache_beam as beam
import numpy as np
import pandas as pd
import tensorflow as tf
from config import Config, MLConfig
from data.utils.gaussian import ObjectCoords, gaussian_radius
from PIL import Image


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

class DataLoader:
    LICENSE_PLATE_LABEL = "license_plate"
    
    def __init__(self) -> None:
        super().__init__()
        self.labels_txt = Config.LABELS_TXT

    def _get_dutch_license_dateframe(self):
        #skip header
        image_info = tf.io.gfile.GFile(self.labels_txt).readlines()[1:]

        data = []
        
        for sample in image_info:
            image_id, _, _, *bboxes = sample.split(",")
            image_id, _ = os.path.splitext(sample)
            bboxes = np.array(bboxes).reshape(-1, 5) # label, xmin, ymin, xmax, ymax
            for bbox in bboxes:
                label, xmin, ymin, xmax, ymax = bbox
                data.append([image_id, int(xmin), int(ymin), int(xmax), int(ymax), label])
        object_detection_dataframe = pd.DataFrame(data, columns=["image_id", "xmin", "ymin", "xmax", "ymax", "label"])
        return object_detection_dataframe

    def _get_license_dataframe(self):
        image_info = tf.io.gfile.GFile(self.labels_txt).readlines()

        data = []
        
        for sample in image_info:
            image_id, _ = os.path.splitext(sample)
            # note dash
            _, _, bbox, *_ = sample.split("-")

            # note underscore
            left_up, right_bottom = bbox.split("_")
            xmin, ymin = left_up.split("&")
            xmax, ymax = right_bottom.split("&")
            label = self.LICENSE_PLATE_LABEL
            data.append([image_id, int(xmin), int(ymin), int(xmax), int(ymax), label])
        object_detection_dataframe = pd.DataFrame(data, columns=["image_id", "xmin", "ymin", "xmax", "ymax", "label"])
        return object_detection_dataframe
    
    def _get_dataframe(self):
        image_info = tf.io.gfile.GFile(self.labels_txt).readlines()

        data = []
        for sample in image_info:
            sample_info = sample.split()
            _, filename = os.path.split(sample_info[0])
            image_id, _ = os.path.splitext(filename)
            for bbox in sample_info[1:]:
                xmin, ymin, xmax, ymax, label = [int(value) for value in bbox.split(",")]
                data.append([image_id, xmin, ymin, xmax, ymax, label])

        object_detection_dataframe = pd.DataFrame(data, columns=["image_id", "xmin", "ymin", "xmax", "ymax", "label"])
        return object_detection_dataframe
    
    @staticmethod
    def _get_labelmap(labelmap_txt):
        with tf.io.gfile.GFile(labelmap_txt) as fid:
            labels = fid.read().split("\n")

        labelmap = {
            classname: index + 1  # reserve 0 for background
            for index, classname in enumerate(labels)
        }
        return labelmap

    @staticmethod
    def _get_groups(dataframe, group_column="image_id"):
        dataframe_groups = [
            dataframe[dataframe[group_column] == id]
            for id in dataframe[group_column].unique()
        ]
        return dataframe_groups


class TargetConstructorLicensePlate:
    def __init__(self) -> None:
        self.downsampling_rate = MLConfig.DOWNSAMPLING_RATE
    
    def generate_target(self, raw_labels, labelmap):
        heatmap_target = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_CLASSES))
        offset_target = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_OFFSETS))
        regression_target = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_REGRESSIONS))
        indices = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT))

        classes_text = []
        classes = []
        
        for _, row in raw_labels.iterrows():
            object_coord = ObjectCoords(row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            label_text = str(row["label"])
            label = labelmap[str(row["label"])]
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
            heatmap = gaussian_radius(MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, xcenter_int, ycenter_int, width, height)
            
            heatmap_target[:, :, label] = np.maximum(heatmap_target[:, :, label], heatmap)
            
            offset_target[xcenter_int, ycenter_int, 0] =  xcenter - xcenter_int
            offset_target[xcenter_int, ycenter_int, 1] =  ycenter - ycenter_int

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

class TargetConstructor:
    def __init__(self) -> None:
        self.downsampling_rate = MLConfig.DOWNSAMPLING_RATE
    
    def generate_target(self, raw_labels, labelmap):
        heatmap_target = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_CLASSES))
        offset_target = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_OFFSETS))
        regression_target = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, MLConfig.N_REGRESSIONS))
        indices = np.zeros((MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT))

        classes_text = []
        classes = []
        
        for _, row in raw_labels.iterrows():
            object_coord = ObjectCoords(row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            label_text = str(row["label"])
            label = labelmap[str(row["label"])]
            classes_text.append(label_text)
            classes.append(label)

            xcenter = object_coord.xc / self.downsampling_rate
            xcenter_int = int(object_coord.xc // self.downsampling_rate)
            ycenter = object_coord.yc // self.downsampling_rate
            ycenter_int = int(object_coord.yc / self.downsampling_rate)
            width = object_coord.w // self.downsampling_rate
            height = object_coord.h // self.downsampling_rate
            heatmap = gaussian_radius(MLConfig.OUTPUT_WIDTH, MLConfig.OUTPUT_HEIGHT, xcenter_int, ycenter_int, width, height)
            
            heatmap_target[:, :, label] = np.maximum(heatmap_target[:, :, label], heatmap)
            
            offset_target[xcenter_int, ycenter_int, 0] =  xcenter - xcenter_int
            offset_target[xcenter_int, ycenter_int, 1] =  ycenter - ycenter_int

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
    def __init__(
        self,
        base_img_dir
    ) -> None:
        self.base_img_dir = base_img_dir

    def to_example(self, sample_dataframe, labelmap, extension=".jpg"):

        filename = self._get_filename(sample_dataframe)
        with tf.io.gfile.GFile(
            os.path.join(self.base_img_dir, f"{filename + extension}"), "rb"
        ) as fid:
            encoded_jpg = fid.read()

        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        image_width, image_height = image.size
        image_format = self.string_encode(extension)
        filename = self.string_encode(filename)
        width_resize_factor = image_width / MLConfig.INPUT_WIDTH
        height_resize_factor = image_height / MLConfig.INPUT_HEIGHT
        sample_dataframe[["xmin", "xmax"]] = sample_dataframe[["xmin", "xmax"]] / width_resize_factor
        sample_dataframe[["ymin", "ymax"]] = sample_dataframe[["ymin", "ymax"]] / height_resize_factor

        CenterNetTargetConstructor = TargetConstructorLicensePlate()

        (
            heatmap_target,
            offset_target,
            regression_target,
            indices,
            classes_text,
            classes,
        ) = CenterNetTargetConstructor.generate_target(
            sample_dataframe, labelmap
        )

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": self._int64_feature(image_height),
                    "image/width": self._int64_feature(image_width),
                    "image/filename": self._bytes_feature(filename),
                    "image/source_id": self._bytes_feature(filename),
                    "image/encoded": self._bytes_feature(encoded_jpg),
                    "image/format": self._bytes_feature(image_format),
                    "image/object/heatmap": self._bytes_feature(
                        self.serialize_array(heatmap_target)
                    ),
                    "image/object/offset": self._bytes_feature(
                        self.serialize_array(offset_target)
                    ),
                    "image/object/regression": self._bytes_feature(
                        self.serialize_array(regression_target)
                    ),
                    "image/object/indices": self._bytes_feature(
                        self.serialize_array(indices)
                    ),
                    "image/object/class/text": self._bytes_feature_list(
                        self.string_list_encode(classes_text)
                    ),
                    "image/object/class/label": self._int64_feature_list(classes),
                }
            )
        )
        return tf_example

    def _get_filename(self, dataframe):
        return dataframe.image_id.unique()[0]


class ConvertToExample(beam.DoFn):
    def __init__(
        self,
        image_dir,
        labelmap,
    ):
        self.image_dir = image_dir
        self.labelmap = labelmap

    def process(self, element, *args, **kwargs):
        Example = CsvExample(
            base_img_dir=self.image_dir
        )
        example_proto = Example.to_example(element, self.labelmap)
        yield example_proto.SerializeToString()
