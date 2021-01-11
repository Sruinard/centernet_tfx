import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_data_validation as tfdv

class SchemaGenerator:
    @staticmethod
    def _get_schema():
        feature_spec = SchemaGenerator._get_feature_spec()
        schema = schema_utils.schema_from_feature_spec(feature_spec)
        return schema
    
    def write_schema(destination):
        schema = SchemaGenerator._get_schema()
        tfdv.write_schema_text(schema, destination)

    @staticmethod
    def _get_feature_spec():
        feature_spec = {
                    'image/height':
                    tf.io.FixedLenFeature([], tf.int64),
                    'image/width':
                    tf.io.FixedLenFeature([], tf.int64),
                    'image/filename':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/source_id':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/encoded':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/format':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/object/heatmap':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/object/offset':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/object/regression':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/object/indices':
                    tf.io.FixedLenFeature([], tf.string),
                    'image/object/class/text':
                    tf.io.VarLenFeature(tf.string),
                    'image/object/class/label':
                    tf.io.VarLenFeature(tf.int64)
                }
        return feature_spec
    
if __name__ == "__main__":
    SchemaGenerator.write_schema("gs://raw_data_layer/schema/schema.pbtxt")