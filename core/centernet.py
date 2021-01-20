import tensorflow as tf
from config import MLConfig
from tensorflow.keras.applications import efficientnet, resnet_v2



class Decoder(tf.keras.layers.Layer):
    def __init__(self, min_confidence) -> None:
        super().__init__()
        self.min_confidence = min_confidence

    @staticmethod
    def _nms(heatmap, pool_size=3):
        hmax = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(heatmap)
        keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
        return hmax * keep

    @staticmethod
    def _topK(heatmap, k):
        B, H, W, C = 1, 128, 128, MLConfig.N_CLASSES# tf.shape(heatmap)
        heatmap = tf.reshape(heatmap, shape=(B, -1))
        topk_scores, topk_inds = tf.math.top_k(input=heatmap, k=k, sorted=True)
        topk_clses = topk_inds % C
        topk_xs = tf.cast(topk_inds // C % W, tf.float32)
        topk_ys = tf.cast(topk_inds // C // W, tf.float32)
        topk_inds = tf.cast(topk_ys + topk_xs  * tf.cast(W, tf.float32), tf.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def __call__(self, pred):
        heatmap, offset, regression = pred
        heatmap = Decoder._nms(heatmap)
        topk_scores, topk_inds, topk_clses, topk_ys, topk_xs = Decoder._topK(heatmap, MLConfig.K)

        offsets = Decoder.gather_object_data(offset, topk_inds)
        regressions = Decoder.gather_object_data(regression, topk_inds)
        xmin = tf.expand_dims(topk_xs + offsets[:,:,0] - regressions[:, :, 0] / 2.0, axis=-1)
        ymin = tf.expand_dims(topk_ys + offsets[:,:,1] - regressions[:, :, 1] / 2.0, axis=-1)
        width = tf.expand_dims(regressions[:, :, 0], axis=-1)
        height = tf.expand_dims(regressions[:, :, 1], axis=-1)
        bboxes = tf.concat([xmin, ymin, width, height], axis=2)
        return self._filtered_objects(bboxes, topk_scores, topk_clses)

    @staticmethod
    def gather_object_data(output, indices):
        B, _, _, C = 1, 128, 128, 2
        # B, _, _, C = tf.shape(output)
        output = tf.reshape(output, (B, -1, C))
        indices = tf.reshape(indices, (B, -1))
        object_data = tf.cast(tf.gather(output, indices, batch_dims=1), tf.float32)
        return object_data

    
    def _get_mask_threshold(self, scores):
        objects_boolean_mask = tf.greater_equal(scores, self.min_confidence)
        return objects_boolean_mask

    def _filtered_objects(self, bboxes, scores, classes):
        boolean_mask = self._get_mask_threshold(scores)
        bbox_boolean_mask = tf.tile(tf.expand_dims(boolean_mask, axis=-1), (1, 1, 4))
        bboxes = tf.where(bbox_boolean_mask, bboxes, tf.zeros_like(bboxes))
        scores = tf.where(boolean_mask, scores, tf.zeros_like(scores))
        classes = tf.where(boolean_mask, classes, tf.zeros_like(classes))
        return bboxes, scores, classes

def get_centernet_model():

    input_tensor = tf.keras.Input((512, 512, 3))

    if MLConfig.BACKBONE == "resnet":
        backbone = resnet_v2.ResNet50V2(include_top=False, weights="imagenet")(input_tensor)
    elif MLConfig.BACKBONE == "efficientnet":
        backbone = efficientnet.EfficientNetB4(include_top=False, weights="imagenet")(input_tensor)
    else:
        raise ValueError()

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(backbone)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D((4, 4))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    conv_head = tf.keras.layers.UpSampling2D((2, 2))(x)


    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(conv_head)
    x = tf.keras.layers.ReLU()(x)
    out_heat = tf.keras.layers.Conv2D(MLConfig.N_CLASSES, 1, padding="same", activation="sigmoid", name="heatmap_head")(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(conv_head)
    x = tf.keras.layers.ReLU()(x)
    out_off = tf.keras.layers.Conv2D(MLConfig.N_OFFSETS, 1, padding="same", name="offset_head")(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(conv_head)
    x = tf.keras.layers.ReLU()(x)
    out_reg = tf.keras.layers.Conv2D(MLConfig.N_REGRESSIONS, 1, padding="same", name="regression_head")(x)

    train_model = tf.keras.Model(inputs=[input_tensor], outputs=[out_heat, out_off, out_reg])

    # Freeze Backbone
    train_model.layers[1].trainable = False
    print(train_model.summary())
    return train_model

def get_transfer_learning_model(path_to_weights):
    transfer_learning_model = get_centernet_model()
    transfer_learning_model.load_weights(path_to_weights)
    
    # Freeze layers
    for layer in transfer_learning_model.layers[:-9]:
        layer.trainable = False

    return transfer_learning_model
    
