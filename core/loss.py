import tensorflow as tf
from config import Config

class FocalLoss:
    def __call__(self, y_true, y_pred, *args, **kwargs):
        return FocalLoss.__neg_loss(y_true, y_pred)

    @staticmethod
    def __neg_loss(y_true_heat, y_pred_heat):
        pos_idx = tf.cast(tf.math.equal(y_true_heat, 1), dtype=tf.float32)
        neg_idx = tf.cast(tf.math.less(y_true_heat, 1), dtype=tf.float32)
        neg_weights = tf.math.pow(1 - y_true_heat, 4)

        loss = 0
        num_pos = tf.math.reduce_sum(pos_idx)
        pos_loss = tf.math.log(y_pred_heat) * tf.math.pow(1 - y_pred_heat, 2) * pos_idx
        pos_loss = tf.math.reduce_sum(pos_loss)
        neg_loss = tf.math.log(1 - y_pred_heat) * tf.math.pow(y_pred_heat, 2) * neg_weights * neg_idx
        neg_loss = tf.math.reduce_sum(neg_loss)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

class RegL1Loss:

    def __call__(self, y_true, y_pred, indices, *args, **kwargs):
        # [1, 1, 1, 2] = batch_index, width index, height index, n_regressions or n_offsets
        mask = tf.tile(tf.expand_dims(indices, axis=-1), tf.constant([1, 1, 1, 2], dtype=tf.int32))
        mask_flat = tf.reshape(mask, (-1))
        y_pred = tf.reshape(y_pred, (-1))
        y_true = tf.reshape(y_true, (-1))
        loss = tf.math.reduce_sum(tf.abs(y_true * mask_flat - y_pred * mask_flat))
        reg_loss = loss / (tf.math.reduce_sum(mask_flat) + 1e-4)
        return reg_loss

class Loss:
    LB_CLIP_VALUE = 1e-4
    UB_CLIP_VALUE = 1.0 - 1e-4

    def __init__(self) -> None:
        super().__init__()
        self.heatmap_loss = FocalLoss()
        self.offset_loss = RegL1Loss()
        self.regression_loss = RegL1Loss()

    def __call__(self, y_true, y_pred):
        y_true_heatmap, y_true_offset, y_true_regression, indices = y_true
        y_pred_heatmap, y_pred_offset, y_pred_regression = y_pred
        y_pred_heatmap = tf.clip_by_value(y_pred_heatmap, self.LB_CLIP_VALUE, self.UB_CLIP_VALUE)
        heatmap_loss = self.heatmap_loss(y_true_heatmap, y_pred_heatmap)
        offset_loss = self.offset_loss(y_true_offset, y_pred_offset, indices)
        regression_loss = self.regression_loss(y_true_regression, y_pred_regression, indices)
        loss = heatmap_loss + 1.0 * offset_loss + 0.1 * regression_loss
        return loss

