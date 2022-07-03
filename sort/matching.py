from scipy.optimize import linear_sum_assignment
import tensorflow as tf


# cost = tf.constant([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=tf.float32)
# row_ind, col_ind = tf.py_function(linear_sum_assignment, inp=[cost], Tout=[tf.int32, tf.int32])
# print(row_ind, col_ind)


def iou(bboxs, candidates):
    bbox_tl = bboxs[:, :2]
    bbox_br = bboxs[:, :2] + bboxs[:, 2:]
    cand_tl = candidates[:, :2]
    cand_br = candidates[:, :2] + candidates[:, 2:]

    tl = tf.maximum(bbox_tl[:, tf.newaxis, :], cand_tl[tf.newaxis, :, :])
    br = tf.minimum(bbox_br[:, tf.newaxis, :], cand_br[tf.newaxis, :, :])
    wh = tf.maximum(0.0, br-tl)
    area_intersection = tf.reduce_prod(wh, axis=-1)

    area_bbox = tf.reduce_prod(bboxs[:, 2:], axis=-1)
    area_cand = tf.reduce_prod(candidates[:, 2:], axis=-1)
    area_union = area_bbox[:, tf.newaxis] + area_cand[tf.newaxis, :] - area_intersection
    return area_intersection / area_union


if __name__ == "__main__":
    bboxs = tf.constant([[0.1, 0.1, 0.2, 0.6], [0.3, 0.4, 0.3, 0.5]])
    candidates = tf.constant([[0.4, 0.4, 0.1, 0.4], [0.2, 0.2, 0.1, 0.3], [0.0, 0.1, 0.2, 0.4]])
    print(iou(bboxs, candidates))
