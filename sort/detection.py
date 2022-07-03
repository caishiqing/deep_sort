import tensorflow as tf


class Detection(object):
    def __init__(self, bboxs, confidences, features):
        """Bounding box detection

        Args:
            bboxs (Union[tf.Tensor, np.ndarray]): N bounding boxes, shape = (N, 4)
            confidences (Union[tf.Tensor, np.ndarray]): shape = (N,)
            features (Union[tf.Tensor, np.ndarray]): shape = (N, D)
        """
        self.bboxs = tf.cast(bboxs, tf.float32)
        self.confidences = tf.cast(confidences, tf.float32)
        self.features = tf.cast(features, tf.float32)

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return self.bboxs[:, :2] + self.bboxs[:, 2:]

    @property
    def xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        xy = self.bboxs[:, :2] + self.bboxs[:, 2:] / 2
        a = self.bboxs[:, 2:3] / self.bboxs[:, -1:]
        return tf.concat([xy, a, self.bboxs[:, -1:]], axis=-1)
