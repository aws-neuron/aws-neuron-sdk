.. _tensorflow-yolo4:

Working with YOLO v4 using AWS Neuron SDK
=========================================

The :doc:`evaluate` notebook contains an example on how to take an open
source YOLO v4 models, and run it on AWS Inferentia.

Optimizing image pre-processing and post-processing for object detection models
-------------------------------------------------------------------------------

End-to-end object detection pipelines usually contain image
pre-post-processing operators that cannot run efficiently on Inferentia.
DecodeJPEG and NonMaxSuppression are typical examples. In practice, we
may simply place these operators on CPU using the AWS Neuron machine
learning framework integration. However, Inferentia is such a high
performance machine learning accelerator that, once the model
successfully compiles and runs, these simple pre-post-processing
operators can become the new performance bottleneck! In this tutorial,
we explain some commonly used tensorflow techniques for optimizing the
performance of these pre-post-processing operators so that we can fully
unleash the potential of Inferentia.

1. Write JPEG decoding and image shifting/scaling as tensorflow
   operators.

In ``yolo_v4_coco_saved_model.py``, you may find the following code
snippet.

.. code:: python

   import tensorflow as tf
   ...

   def YOLOv4(...
       ...
       x, image_shape = layers.Lambda(lambda t: preprocessor(t, input_shape))(inputs)

       # cspdarknet53
       x = conv2d_unit(x, i32, 3, strides=1, padding='same')
   ...

   def decode_jpeg_resize(input_tensor, image_size):
       tensor = tf.image.decode_png(input_tensor, channels=3)
       shape = tf.shape(tensor)
       tensor = tf.cast(tensor, tf.float32)
       tensor = tf.image.resize(tensor, image_size)
       tensor /= 255.0
       return tf.cast(tensor, tf.float16), shape

   def preprocessor(input_tensor, image_size):
       with tf.name_scope('Preprocessor'):
           tensor = tf.map_fn(
               partial(decode_jpeg_resize, image_size=image_size), input_tensor,
               dtype=(tf.float16, tf.int32), back_prop=False, parallel_iterations=16)
       return tensor

Comparing with the implementation in `the original
repo <https://github.com/miemie2013/Keras-YOLOv4/blob/master/model/yolov4.py>`__,
our difference is the use of ``tf.image.decode_png`` and
``tf.image.resize``, along with a small number of scaling/casting
operators. After this modification, the generated tensorflow SavedModel
now takes JPEG image raw bytes as input, instead of a float32 array
representing the image. When the image resolution is 608x608, this
technique effectively reduces the input image size from 4.4 MB to the
size of a typical JPEG image, which can be as little as hundreds of KB.
When the tensorflow SavedModel is deployed through
`tensorflow/serving <https://github.com/tensorflow/serving>`__, this
technique can very effectively reduce the gRPC transfer overhead of
input images.

2. Replace non-max suppression (NMS) operations by
   ``tf.image.combined_non_max_suppression``.

Another difference of our implementation is the treatment of non-max
suppression, a commmonly used operation for removing redundant bounding
boxes that overlap with other boxes. In an object detection scenario
represented by the COCO dataset where the number of output classes is
large, the hand-fused :literal:`\`tf.image.combined_non_max_suppression`
<https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/combined_non_max_suppression>`_\_
operator can parallelize multi-class NMS on CPU in a very efficient
manner. With proper use of this operator, the bounding box
post-processing step has a less chance of becoming the performance
bottleneck in the end-to-end object detection pipeline.

The following sample code (from ``yolo_v4_coco_saved_model.py``)
demonstrates our method of writing the bounding box post-processing step
using efficient tensorflow operations.

.. code:: python

   ...
       def filter_boxes(outputs):
           boxes_l, boxes_m, boxes_s, box_scores_l, box_scores_m, box_scores_s, image_shape = outputs
           boxes_l, box_scores_l = filter_boxes_one_size(boxes_l, box_scores_l)
           boxes_m, box_scores_m = filter_boxes_one_size(boxes_m, box_scores_m)
           boxes_s, box_scores_s = filter_boxes_one_size(boxes_s, box_scores_s)
           boxes = tf.concat([boxes_l, boxes_m, boxes_s], axis=0)
           box_scores = tf.concat([box_scores_l, box_scores_m, box_scores_s], axis=0)
           image_shape_wh = image_shape[1::-1]
           image_shape_whwh = tf.concat([image_shape_wh, image_shape_wh], axis=-1)
           image_shape_whwh = tf.cast(image_shape_whwh, tf.float32)
           boxes *= image_shape_whwh
           boxes = tf.expand_dims(boxes, 0)
           box_scores = tf.expand_dims(box_scores, 0)
           boxes = tf.expand_dims(boxes, 2)
           nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
               boxes,
               box_scores,
               max_output_size_per_class=nms_top_k,
               max_total_size=nms_top_k,
               iou_threshold=nms_thresh,
               score_threshold=conf_thresh,
               pad_per_class=False,
               clip_boxes=False,
               name='CombinedNonMaxSuppression',
           )
           return nms_boxes[0], nms_scores[0], nms_classes[0]

       def filter_boxes_one_size(boxes, box_scores):
           box_class_scores = tf.reduce_max(box_scores, axis=-1)
           keep = box_class_scores > conf_thresh
           boxes = boxes[keep]
           box_scores = box_scores[keep]
           return boxes, box_scores

       def batch_yolo_out(outputs):
           with tf.name_scope('yolo_out'):
               b_output_lr, b_output_mr, b_output_sr, b_image_shape = outputs
               with tf.name_scope('process_feats'):
                   b_boxes_l, b_box_scores_l = batch_process_feats(b_output_lr, anchors, masks[0])
               with tf.name_scope('process_feats'):
                   b_boxes_m, b_box_scores_m = batch_process_feats(b_output_mr, anchors, masks[1])
               with tf.name_scope('process_feats'):
                   b_boxes_s, b_box_scores_s = batch_process_feats(b_output_sr, anchors, masks[2])
               with tf.name_scope('filter_boxes'):
                   b_nms_boxes, b_nms_scores, b_nms_classes = tf.map_fn(
                       filter_boxes, [b_boxes_l, b_boxes_m, b_boxes_s, b_box_scores_l, b_box_scores_m, b_box_scores_s, b_image_shape],
                       dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
           return b_nms_boxes, b_nms_scores, b_nms_classes

       boxes_scores_classes = layers.Lambda(batch_yolo_out)([output_lr, output_mr, output_sr, image_shape])
   ...

For other advanced data input/output pipeline optimization techniques,
please refer to
https://www.tensorflow.org/guide/data#preprocessing_data.
