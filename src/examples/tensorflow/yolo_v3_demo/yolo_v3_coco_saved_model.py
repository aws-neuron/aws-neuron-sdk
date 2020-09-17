import argparse
import os
import urllib.request
import tempfile
import shutil
from functools import partial
import numpy as np
import tensorflow as tf


STRIDES = [8, 16, 32]
ANCHORS = np.array([1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]).astype(np.float32).reshape([3, 3, 2])
ANCHOR_PER_SCALE = 3
BOX_SCORE_THRESH = 0.3
UPSAMPLE_METHOD = "resize"
NUM_CLASSES = 80


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, input_size, trainable):

        self.trainable        = trainable
        self.num_class        = NUM_CLASSES
        self.strides          = STRIDES
        self.anchors          = ANCHORS
        self.anchor_per_scale = ANCHOR_PER_SCALE
        self.box_score_thresh = BOX_SCORE_THRESH
        self.upsample_method  = UPSAMPLE_METHOD

        input_data, decoded_shape = preprocessor(input_data, [input_size, input_size])
        self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data)

        def decode_boxes(bboxes_and_decoded_shape):
            conv_lbbox, conv_mbbox, conv_sbbox, decoded_shape = bboxes_and_decoded_shape
            conv_lbbox = tf.cast(conv_lbbox, tf.float32)
            conv_mbbox = tf.cast(conv_mbbox, tf.float32)
            conv_sbbox = tf.cast(conv_sbbox, tf.float32)
            conv_lbbox = conv_lbbox[tf.newaxis, ...]
            conv_mbbox = conv_mbbox[tf.newaxis, ...]
            conv_sbbox = conv_sbbox[tf.newaxis, ...]
            decoded_shape = decoded_shape[tf.newaxis, ...]
            with tf.variable_scope('pred_sbbox'):
                pred_sbbox_coors, pred_sbbox_class_scores = self.decode(conv_sbbox, self.anchors[0], self.strides[0], decoded_shape, input_size)

            with tf.variable_scope('pred_mbbox'):
                pred_mbbox_coors, pred_mbbox_class_scores = self.decode(conv_mbbox, self.anchors[1], self.strides[1], decoded_shape, input_size)

            with tf.variable_scope('pred_lbbox'):
                pred_lbbox_coors, pred_lbbox_class_scores = self.decode(conv_lbbox, self.anchors[2], self.strides[2], decoded_shape, input_size)

            with tf.variable_scope('pred_bbox_filter'):
                pred_bbox_coors = tf.concat([pred_sbbox_coors, pred_mbbox_coors, pred_lbbox_coors], axis=1)
                pred_bbox_class_scores = tf.concat([pred_sbbox_class_scores, pred_mbbox_class_scores, pred_lbbox_class_scores], axis=1)
                nms_top_k = 100
                nms_thresh= 0.45
                coors, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    pred_bbox_coors,
                    pred_bbox_class_scores,
                    max_output_size_per_class=nms_top_k,
                    max_total_size=nms_top_k,
                    iou_threshold=nms_thresh,
                    score_threshold=self.box_score_thresh,
                    pad_per_class=False,
                    clip_boxes=False,
                    name='CombinedNonMaxSuppression',
                )
                scores = scores[..., tf.newaxis]
                classes = classes[..., tf.newaxis]
            return coors[0], scores[0], classes[0]

        with tf.name_scope('Postprocessor'):
            coors, scores, classes = tf.map_fn(
                decode_boxes, [self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, decoded_shape],
                dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)

        with tf.variable_scope('pred_bbox'):
            self.pred_bbox_boxes = tf.identity(coors, name='boxes')
            self.pred_bbox_scores = tf.identity(scores[..., 0], name='scores')
            self.pred_bbox_classes = tf.identity(classes[..., 0], name='classes')

    def __build_nework(self, input_data):
        route_1, route_2, input_data = darknet53(input_data, self.trainable)

        input_data = convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        conv_lobj_branch = convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                   trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
        conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                   trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                   trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride, decoded_shape, input_size):
        conv_output = tf.cast(conv_output, tf.float32)
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_xywh = tf.reshape(pred_xywh, (-1, output_size*output_size*3, pred_xywh.shape[-1]))
        pred_conf = tf.reshape(pred_conf, (-1, output_size*output_size*3))
        pred_prob = tf.reshape(pred_prob, (-1, output_size*output_size*3, pred_prob.shape[-1]))

        return tf_postprocess_boxes(pred_xywh, pred_conf, pred_prob, decoded_shape, input_size, self.box_score_thresh)


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = convolutional(input_data, filters_shape=(3, 3, 32,  64), trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = convolutional(input_data, filters_shape=(3, 3,  64, 128), trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = convolutional(input_data, filters_shape=(3, 3, 128, 256), trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 256, 512), trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        weight = tf.cast(weight, tf.float16)
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable,
                                                 fused=False)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            bias = tf.cast(bias, tf.float16)
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    short_cut = input_data
    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')
        residual_output = input_data + short_cut
    return residual_output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output


def decode_jpeg_resize(input_tensor, image_size):
    tensor = tf.image.decode_png(input_tensor, channels=3)
    shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    tensor = tf.image.resize_image_with_pad(tensor, image_size[0], image_size[1])
    tensor /= 255.0
    return tf.cast(tensor, tf.float16), shape


def preprocessor(input_tensor, image_size):
    with tf.name_scope('Preprocessor'):
        batch_tensor, batch_shape = tf.map_fn(
            partial(decode_jpeg_resize, image_size=image_size), input_tensor,
            dtype=(tf.float16, tf.int32), back_prop=False, parallel_iterations=16)
    return batch_tensor, batch_shape


def tf_postprocess_boxes(pred_xywh, pred_conf, pred_prob, org_img_shape, input_size, score_threshold):
    batch_size = tf.shape(pred_xywh)[0]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = tf.concat([pred_xywh[:, :, :2] - pred_xywh[:, :, 2:] * 0.5,
                           pred_xywh[:, :, :2] + pred_xywh[:, :, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_wh = org_img_shape[:, tf.newaxis, 1::-1]
    org_whwh = tf.concat([org_wh, org_wh], axis=-1)
    org_whwh = tf.cast(org_whwh, tf.float32)
    input_size = np.float32(input_size)
    resize_ratio = input_size / tf.reduce_max(org_whwh, axis=-1)
    dwhwh = (input_size - resize_ratio * org_whwh) / 2
    pred_coor = (pred_coor - dwhwh) / resize_ratio

    # # (5) discard some boxes with low scores
    scores = pred_conf * tf.reduce_max(pred_prob, axis=-1)
    score_mask = scores > score_threshold
    coors = pred_coor[score_mask]
    pred_conf = pred_conf[score_mask]
    pred_conf = tf.reshape(pred_conf, [batch_size, -1, 1])
    pred_prob = pred_prob[score_mask]
    pred_prob = tf.reshape(pred_prob, [batch_size, -1, pred_prob.shape[-1]])
    class_scores = pred_conf * pred_prob
    coors = tf.reshape(coors, [batch_size, -1, 1, coors.shape[-1]])
    class_scores = tf.reshape(class_scores, [batch_size, -1, class_scores.shape[-1]])
    return coors, class_scores


def convert_weights(org_weights_path, cur_weights_path, input_size):
    org_weights_mess = []
    with tf.Session(graph=tf.Graph()) as sess:
        load = tf.train.import_meta_graph(org_weights_path + '.meta')
        load.restore(sess, org_weights_path)
        for var in tf.global_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            var_shape = var.shape
            org_weights_mess.append([var_name, var_shape])
            print("=> " + str(var_name).ljust(50), var_shape)
        print()

    cur_weights_mess = []
    with tf.Session(graph=tf.Graph()) as sess:
        with tf.name_scope('input'):
            input_data = tf.placeholder(dtype=tf.string, shape=(None,), name='input_data')
            training = tf.placeholder(dtype=tf.bool, name='trainable')
        model = YOLOV3(input_data, input_size, training)
        for var in tf.global_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            var_shape = var.shape
            print(var_name_mess[0])
            cur_weights_mess.append([var_name, var_shape])
            print("=> " + str(var_name).ljust(50), var_shape)

        org_weights_num = len(org_weights_mess)
        cur_weights_num = len(cur_weights_mess)
        if cur_weights_num != org_weights_num:
            raise RuntimeError

        print('=> Number of weights that will rename:\t%d' % cur_weights_num)
        cur_to_org_dict = {}
        for index in range(org_weights_num):
            org_name, org_shape = org_weights_mess[index]
            cur_name, cur_shape = cur_weights_mess[index]
            if cur_shape != org_shape:
                print(org_weights_mess[index])
                print(cur_weights_mess[index])
                raise RuntimeError
            cur_to_org_dict[cur_name] = org_name
            print("=> " + str(cur_name).ljust(50) + ' : ' + org_name)

        with tf.name_scope('load_save'):
            name_to_var_dict = {var.op.name: var for var in tf.global_variables()}
            restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
            load = tf.train.Saver(restore_dict)
            save = tf.train.Saver(tf.global_variables())
            for var in tf.global_variables():
                print("=> " + var.op.name)

        sess.run(tf.global_variables_initializer())
        print('=> Restoring weights from:\t %s' % org_weights_path)
        load.restore(sess, org_weights_path)
        save.save(sess, cur_weights_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()
    if os.path.exists(args.model_dir):
        raise OSError('Directory {} already exists; please specify a different path for the tensorflow SavedModel'.format(args.model_dir))
    with tempfile.TemporaryDirectory() as workdir:
        ckpt_file = os.path.join(workdir, './yolov3_coco_demo.ckpt')
        input_size = 416
        if not os.path.isfile(ckpt_file + '.meta'):
            yolov3_coco_tar_gz = os.path.join(workdir, './yolov3_coco.tar.gz')
            url = 'https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz'
            print('Downloading from {}'.format(url))
            urllib.request.urlretrieve(url, yolov3_coco_tar_gz)
            shutil.unpack_archive(yolov3_coco_tar_gz, extract_dir=workdir)
            convert_weights(os.path.join(workdir, './yolov3_coco.ckpt'), ckpt_file, input_size)

        input_tensor_name = 'input/input_data:0'
        output_names = ['boxes', 'scores', 'classes']
        output_tensor_names = ['pred_bbox/boxes:0', 'pred_bbox/scores:0', 'pred_bbox/classes:0']
        with tf.Session(graph=tf.Graph()) as sess:
            with tf.name_scope('input'):
                input_data = tf.placeholder(dtype=tf.string, shape=[None], name='input_data')
            model = YOLOV3(input_data, input_size, trainable=False)
            print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)
            input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
            inputs = {'image': input_tensor}
            outputs = {name: sess.graph.get_tensor_by_name(tensor_name) for name, tensor_name in zip(output_names, output_tensor_names)}
            tf.saved_model.simple_save(sess, args.model_dir, inputs, outputs)
    print('tensorflow YOLO v3 SavedModel generated at {}'.format(args.model_dir))


if __name__ == '__main__':
    main()
