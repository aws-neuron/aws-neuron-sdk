import sys
import os
import argparse
import time
import itertools
from functools import partial
from collections import Counter
import json
import shutil
import pkg_resources
from distutils.version import LooseVersion
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
import tensorflow.neuron as tfn
import torch


def decode_jpeg_resize(input_tensor, image_size):
    # decode jpeg
    tensor = tf.image.decode_png(input_tensor, channels=3)

    # resize
    decoded_shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    decoded_shape_hw = decoded_shape[0:2]
    decoded_shape_hw_float32 = tf.cast(decoded_shape_hw, tf.float32)
    tensor = tf.image.resize(tensor, image_size)

    # normalize
    tensor -= np.array([0.485, 0.456, 0.406]).astype(np.float32) * 255.0
    return tensor, decoded_shape_hw_float32[::-1]


def preprocessor(input_tensor, image_size):
    with tf.name_scope('Preprocessor'):
        tensor, bbox_scale_hw = tf.map_fn(
            partial(decode_jpeg_resize, image_size=image_size), input_tensor,
            dtype=(tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
    return tensor, bbox_scale_hw


def tf_Conv2d(input_tensor, module, first_conv=False):
    np_dtype = input_tensor.dtype.as_numpy_dtype
    kernel_np = module.weight.detach().numpy().transpose([2, 3, 1, 0])
    if first_conv:
        kernel_np /= (np.array([0.229, 0.224, 0.225]).astype(np.float32) * 255.0)[:, np.newaxis]
    kernel = tf.constant(kernel_np.astype(np_dtype))
    if any(module.padding):
        pad_h, pad_w = module.padding
        padding = [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]
        input_tensor = tf.pad(input_tensor, padding)
    stride_h, stride_w = module.stride
    tensor = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride_h, stride_w, 1], padding='VALID')
    if module.bias is not None:
        bias = tf.constant(module.bias.detach().numpy().astype(np_dtype))
        tensor = tf.nn.bias_add(tensor, bias)
    return tensor

def tf_BatchNorm2d(input_tensor, module):
    def _norm_np(ts):
        return ts.astype(input_tensor.dtype.as_numpy_dtype)
    mean = _norm_np(module.running_mean.detach().numpy())
    offset = _norm_np(module.bias.detach().numpy())
    inv_std = np.sqrt(module.running_var.detach().numpy() + module.eps)
    scale_inv_std = _norm_np(module.weight.detach().numpy() / inv_std)
    return scale_inv_std * (input_tensor - mean) + offset

def tf_MaxPool2d(input_tensor, module):
    pad = module.padding
    tensor = tf.pad(input_tensor, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return tf.nn.max_pool2d(tensor, ksize=module.kernel_size, strides=module.stride, padding='VALID')

def tf_Bottleneck(input_tensor, module):
    tensor = tf_Conv2d(input_tensor, module.conv1)
    tensor = tf_BatchNorm2d(tensor, module.bn1)
    tensor = tf.nn.relu(tensor)
    tensor = tf_Conv2d(tensor, module.conv2)
    tensor = tf_BatchNorm2d(tensor, module.bn2)
    tensor = tf.nn.relu(tensor)
    tensor = tf_Conv2d(tensor, module.conv3)
    tensor = tf_BatchNorm2d(tensor, module.bn3)
    if module.downsample is not None:
        input_tensor = tf_Conv2d(input_tensor, module.downsample[0])
        input_tensor = tf_BatchNorm2d(input_tensor, module.downsample[1])
    return tf.nn.relu(input_tensor + tensor)

def tf_SequentialBottleneck(tensor, seq, resnet):
    with tf.name_scope('{}.Sequential'.format(seq)):
        for idx, module in enumerate(resnet[seq]):
            with tf.name_scope('{}.BasicBlock'.format(idx)):
                tensor = tf_Bottleneck(tensor, module)
    return tensor

def tf_bbox_view(detection_feed, modules, ndim):
    results = []
    for idx, (tensor, mod) in enumerate(zip(detection_feed, modules)):
        with tf.name_scope('branch{}'.format(idx)):
            tensor = tf_Conv2d(tensor, mod)
            tensor = tf.transpose(tensor, [0, 3, 1, 2])
            tensor = tf.cast(tensor, tf.float32)

            shape = tensor.shape.as_list()
            batch_size = -1 if shape[0] is None else shape[0]
            new_shape = [batch_size, ndim, np.prod(shape[1:]) // ndim]
            results.append(tf.reshape(tensor, new_shape))
    tensor = tf.concat(results, axis=-1)
    return tensor


def tf_feature_extractor(input_tensor, resnet):
    with tf.name_scope('FeatureExtractor'):
        with tf.name_scope('0.Conv2d'):
            tensor = tf_Conv2d(input_tensor, resnet[0], first_conv=True)
        with tf.name_scope('1.BatchNorm2d'):
            tensor = tf_BatchNorm2d(tensor, resnet[1])
        with tf.name_scope('2.ReLU'):
            tensor = tf.nn.relu(tensor)
        with tf.name_scope('3.MaxPool2d'):
            tensor = tf_MaxPool2d(tensor, resnet[3])
        tensor = tf_SequentialBottleneck(tensor, 4, resnet)
        tensor = tf_SequentialBottleneck(tensor, 5, resnet)
        tensor = tf_SequentialBottleneck(tensor, 6, resnet)
        tensor = tf.cast(tensor, tf.float16)
    return tensor


def tf_box_predictor(tensor, ssd300_torch):
    with tf.name_scope('BoxPredictor'):
        detection_feed = [tensor]
        for idx, block in enumerate(ssd300_torch.additional_blocks):
            with tf.name_scope('{}.Sequential'.format(idx)):
                tensor = tf_Conv2d(tensor, block[0])
                tensor = tf_BatchNorm2d(tensor, block[1])
                tensor = tf.nn.relu(tensor)
                tensor = tf_Conv2d(tensor, block[3])
                tensor = tf_BatchNorm2d(tensor, block[4])
                tensor = tf.nn.relu(tensor)
                detection_feed.append(tensor)
        with tf.name_scope('Boxes'):
            loc = tf_bbox_view(detection_feed, ssd300_torch.loc, ndim=4)
        with tf.name_scope('Probabilities'):
            conf = tf_bbox_view(detection_feed, ssd300_torch.conf, ndim=ssd300_torch.label_num)
    return loc, conf


@tfn.fuse(batch_size=1, dynamic_batch_size=True, compiler_args=['-O2'])
def tf_ssd300(input_tensor, ssd300_torch):
    with tf.name_scope('SSD300'):
        tensor = tf_feature_extractor(input_tensor, ssd300_torch.feature_extractor.feature_extractor)
        loc, conf = tf_box_predictor(tensor, ssd300_torch)
    return loc, conf


def scale_back_batch(bboxes_in, scores_in, scale_xy, scale_wh, dboxes_xywh):
    """
        Do scale and transform from xywh to ltrb
        suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
    """
    with tf.name_scope('ScaleBackBatch'):
        bboxes_in = tf.transpose(bboxes_in, [0, 2, 1])
        scores_in = tf.transpose(scores_in, [0, 2, 1])

        bboxes_xy = bboxes_in[:, :, :2]
        bboxes_wh = bboxes_in[:, :, 2:]
        bboxes_xy *= scale_xy
        bboxes_wh *= scale_wh

        bboxes_xy = bboxes_xy * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
        bboxes_wh = tf.exp(bboxes_wh) * dboxes_xywh[:, :, 2:]

        bboxes_wh_half = 0.5 * bboxes_wh
        bboxes_lt = bboxes_xy - bboxes_wh_half
        bboxes_rb = bboxes_xy + bboxes_wh_half

        bboxes_in = tf.concat([bboxes_lt, bboxes_rb], axis=-1)

        return bboxes_in, tf.nn.softmax(scores_in, axis=-1)

def select_nms_outputs(input_tensors):
    boxes_xywh, scores, classes, valid_detections = input_tensors
    return boxes_xywh[:valid_detections], scores[:valid_detections], classes[:valid_detections]

def postprocessor(ploc_ts, plabel_ts, bbox_scale_hw_ts, scale_xy, scale_wh, dboxes_xywh):
    with tf.name_scope('Postprocessor'):
        ploc_ts = tf.cast(ploc_ts, tf.float32)
        plabel_ts = tf.cast(plabel_ts, tf.float32)
        bboxes_ts, probs_ts = scale_back_batch(ploc_ts, plabel_ts, scale_xy, scale_wh, dboxes_xywh)
        bboxes_ts = bboxes_ts[:, :, tf.newaxis, :]
        probs_ts = probs_ts[:, :, 1:]
        nms_outputs = tf.image.combined_non_max_suppression(
            bboxes_ts,
            probs_ts,
            max_output_size_per_class=200,
            max_total_size=200,
            iou_threshold=0.5,
            score_threshold=0.05,
            pad_per_class=False,
            clip_boxes=False,
            name='CombinedNonMaxSuppression',
        )
        nmsed_boxes_x0y0x1y1, nmsed_scores, nmsed_classes, valid_detections = nms_outputs
        nmsed_boxes_x0y0 = nmsed_boxes_x0y0x1y1[..., :2]
        nmsed_boxes_x1y1 = nmsed_boxes_x0y0x1y1[..., 2:]
        bbox_scale_hw_ts = bbox_scale_hw_ts[:, tf.newaxis, :]
        nmsed_boxes_xy = nmsed_boxes_x0y0 * bbox_scale_hw_ts
        nmsed_boxes_wh = (nmsed_boxes_x1y1 - nmsed_boxes_x0y0) * bbox_scale_hw_ts
        nmsed_boxes_xywh = tf.concat([nmsed_boxes_xy, nmsed_boxes_wh], axis=-1)
        nmsed_boxes_xywh, nmsed_scores, nmsed_classes = tf.map_fn(
            select_nms_outputs, (nmsed_boxes_xywh, nmsed_scores, nmsed_classes, valid_detections),
            dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
    return nmsed_boxes_xywh, nmsed_scores, nmsed_classes


class DefaultBoxes(object):

    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios,
                 scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = np.sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*np.sqrt(alpha), sk1/np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = np.array(self.default_boxes)
        self.dboxes = self.dboxes.clip(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.copy()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_checkpoint', required=True, help='Path to PyTorch SSD300 model checkpoint')
    parser.add_argument('--output_saved_model', required=True, help='Output TensorFlow SavedModel that runs on Inferentia')
    parser.add_argument('--disable_version_check', action='store_true')
    args = parser.parse_args()
    if os.path.exists(args.output_saved_model):
        raise OSError('SavedModel dir {} already exists'.format(args.output_saved_model))

    if not args.disable_version_check:
        neuroncc_version = LooseVersion(pkg_resources.get_distribution('neuron-cc').version)
        if neuroncc_version < LooseVersion('1.0.18000'):
            raise RuntimeError(
                'neuron-cc version {} is too low for this demo. Please upgrade '
                'by "pip install -U neuron-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com"'.format(neuroncc_version))
        tfn_version = LooseVersion(pkg_resources.get_distribution('tensorflow-neuron').version)
        if tfn_version < LooseVersion('1.15.3.1.0.1900.0'):
            raise RuntimeError(
                'tensorflow-neuron version {} is too low for this demo. Please upgrade '
                'by "pip install -U tensorflow-neuron --extra-index-url=https://pip.repos.neuron.amazonaws.com"'.format(tfn_version))

    sys.path.append(os.getcwd())
    from DeepLearningExamples.PyTorch.Detection.SSD.src import model as torch_ssd300_model
    ssd300_torch = torch_ssd300_model.SSD300()
    ckpt = torch.load(args.torch_checkpoint, map_location=torch.device('cpu'))
    ssd300_torch.load_state_dict(ckpt['model'])
    ssd300_torch.eval()

    input_tensor = tf.placeholder(tf.string, [None])
    image_tensor, bbox_scale_hw_tensor = preprocessor(input_tensor, [300, 300])

    dboxes = dboxes300_coco()
    dboxes_xywh = dboxes(order="xywh")[np.newaxis, ...]

    ploc_tensor, plabel_tensor = tf_ssd300(image_tensor, ssd300_torch)
    boxes_tensor, scores_tensor, classes_tensor = postprocessor(
        ploc_tensor, plabel_tensor, bbox_scale_hw_tensor, dboxes.scale_xy, dboxes.scale_wh, dboxes_xywh)
    outputs = {
        'boxes': boxes_tensor,
        'scores': scores_tensor,
        'classes': classes_tensor,
    }

    sess = tf.Session()
    try:
        sess.run(outputs)
    except:
        pass

    for op in sess.graph.get_operations():
        if op.type == 'NeuronOp':
            if not op.get_attr('executable'):
                raise AttributeError(
                    'Neuron executable (neff) is empty. Please check neuron-cc is installed and working properly '
                    '("pip install neuron-cc --force --extra-index-url=https://pip.repos.neuron.amazonaws.com" '
                    'to force reinstall neuron-cc).')
            model_config = op.node_def.attr['model_config'].list
            if model_config.i:
                model_config.i[0] = 1
            else:
                model_config.i.extend([1, 1, 1, 10])
            op._set_attr('model_config', attr_value_pb2.AttrValue(list=model_config))
    tf.saved_model.simple_save(sess, args.output_saved_model, {'batch_image': input_tensor}, outputs)


if __name__ == '__main__':
    main()
