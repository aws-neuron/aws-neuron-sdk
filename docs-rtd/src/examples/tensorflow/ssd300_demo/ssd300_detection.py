import argparse
import json
import pkg_resources
from distutils.version import LooseVersion
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow.neuron as tfn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image that is to be detected. Support jpeg and png format.')
    parser.add_argument('--image_with_detections', required=True, help='Path to save image after detection (with bounding boxes drawn). Png format.')
    parser.add_argument('--saved_model', required=True, help='TensorFlow SSD300 SavedModel')
    parser.add_argument('--score_threshold', type=float, default=0.15, help='Minimum required score for drawing a bounding box')
    parser.add_argument('--instances_val2017_json', default=None, help='Json file that contains labeling information')
    parser.add_argument('--save_results', default=None)
    parser.add_argument('--disable_version_check', action='store_true')
    args = parser.parse_args()
    if not args.disable_version_check:
        tfn_version = LooseVersion(pkg_resources.get_distribution('tensorflow-neuron').version)
        if tfn_version < LooseVersion('1.15.0.1.0.1333.0'):
            raise RuntimeError(
                'tensorflow-neuron version {} is too low for this demo. Please upgrade '
                'by "pip install -U tensorflow-neuron --extra-index-url=https://pip.repos.neuron.amazonaws.com"'.format(tfn_version))

    with open(args.image, 'rb') as f:
        img_jpg_bytes = f.read()
    model_feed_dict = {'batch_image': [img_jpg_bytes]}

    predictor = tf.contrib.predictor.from_saved_model(args.saved_model)
    results = predictor(model_feed_dict)
    if args.save_results is not None:
        np.savez(args.save_results, **results)
    boxes_np = results['boxes']
    scores_np = results['scores']
    classes_np = results['classes']

    if args.instances_val2017_json is not None:
        with open(args.instances_val2017_json) as f:
            annotate_json = json.load(f)
        label_info = {idx+1: cat['name'] for idx, cat in enumerate(annotate_json['categories'])}

    plt.switch_backend('agg')
    fig, ax = plt.subplots(1)
    ax.imshow(Image.open(args.image).convert('RGB'))

    wanted = scores_np[0] > args.score_threshold
    for xywh, label_no_bg in zip(boxes_np[0][wanted], classes_np[0][wanted]):
        rect = patches.Rectangle((xywh[0], xywh[1]), xywh[2], xywh[3], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        rx = rx + rect.get_width() / 2.0
        if args.instances_val2017_json is not None:
            ax.annotate(label_info[label_no_bg + 1], (rx, ry), color='w', backgroundcolor='g', fontsize=10,
                        ha='center', va='center', bbox=dict(boxstyle='square,pad=0.01', fc='g', ec='none', alpha=0.5))
    plt.savefig(args.image_with_detections)
    plt.close(fig)


if __name__ == '__main__':
    main()
