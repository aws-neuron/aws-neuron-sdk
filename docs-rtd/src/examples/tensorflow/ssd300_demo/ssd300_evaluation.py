import argparse
import os
import json
import glob
from concurrent import futures
import time
import pkg_resources
from distutils.version import LooseVersion
import numpy as np
import tensorflow as tf
import tensorflow.neuron as tfn
from pycocotools.cocoeval import COCOeval
from DeepLearningExamples.PyTorch.Detection.SSD.src.coco import COCO
from DeepLearningExamples.PyTorch.Detection.SSD.src.utils import dboxes300_coco
from DeepLearningExamples.PyTorch.Detection.SSD.src.utils import SSDTransformer
from DeepLearningExamples.PyTorch.Detection.SSD.src.utils import COCODetection


def get_val_dataset(val_annotate, val_coco_root):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', required=True, help='TensorFlow SSD300 SavedModel')
    parser.add_argument('--val2017', required=True, help='Path to COCO 2017 validation dataset')
    parser.add_argument('--instances_val2017_json', required=True, help='Json file that contains labeling information')
    parser.add_argument('--num_sessions', type=int, default=1, help='Number of tensorflow sessions')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--throughput_interval', type=int, default=10, help='Interval for counting throughput')
    parser.add_argument('--save_results', default=None)
    parser.add_argument('--disable_version_check', action='store_true')
    args = parser.parse_args()
    if not args.disable_version_check:
        tfn_version = LooseVersion(pkg_resources.get_distribution('tensorflow-neuron').version)
        if tfn_version < LooseVersion('1.15.0.1.0.1333.0'):
            raise RuntimeError(
                'tensorflow-neuron version {} is too low for this demo. Please upgrade '
                'by "pip install -U tensorflow-neuron --extra-index-url=https://pip.repos.neuron.amazonaws.com"'.format(tfn_version))
    predictor_list = [tf.contrib.predictor.from_saved_model(args.saved_model) for _ in range(args.num_sessions)]

    val_dataset = get_val_dataset(args.instances_val2017_json, args.val2017)
    inv_map = {v: k for k, v in val_dataset.label_map.items()}
    model_feed_dict_list = []
    for img_id in val_dataset.img_keys:
        img_path = os.path.join(args.val2017, val_dataset.images[img_id][0])
        with open(img_path, 'rb') as f:
            img_jpg_bytes = f.read()
        model_feed_dict_list.append({'batch_image': [img_jpg_bytes]})

    latency_list = []
    throughput_list = []
    def predict(pred, model_feed_dict):
        start = time.time()
        result = pred(model_feed_dict)
        latency_list.append(time.time() - start)
        return result

    def performance():
        last_num_infer = len(latency_list)
        while len(latency_list) < len(model_feed_dict_list):
            current_num_infer = len(latency_list)
            throughput = (current_num_infer - last_num_infer) / args.throughput_interval
            throughput_list.append(throughput)
            p50 = 0.0
            p90 = 0.0
            if latency_list:
                p50 = np.percentile(latency_list, 50)
                p90 = np.percentile(latency_list, 90)
            print('pid {}: current throughput {}, latency p50={:.3f} p90={:.3f}'.format(os.getpid(), throughput, p50, p90))
            last_num_infer = current_num_infer
            time.sleep(args.throughput_interval)

    executor = futures.ThreadPoolExecutor(max_workers=(args.num_sessions*args.num_threads)+1)
    performance_future = executor.submit(performance)
    eval_futures = []
    for idx, model_feed_dict in enumerate(model_feed_dict_list):
        eval_fut = executor.submit(predict, predictor_list[idx%len(predictor_list)], model_feed_dict)
        eval_futures.append(eval_fut)
    waited_results = []
    for idx, eval_fut in enumerate(eval_futures):
        if idx % 100 == 0:
            print('evaluating image {}/{}'.format(idx, len(eval_futures)))
        waited_results.append(eval_fut.result())
    eval_results = []
    for idx, (img_id, results) in enumerate(zip(val_dataset.img_keys, waited_results)):
        boxes = results['boxes']
        for box, label, prob in zip(results['boxes'][0], results['classes'][0], results['scores'][0]):
            res = [img_id, box[0], box[1], box[2], box[3], prob, inv_map[label+1]]  # +1 to account for background
            eval_results.append(res)
    performance_future.result()

    coco_gt = COCO(annotation_file=args.instances_val2017_json)
    coco_dt = coco_gt.loadRes(np.array(eval_results).astype(np.float32))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if args.save_results is not None:
        np.save(args.save_results, coco_eval.stats)


if __name__ == '__main__':
    main()
