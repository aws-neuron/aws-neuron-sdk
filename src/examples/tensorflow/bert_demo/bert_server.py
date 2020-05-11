# coding=utf-8

""" Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
    Program to gather information from a system
"""

import sys
import os
import collections
import argparse
import time
import csv
import random
from concurrent import futures
import multiprocessing
from multiprocessing.dummy import Pool
from threading import Lock
import pkg_resources
from distutils.version import LooseVersion
import grpc
import numpy as np
import tensorflow as tf
import mrpc_feature
import tokenization
import mrpc_pb2
sys.path.append(os.path.dirname(__file__))
import mrpc_pb2_grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class BERTService(mrpc_pb2_grpc.mrpcServicer):

    def __init__(self, model_path, parallel, batch_size, bootstrap, vocab_txt, num_thread_per_predictor=2):
        num_queues = parallel * num_thread_per_predictor
        config = tf.ConfigProto(inter_op_parallelism_threads=num_queues, intra_op_parallelism_threads=1)
        tfn_version = LooseVersion(pkg_resources.get_distribution('tensorflow-neuron').version)
        if tfn_version >= LooseVersion('1.15.0.1.0.1333.0'):
            neuroncore_group_sizes = '{}x1'.format(parallel)
            predictor = tf.contrib.predictor.from_saved_model(model_path, config=config)
            self.predictor_list = [predictor for _ in range(num_queues)]
        else:
            neuroncore_group_sizes = ','.join('1' for _ in range(parallel))
            predictor_list = [tf.contrib.predictor.from_saved_model(model_path, config=config) for _ in range(parallel)]
            self.predictor_list = []
            for pred in predictor_list:
                self.predictor_list.extend(pred for _ in range(num_thread_per_predictor))
        os.environ['NEURONCORE_GROUP_SIZES'] = neuroncore_group_sizes
        if self.predictor_list[0].feed_tensors['input_ids'].shape.is_fully_defined():
            self.batch_size = self.predictor_list[0].feed_tensors['input_ids'].shape.as_list()[0]
        else:
            self.batch_size = batch_size
        self.bootstrap = bootstrap
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_txt, do_lower_case=True)
        self.num_infer = 0
        self.num_correct = 0
        self.output_name = list(self.predictor_list[0].fetch_tensors.keys())[0]
        self.iid = 0
        self.throughput_list = []
        self.latency_list = []
        self.max_len_latency_list = 1000
        self.iid_lock = Lock()
        if bootstrap:
            self.request_queue_list = [collections.deque() for _ in self.predictor_list]
            eval_data_path = os.path.join(os.path.dirname(__file__), 'glue_mrpc_dev.tsv')
            tsv = mrpc_feature.read_tsv(eval_data_path)
            for request_queue in self.request_queue_list:
                for _ in range(1024):
                    data_list = random.choices(tsv[1:], k=self.batch_size)
                    model_feed_dict_list = [mrpc_feature.text_pair_to_model_feed_dict(data[3], data[4], self.tokenizer) for data in data_list]
                    label_list = [int(data[0]) for data in data_list]
                    batch_labels = np.array(label_list)
                    batch_feeds = {
                        key: np.concatenate([feed[key] for feed in model_feed_dict_list], axis=0)
                        for key in model_feed_dict_list[0].keys()
                    }
                    request_queue.append((batch_feeds, batch_labels))
        else:
            self.request_queue_list = [[] for _ in self.predictor_list]
        self.result_map = {}
        self.alive = True
        dummy_feed = {
            'input_ids': np.zeros([1, 128], dtype=np.int32),
            'input_mask': np.zeros([1, 128], dtype=np.int32),
            'segment_ids': np.zeros([1, 128], dtype=np.int32),
        }
        self.dummy_feeds = [(None, dummy_feed) for _ in range(self.batch_size)]
        model_feed_dict_list = [dummy_feed for _ in range(self.batch_size)]
        batch_feeds = {
            key: np.concatenate([feed[key] for feed in model_feed_dict_list], axis=0)
            for key in model_feed_dict_list[0].keys()
        }
        pool = Pool(len(self.predictor_list))
        for pred in self.predictor_list:
            pool.apply_async(pred, (batch_feeds,))
            time.sleep(1)
        pool.close()
        pool.join()

    def cleanup(self):
        for pred in self.predictor_list:
            print(pred)
            pred.session.close()

    def current_throughput(self):
        last_num_infer = self.num_infer
        while self.alive:
            current_num_infer = self.num_infer
            throughput = current_num_infer - last_num_infer
            self.throughput_list.append(throughput)
            print('current throughput {}'.format(throughput))
            last_num_infer = current_num_infer
            time.sleep(1)

    def current_throughput_accuracy(self):
        last_num_infer = self.num_infer
        while self.alive:
            current_num_infer = self.num_infer
            accuracy = 0.0 if self.num_infer == 0 else self.num_correct / self.num_infer
            print('current throughput {}, accuracy {}'.format(current_num_infer - last_num_infer, accuracy))
            last_num_infer = current_num_infer
            time.sleep(1)

    def paraphrase(self, text_pair, context):
        iid = self.put_input(text_pair.text_a, text_pair.text_b)
        yes_no = mrpc_pb2.YesNo()
        if self.get_output(iid) == 1:
            yes_no.message = b'paraphrase!'
            yes_no.prediction = b'1'
        else:
            yes_no.message = b'not paraphrase!'
            yes_no.prediction = b'0'
        return yes_no

    def put_input(self, text_a, text_b):
        model_feed_dict = mrpc_feature.text_pair_to_model_feed_dict(text_a, text_b, self.tokenizer)
        with self.iid_lock:
            self.iid += 1
            iid = self.iid
        self.request_queue_list[iid % len(self.request_queue_list)].append((iid, model_feed_dict))
        return iid

    def process_input(self, idx):
        print('input processor is waiting')
        request_queue = self.request_queue_list[idx]
        predictor = self.predictor_list[idx]
        while self.alive:
            if len(request_queue) > 0:
                sublist = request_queue[:self.batch_size]
                request_queue[:self.batch_size] = []
                if len(sublist) != self.batch_size:
                    print('batch with {} garbage entries!'.format(self.batch_size - len(sublist)))
                if len(sublist) < self.batch_size:
                    pad_batch_size = self.batch_size - len(sublist)
                    sublist.extend(self.dummy_feeds[:pad_batch_size])
                iid_list = [iid for iid, _ in sublist]
                model_feed_dict_list = [feed for _, feed in sublist]
                batch_feeds = {
                    key: np.concatenate([feed[key] for feed in model_feed_dict_list], axis=0)
                    for key in model_feed_dict_list[0].keys()
                }
                start = time.time()
                batch_predictions = predictor(batch_feeds)[self.output_name].argmax(-1)
                latency = time.time() - start
                if len(self.latency_list) < self.max_len_latency_list:
                    self.latency_list.append(latency)
                self.result_map.update({iid: pred for iid, pred in zip(iid_list, batch_predictions)})
            time.sleep(0.001)

    def process_input_bootstrap(self, idx):
        print('input processor is waiting')
        request_queue = self.request_queue_list[idx]
        predictor = self.predictor_list[idx]
        while self.alive:
            if len(request_queue) > 0:
                batch_feeds, batch_labels = request_queue.popleft()
                batch_predictions = predictor(batch_feeds)[self.output_name].argmax(-1)
                self.num_infer += self.batch_size
                self.num_correct += (batch_predictions == batch_labels).sum()
                continue
            time.sleep(0.0001)

    def get_output(self, iid):
        while iid not in self.result_map:
            time.sleep(0.001)
        self.num_infer += 1
        return self.result_map.pop(iid)


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=60061, help='gRPC port')
    parser.add_argument('--dir', required=True, help='TensorFlow SavedModel dir')
    parser.add_argument('--parallel', type=int, default=4, help='Number of predictors')
    parser.add_argument('--thread', type=int, default=2, help='Number of threads used by each predictor')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Server loads a dataset and run inference itself')
    args = parser.parse_args()
    vocab_txt = os.path.join(os.path.dirname(__file__), 'uncased_L-24_H-1024_A-16.vocab.txt')
    bert_service = BERTService(args.dir, args.parallel, args.batch, args.bootstrap, vocab_txt, args.thread)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=128),
        options=[('grpc.max_send_message_length', -1),
                 ('grpc.max_receive_message_length', -1)])
    mrpc_pb2_grpc.add_mrpcServicer_to_server(bert_service, server)
    server.add_insecure_port('[::]:{}'.format(args.port))
    server.start()
    try:
        pool = Pool(len(bert_service.predictor_list) + 1)  # +1 for bert_service.current_throughput
        if args.bootstrap:
            monitor_func = bert_service.current_throughput_accuracy
            process_func = bert_service.process_input_bootstrap
        else:
            monitor_func = bert_service.current_throughput
            process_func = bert_service.process_input
        pool.apply_async(monitor_func)
        if args.parallel == 1:
            process_func(0)
        else:
            for idx in range(len(bert_service.predictor_list)):
                pool.apply_async(process_func, (idx,))
        pool.close()
        time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        pass
    bert_service.cleanup()
    bert_service.alive = False
    server.stop(0)


if __name__ == '__main__':
    serve()
