# coding=utf-8

""" Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
    Program to gather information from a system
"""

import sys
import os
import argparse
import random
import time
import grpc
import mrpc_pb2
sys.path.append(os.path.dirname(__file__))
import mrpc_pb2_grpc
import mrpc_feature


def client():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=60061, help='gRPC port')
    parser.add_argument('--pair', default=None, help='Text pair')
    parser.add_argument('--cycle', type=int, default=1, help='Number of inference cycles')
    parser.add_argument('--save-accuracy', default=None, help='Save accuracy to file')
    args = parser.parse_args()
    text_pair = mrpc_pb2.TextPair()
    if args.pair is not None:
        text_a, text_b = args.pair
        text_pair.text_a = text_a.encode()
        text_pair.text_b = text_b.encode()
    else:
        eval_data_path = os.path.join(os.path.dirname(__file__), 'glue_mrpc_dev.tsv')
        tsv = mrpc_feature.read_tsv(eval_data_path)
    with grpc.insecure_channel('127.0.0.1:{}'.format(args.port)) as channel:
        stub = mrpc_pb2_grpc.mrpcStub(channel)
        num_correct = 0
        very_start = time.time()
        for _ in range(args.cycle):
            if args.pair is None:
                data = random.choice(tsv[1:])
                text_pair.text_a = data[3].encode()
                text_pair.text_b = data[4].encode()
            start = time.time()
            yes_no = stub.paraphrase(text_pair)
            elapsed = time.time() - start
            if data is None:
                evaluation = ''
            else:
                if yes_no.prediction.decode() == data[0]:
                    num_correct += 1
                evaluation = 'correct, ' if yes_no.prediction.decode() == data[0] else 'incorrect, '
            print('{} ({}latency {} s)'.format(yes_no.message.decode(), evaluation, elapsed))
        if args.cycle > 1:
            accuracy = num_correct / args.cycle
            print('took {} s for {} cycles, accuracy {}'.format(time.time() - very_start, args.cycle, accuracy))
            if args.save_accuracy is not None:
                with open(args.save_accuracy, 'w') as f:
                    f.write(str(accuracy))


if __name__ == '__main__':
    client()
