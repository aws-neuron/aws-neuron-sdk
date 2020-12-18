import os
import argparse
import time
import numpy as np
import requests
import sys
from concurrent import futures

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--url', help='Torchserve model URL', type=str, default=f'http://127.0.0.1:8080/predictions/bert-max_length128-batch_size6')
parser.add_argument('--num_thread', type=int, default=64, help='Number of threads invoking the model URL')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--sequence_length', type=int, default=128)
parser.add_argument('--latency_window_size', type=int, default=1000)
parser.add_argument('--throughput_time', type=int, default=300)
parser.add_argument('--throughput_interval', type=int, default=10)
args = parser.parse_args()

data = { 'seq_0': 'A completely made up sentence.',
    'seq_1': 'Well, I suppose they are all made up.' }

live = True
num_infer = 0
latency_list = []
def one_thread(pred, feed_data):
    global latency_list
    global num_infer
    global live
    session = requests.Session()
    while True:
        start = time.time()
        result = session.post(pred, data=feed_data)
        latency = time.time() - start
        latency_list.append(latency)
        num_infer += 1
        if not live:
            break

def current_performance():
    last_num_infer = num_infer
    for _ in range(args.throughput_time // args.throughput_interval):
        current_num_infer = num_infer
        throughput = (current_num_infer - last_num_infer) / args.throughput_interval
        p50 = 0.0
        p90 = 0.0
        if latency_list:
            p50 = np.percentile(latency_list[-args.latency_window_size:], 50)
            p90 = np.percentile(latency_list[-args.latency_window_size:], 90)
        print('pid {}: current throughput {}, latency p50={:.3f} p90={:.3f}'.format(os.getpid(), throughput, p50, p90))
        sys.stdout.flush()
        last_num_infer = current_num_infer
        time.sleep(args.throughput_interval)
    global live
    live = False

with futures.ThreadPoolExecutor(max_workers=args.num_thread+1) as executor:
    executor.submit(current_performance)
    for _ in range(args.num_thread):
        executor.submit(one_thread, args.url, data)