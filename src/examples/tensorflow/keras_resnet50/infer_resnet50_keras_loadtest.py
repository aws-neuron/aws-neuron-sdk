""" Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
"""

import shutil
import tensorflow as tf
import os
import time
from concurrent import futures
import numpy as np
import statistics
import argparse
import requests
import tensorflow as tf
import tensorflow.neuron
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

tf.keras.backend.set_image_data_format('channels_last')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--batch_size', type=int, default=5, choices=range(1, 6), help='Batch size of model as it was compiled')
arg_parser.add_argument('--num_neuroncores', type=int, default=1, choices=range(1, 17), help='Number of NeuronCores limit for each partitioned graph')
args = arg_parser.parse_args()

instance_type = requests.get('http://169.254.169.254/latest/meta-data/instance-type').text

avail_neuroncores_dict = {
    'inf1.xlarge' : 4,
    'inf1.2xlarge' : 4,
    'inf1.6xlarge' : 16,
    'inf1.24xlarge' : 64
}

avail_neuroncores = avail_neuroncores_dict.get(instance_type, 0)

USER_BATCH_SIZE = 10 * args.batch_size
NUM_LOOPS_PER_THREAD = 100
COMPILED_MODEL_DIR = "./rn50_fp16_compiled_b" + str(args.batch_size) + "_nc" + str(args.num_neuroncores) + "/1"

# Ensure there's enough buffer capacity to hold in-flight requests in runtime
NUM_INFERS_IN_FLIGHT = args.num_neuroncores + 3
os.environ['NEURON_MAX_NUM_INFERS'] = str(NUM_INFERS_IN_FLIGHT)

num_groups = avail_neuroncores // args.num_neuroncores
group_sizes = [str(args.num_neuroncores)] * num_groups
os.environ['NEURONCORE_GROUP_SIZES'] = ','.join(group_sizes)

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(np.repeat(img_arr2, USER_BATCH_SIZE, axis=0))

# Load model
NUM_THREADS_PER_PREDICTOR = args.num_neuroncores + 1
pred_list = [tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR) for _ in range(num_groups)]
pred_list = pred_list * NUM_THREADS_PER_PREDICTOR
num_threads = len(pred_list)

num_infer_per_thread = []
tot_latency_per_thread = []
thread_active = []
for i in range(num_threads):
    num_infer_per_thread.append(0)
    tot_latency_per_thread.append(0)
    thread_active.append(0)

def one_thread(pred, model_feed_dict, index):
    global num_infer_per_thread
    thread_active[index] = 1
    for i in range(NUM_LOOPS_PER_THREAD):
        start = time.time()
        result = pred(model_feed_dict)
        delta = time.time() - start
        # skip first warmup run
        if i > 0:
            tot_latency_per_thread[index] += delta
        num_infer_per_thread[index] += USER_BATCH_SIZE
        #print(num_infer_per_thread[index])
    thread_active[index] = 0

def current_throughput():
    global num_infer_per_thread
    global args
    iteration = 0
    num_infer = 0
    last_num_infer = num_infer
    throughput_stats = []
    print("Instance type {} with {} NeuronCores".format(instance_type, avail_neuroncores))
    print("NEURON_MAX_NUM_INFERS (env): " + os.environ.get('NEURON_MAX_NUM_INFERS', '<unset>'))
    print("NEURONCORE_GROUP_SIZES (env): " + os.environ.get('NEURONCORE_GROUP_SIZES', '<unset>'))
    print("NUM THREADS: ", num_threads)
    print("NUM_LOOPS_PER_THREAD: ", NUM_LOOPS_PER_THREAD)
    print("USER_BATCH_SIZE: ", USER_BATCH_SIZE)
    while num_infer < NUM_LOOPS_PER_THREAD * USER_BATCH_SIZE * num_threads:
        num_infer = 0
        total_thread_cnt = 0
        for i in range(num_threads):
            num_infer = num_infer + num_infer_per_thread[i]
            total_thread_cnt = total_thread_cnt + thread_active[i]
        current_num_infer = num_infer
        throughput = current_num_infer - last_num_infer
        #print('Active threads: {}, current throughput: {} images/sec'.format(total_thread_cnt, throughput))
        # track throughput over time, after warmup
        if iteration > 4 and total_thread_cnt == num_threads:
            throughput_stats.append(throughput)
        last_num_infer = current_num_infer
        iteration += 1
        time.sleep(1.0)
    time.sleep(1.0)
    tot_latency = 0
    for i in range(num_threads):
        tot_latency += tot_latency_per_thread[i]
    # adjust loop count to remove the first warmup run
    print("Throughput values collected:")
    print(throughput_stats)
    avg_latency = tot_latency / ( (NUM_LOOPS_PER_THREAD-1) * num_threads) * 1000
    print("\nCompiled batch size {}, user batch size {}, throughput stats (images/sec): max={} p99={} p50={}, avg latency {:0.4f} msec/user-batch\n".format(args.batch_size, USER_BATCH_SIZE, max(throughput_stats),
                                                            int(np.percentile(throughput_stats, 99)), 
                                                            int(np.percentile(throughput_stats, 50)),
                                                            avg_latency))  


print("\n*** Compiled batch size {}, user batch size {}, num NeuronCores {} (input shape: {}, saved model dir: {}) ***\n".format(args.batch_size, USER_BATCH_SIZE, args.num_neuroncores, img_arr3.shape, COMPILED_MODEL_DIR))

# Run inference
model_feed_dict={'input_1:0': img_arr3}

executor = futures.ThreadPoolExecutor(max_workers = num_threads + 1)
executor.submit(current_throughput)
for i,pred in enumerate(pred_list):
    executor.submit(one_thread, pred, model_feed_dict, i)
