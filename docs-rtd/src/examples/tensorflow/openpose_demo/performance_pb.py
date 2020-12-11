"""
Usage: python performance_pb.py /path/to/graph_opt_neuron.pb
"""
import os
import argparse
from concurrent import futures
import numpy as np
import tensorflow as tf


def run_with_dummy(sess, dummy_feed_dict, num_inferences):
    for _ in range(num_inferences):
        sess.run('Openpose/concat_stage7:0', dummy_feed_dict)


def main():
    os.environ['NEURONCORE_GROUP_SIZES'] = '4x1'
    parser = argparse.ArgumentParser()
    parser.add_argument('pb_path', help='SaveModel')
    parser.add_argument('--num_thread', type=int, default=8, help='Number of threads that work on each tensorflow session')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--net_resolution', default='656x368')
    parser.add_argument('--num_inferences', type=int, default=200)
    args = parser.parse_args()
    dim_w, dim_h = args.net_resolution.split('x')
    dim_w = int(dim_w)
    dim_h = int(dim_h)
    graph_def = tf.GraphDef()
    with open(args.pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    measure_performance(window_size=10)
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name='')
        input_name = 'image:0'
        input_shape = sess.graph.get_tensor_by_name(input_name).shape.as_list()
        input_shape[0] = args.batch_size
        input_shape[1] = dim_h
        input_shape[2] = dim_w
        dummy_feed_dict = {input_name: np.zeros(input_shape).astype(np.float32)}
        with futures.ThreadPoolExecutor(max_workers=args.num_thread) as executor:
            fut_list = [executor.submit(run_with_dummy, sess, dummy_feed_dict, args.num_inferences) for _ in range(args.num_thread)]
            res_list = [fut.result() for fut in fut_list]


"""
Copyright (C) 2020, Amazon.com. All Rights Reserved
"""
import os
import atexit
import time
import math
import json
from collections import OrderedDict, Counter
from contextlib import contextmanager, ContextDecorator
from functools import wraps
from tensorflow.python.client import session
from tensorflow.python.platform import tf_logging as logging


class measure_performance(ContextDecorator):
    """Convenient tool for performance measurements.

    Can be apply on tensorflow session.run, tf-serving unary gRPC calls, or a given custom function.

    Usage:

    To generate performance report for the entire Python or gRPC-client process, insert
    the following function call before running inferences:

    `tfn.measure_performance()`

    Then latency/throughput report will be generated when the process terminates.

    Alternatively, it is possible to use `tfn.measure_performance` programmatically
    as a context manager. Performance measurement will be done for all inferences
    happening under this context. Report will be displayed as INFO level log when exiting
    the context. It is also possible to obtain a JSON format report in Python.

    For example:

    ```
    with tfn.measure_performance() as perf:
        ... (run some inferences) ...
    report_json = perf.report()
    report_full_json = perf.report(verbosity=1)
    ```

    """

    def __init__(self, func=None, window_size=1):
        self.perf_tracker = PerformanceTracker(window_size)
        atexit.register(self.perf_tracker.report)
        self._original_run = session.Session.run
        self._original_grpc_call = None
        if callable(func):
            self.perf_tracker.register_func(self._track_performance(func))
        else:
            session.Session.run = self._track_performance(session.Session.run)
            try:
                import grpc
                from tensorflow_serving.apis import prediction_service_pb2_grpc
                dummy_stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc.insecure_channel(''))
                self._grpc_callable_type = type(dummy_stub.Predict)
                self._original_grpc_call = self._grpc_callable_type.__call__
            except ImportError:
                pass
            if callable(self._original_grpc_call):
                self._grpc_callable_type.__call__ = self._track_performance(
                    grpc._channel._UnaryUnaryMultiCallable.__call__
                )

    def __enter__(self):
        return self.perf_tracker

    def __exit__(self, *exc):
        atexit.unregister(self.perf_tracker.report)
        self.perf_tracker.report()
        session.Session.run = self._original_run
        if self._original_grpc_call is not None:
            self._grpc_callable_type.__call__ = self._original_grpc_call
        return False

    def _track_performance(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            self.perf_tracker.add_timestamps(start, end)
            return result
        return wrapper


class PerformanceTracker(ContextDecorator):

    description = (
        "Latency unit: second. Throughput unit: number of batched inferences per second. "
        "Reported throughput is a lower bound of the actual throughput as inferences "
        "spanning across window boundaries are not counted towards any of the windows. "
        "'Quiet' periods (i. e., window buckets where the inference function is not called) "
        "are not counted towards the reported average throughput."
    )

    def __init__(self, window_size):
        self.window_size = window_size
        self.timestamps_list = []
        self._func = None

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def register_func(self, func):
        self._func = func

    def add_timestamps(self, start, end):
        self.timestamps_list.append([start, end])

    def report(self, verbosity=0):
        if self.timestamps_list:
            latency_list = [end - start for start, end in self.timestamps_list]
            latency_json = {
                'p50': percentile(latency_list, 50),
                'p90': percentile(latency_list, 90),
                'p99': percentile(latency_list, 99),
                'p100': percentile(latency_list, 100),
            }
            bucketed_timestamps = [self._get_bucket(start, end) for start, end in self.timestamps_list]
            counted_buckets = Counter(item for item in bucketed_timestamps if item is not None)
            bucket_throughputs = [(key, value / self.window_size) for key, value in sorted(counted_buckets.items())]
            busy_throughputs = list(OrderedDict((key, value) for key, value in bucket_throughputs).values())
            throughput_json = {
                'peak': max(busy_throughputs),
                'median': percentile(busy_throughputs, 50),
                'average': sum(busy_throughputs) / len(busy_throughputs),
            }
            if verbosity > 0:
                throughput_json['trend'] = busy_throughputs
            report_json = {
                'pid': os.getpid(),
                'throughput': throughput_json,
                'latency': latency_json,
                'description': PerformanceTracker.description,
            }
            with _logging_show_info():
                logging.info('performance report:\n{}'.format(json.dumps(report_json, indent=4)))
            return report_json

    def _get_bucket(self, start, end):
        bucketed_start = math.floor(start / self.window_size) * self.window_size
        bucketed_end = math.ceil(end / self.window_size) * self.window_size
        if bucketed_end - bucketed_start == self.window_size:
            return bucketed_start
        else:
            return None


def percentile(number_list, percent):
    pos_float = len(number_list) * percent / 100
    max_pos = len(number_list) - 1
    pos_floor = min(math.floor(pos_float), max_pos)
    pos_ceil = min(math.ceil(pos_float), max_pos)
    number_list = sorted(number_list)
    return number_list[pos_ceil] if pos_float - pos_floor > 0.5 else number_list[pos_floor]


@contextmanager
def _logging_show_info():
    try:
        verbosity = logging.get_verbosity()
        logging.set_verbosity(logging.INFO)
        yield
    finally:
        logging.set_verbosity(verbosity)


if __name__ == '__main__':
    main()
