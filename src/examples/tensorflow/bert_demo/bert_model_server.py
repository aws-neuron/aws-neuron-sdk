# coding=utf-8

""" Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
    Program to gather information from a system
"""

import os
import argparse
import subprocess
import time


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving', required=True, help='Path to tf-serving binary')
    parser.add_argument('--dir', required=True, help='TensorFlow SavedModel dir')
    parser.add_argument('--port', default=8500, help='gRPC port')
    parser.add_argument('--parallel', type=int, default=8, help='Number of predictors')
    args = parser.parse_args()
    model = os.path.abspath(args.dir)
    model_with_version = os.path.join(model, '1')
    if not os.path.exists(model_with_version):
        os.makedirs(model_with_version)
        os.symlink(os.path.join(model, 'variables'), os.path.join(model_with_version, 'variables'))
        os.symlink(os.path.join(model, 'saved_model.pb'), os.path.join(model_with_version, 'saved_model.pb'))
    process_list = []
    for _ in range(args.parallel):
        proc = subprocess.Popen([
            args.serving, '--model_base_path={}'.format(model), '--port={}'.format(args.port),
            '--tensorflow_intra_op_parallelism=1', '--tensorflow_inter_op_parallelism=1'
        ])
        process_list.append(proc)
    try:
        time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        for proc in process_list:
            proc.terminate()
            proc.wait()


if __name__ == '__main__':
    serve()
