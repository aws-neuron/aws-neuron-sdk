""" Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
"""

import time
import shutil
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
import tensorflow.neuron as tfn

tf.keras.backend.set_image_data_format('channels_last')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--batch_size', type=int, default=5, choices=range(1, 6), help='Input data batch size for compilation of model')
arg_parser.add_argument('--num_neuroncores', type=int, default=1, choices=range(1, 17), help='Number of NeuronCores limit for each partitioned graph')
args = arg_parser.parse_args()

def pb_to_saved_model(pb_path, input_names, output_names, model_dir):
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(open(pb_path, 'rb').read())
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name='')
        inputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in input_names.items()}
        outputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in output_names.items()}
        tf.saved_model.simple_save(sess, model_dir, inputs, outputs)

saved_model_dir = "rn50_fp16"

shutil.rmtree(saved_model_dir, ignore_errors=True)

pb_to_saved_model("resnet50_fp16_keras_opt.pb", {"input_1:0": "input_1:0"}, {"probs/Softmax:0" : "probs/Softmax:0"}, saved_model_dir)

batch_size = args.batch_size
img_arr = np.zeros([batch_size, 224, 224, 3], dtype='float16')
compiled_saved_model_dir = saved_model_dir + "_compiled_b" + str(batch_size) + "_nc" + str(args.num_neuroncores)
shutil.rmtree(compiled_saved_model_dir + "/1", ignore_errors=True)

print("\n*** Batch size {}, num NeuronCores {} (input shape: {}, saved model dir: {}) ***\n".format(batch_size, args.num_neuroncores, img_arr.shape, compiled_saved_model_dir))
compiler_args = ['--batching_en', '--rematerialization_en', '--spill_dis',
                 '--sb_size', str((batch_size + 6)*10), 
                 '--enable-replication', 'True',
                 '--num-neuroncores', str(args.num_neuroncores)]
static_weights = False
if args.num_neuroncores >= 8:
    compiler_args.append('--static-weights')
    static_weights = True

shutil.rmtree('compiler_workdir', ignore_errors=True)
start = time.time()
rslts = tfn.saved_model.compile(saved_model_dir, compiled_saved_model_dir + "/1",
               model_feed_dict={'input_1:0' : img_arr},
               compiler_workdir='compiler_workdir',
               dynamic_batch_size=True,
               compiler_args = compiler_args)
delta = time.time() - start
perc_on_inf = rslts['OnNeuronRatio'] * 100

compile_success = False
if perc_on_inf < 50:
    print("\nERROR: Compilation finished in {:.0f} seconds with less than 50% operations placed on Inferentia ({:.1f}%)\n".format(delta, perc_on_inf))
    if '--static-weights' in compiler_args:
        print("INFO: Retry compilation without static weights")
        compiler_args.remove('--static-weights')
        static_weights = False
        shutil.rmtree(compiled_saved_model_dir + "/1", ignore_errors=True)
        shutil.rmtree('compiler_workdir2', ignore_errors=True)
        start = time.time()
        rslts = tfn.saved_model.compile(saved_model_dir, compiled_saved_model_dir + "/1",
                   model_feed_dict={'input_1:0' : img_arr},
                   compiler_workdir='compiler_workdir2',
                   dynamic_batch_size=True,
                   compiler_args = compiler_args)
        delta = time.time() - start
        perc_on_inf = rslts['OnNeuronRatio'] * 100
        if perc_on_inf < 50:
            print("\nERROR: Retry compilation finished in {:.0f} seconds with less than 50% operations placed on Inferentia ({:.1f}%)\n".format(delta, perc_on_inf))
        else:    
            print("\nINFO: Retry compilation finished in {:.0f} seconds with {:.1f}% operations placed on Inferentia\n".format(delta, perc_on_inf))
            compile_success = True
else:    
    print("\nINFO: Compilation finished in {:.0f} seconds with {:.1f}% operations placed on Inferentia\n".format(delta, perc_on_inf))
    compile_success = True

# Prepare SavedModel for uploading to Inf1 instance
completion_code = 0
if compile_success:
    shutil.make_archive('./' + compiled_saved_model_dir, 'zip', './', compiled_saved_model_dir)
    completion_code = 1 + int(static_weights)

print(completion_code)

exit(int(not compile_success))
