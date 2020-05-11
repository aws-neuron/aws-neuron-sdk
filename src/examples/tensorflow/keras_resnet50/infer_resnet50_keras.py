""" Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
"""

import os
import time
import shutil
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

parser = argparse.ArgumentParser()
parser.add_argument("--graph", default="resnet50_fp32_keras.pb", help="Graph to use for inference", required=True)
parser.add_argument("--input", default="input_1", help="Input of graph")
parser.add_argument("--output", default="probs/Softmax", help="Output of graph")
args = parser.parse_args()

tf.keras.backend.set_image_data_format('channels_last')

def pb_to_saved_model(pb_path, input_names, output_names, model_dir):
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(open(pb_path, 'rb').read())
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name='')
        inputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in input_names.items()}
        outputs = {name: sess.graph.get_tensor_by_name(ts_name) for name, ts_name in output_names.items()}
        tf.saved_model.simple_save(sess, model_dir, inputs, outputs)

SAVED_MODEL_DIR = './rn50_fp16'
shutil.rmtree(SAVED_MODEL_DIR, ignore_errors=True)
input_tname="{}:0".format(args.input)
output_tname="{}:0".format(args.output)
pb_to_saved_model(args.graph, {input_tname : input_tname}, {output_tname : output_tname}, SAVED_MODEL_DIR)

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(np.repeat(img_arr2, 1, axis=0))

# Load model
predictor_host = tf.contrib.predictor.from_saved_model(SAVED_MODEL_DIR)

# Run inference
model_feed_dict={'input_1:0': img_arr3}
infa_rslts = predictor_host(model_feed_dict);
print(resnet50.decode_predictions(infa_rslts[output_tname], top=5)[0])
