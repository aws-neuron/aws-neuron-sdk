# coding=utf-8

""" Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
    Program to gather information from a system
"""

import os
import argparse
import shlex
import numpy as np
import tensorflow as tf
from tensorflow.neuron import fuse
from tensorflow.core.framework import attr_value_pb2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_saved_model', required=True, help='Original SaveModel')
    parser.add_argument('--output_saved_model', required=True, help='Output SavedModel that runs on Inferentia')
    parser.add_argument('--dtype', default='float16', help='Data type for weights')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sequence_length', type=int, default=128)
    parser.add_argument('--crude_gelu', action='store_true')
    parser.add_argument('--aggressive_optimizations', action='store_true')
    args = parser.parse_args()
    if os.path.exists(args.output_saved_model):
        raise OSError('output_saved_model {} already exists'.format(args.output_saved_model))
    dtype = tf.float16 if args.dtype == 'float16' else tf.float32
    if args.aggressive_optimizations:
        args.crude_gelu = True
    bert = NeuronBERTMRPC(
        args.input_saved_model,
        dtype=dtype,
        batch_size=args.batch_size,
        seq_len=args.sequence_length,
        crude_gelu=args.crude_gelu,
        aggressive_fp16_cast=args.aggressive_optimizations,
    )

    fuser = fuse(compiler_args=['--fp32-cast', 'matmult'], timeout=360000)
    bert.encoder = fuser(bert.encoder)

    input_ids = bert.input_ids
    input_mask = bert.input_mask
    segment_ids = bert.segment_ids
    with tf.Session(graph=tf.Graph()) as sess:
        input_ids_ph_shape = input_ids.shape.as_list()
        input_ids_ph_shape[0] = None
        input_ids_ph = tf.placeholder(input_ids.dtype, input_ids_ph_shape, name='input_ids')

        input_mask_ph_shape = input_mask.shape.as_list()
        input_mask_ph_shape[0] = None
        input_mask_ph = tf.placeholder(input_mask.dtype, input_mask_ph_shape, name='input_mask')

        segment_ids_ph_shape = segment_ids.shape.as_list()
        segment_ids_ph_shape[0] = None
        segment_ids_ph = tf.placeholder(segment_ids.dtype, segment_ids_ph_shape, name='segment_ids')

        dummy_reshapes = []
        discard_op_names = set()

        with tf.name_scope('bert/embeddings'):
            expand_dims = tf.expand_dims(input_ids_ph, axis=-1)
            batch_size = tf.shape(input_ids_ph)[0]
            reshape = tf.reshape(expand_dims, [batch_size * bert.seq_len])
            gatherv2 = tf.gather(bert.weights_dict['bert/embeddings/word_embeddings:0'], reshape, axis=0)
            reshape_1 = tf.reshape(gatherv2, [batch_size, bert.seq_len, bert.hid_size])
            reshape_2 = tf.reshape(segment_ids_ph, [batch_size * bert.seq_len])
            one_hot = tf.one_hot(reshape_2, depth=2)
            matmul = tf.matmul(one_hot, bert.weights_dict['bert/embeddings/token_type_embeddings:0'])
            reshape_3 = tf.reshape(matmul, [batch_size, bert.seq_len, bert.hid_size])
            slice0 = tf.slice(bert.weights_dict['bert/embeddings/position_embeddings:0'], begin=[0, 0], size=[bert.seq_len, -1])
            add_1 = reshape_1 + reshape_3 + slice0
            input_tensor = tf.reshape(add_1, [batch_size, bert.seq_len, bert.hid_size])
        with tf.name_scope('bert/encoder'):
            reshape = tf.reshape(input_mask_ph, [batch_size, 1, 1, bert.seq_len])
            bias_tensor = tf.cast(reshape, tf.float32)
            bias_tensor = 1.0 - bias_tensor
            bias_tensor = bias_tensor * -10000.0
            bias_tensor = tf.cast(bias_tensor, bert.dtype)
        tensor = bert.layer_norm(input_tensor, 'embeddings', force_float32=True)

        tensor = tf.reshape(tensor, [bert.batch_size, bert.seq_len, bert.hid_size])
        dummy_reshapes.append(tensor)
        discard_op_names.add(tensor.op.name)
        bias_tensor = tf.reshape(bias_tensor, [bert.batch_size, 1, 1, bert.seq_len])
        dummy_reshapes.append(bias_tensor)
        discard_op_names.add(bias_tensor.op.name)

        logits = bert.encoder(tensor, bias_tensor)
        with tf.name_scope('loss'):
            if bert.dtype is not tf.float32:
                logits = tf.cast(logits, tf.float32)
            probabilities = tf.nn.softmax(logits)
        for rts in dummy_reshapes:
            neuron_op = rts.consumers()[0]
            neuron_op._update_input(list(neuron_op.inputs).index(rts), rts.op.inputs[0])
        try:
            sess.run(probabilities)
        except:
            pass
        graph_def = sess.graph.as_graph_def()
    new_graph_def = tf.GraphDef()
    new_graph_def.node.MergeFrom(node for node in graph_def.node if node.name not in discard_op_names)
    neuron_op_node = [node for node in new_graph_def.node if node.op == 'NeuronOp'][0]
    neuron_op_node.attr['input_batch_axis'].list.i[:] = [0, 0]
    neuron_op_node.attr['output_batch_axis'].list.i[:] = [0]

    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(new_graph_def, name='')
        inputs = {
            'input_ids': sess.graph.get_tensor_by_name(input_ids_ph.name),
            'input_mask': sess.graph.get_tensor_by_name(input_mask_ph.name),
            'segment_ids': sess.graph.get_tensor_by_name(segment_ids_ph.name),
        }
        outputs = {
            'probabilities': sess.graph.get_tensor_by_name(probabilities.name)
        }
        try:
            sess.run(probabilities)
        except:
            pass
        neuron_op = [op for op in sess.graph.get_operations() if op.type == 'NeuronOp'][0]
        if not neuron_op.get_attr('executable'):
            raise AttributeError('Neuron executable (neff) is empty. Please check neuron-cc is installed and working properly (`pip install neuron-cc` to install neuron-cc).')
        tf.saved_model.simple_save(sess, args.output_saved_model, inputs, outputs)


class NeuronBERTMRPC:

    def __init__(self, bert_saved_model, dtype=tf.float16, batch_size=4, seq_len=128, crude_gelu=False, aggressive_fp16_cast=False):
        predictor = tf.contrib.predictor.from_saved_model(bert_saved_model)
        sess = predictor.session
        self.input_ids = predictor.feed_tensors['input_ids']
        self.input_mask = predictor.feed_tensors['input_mask']
        self.segment_ids = predictor.feed_tensors['segment_ids']
        weights_dict = {}
        for op in sess.graph.get_operations():
            if op.type == 'Const':
                tensor = op.outputs[0]
                weights_dict[tensor.name] = tensor
            if op.type == 'Identity' and op.name.endswith('read'):
                tensor = op.outputs[0]
                weights_dict[tensor.op.inputs[0].name] = tensor
        self.weights_dict = sess.run(weights_dict)
        self.dtype = dtype
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hid_size, self.inter_size = self.weights_dict['bert/encoder/layer_0/intermediate/dense/kernel:0'].shape
        self.num_heads = sess.graph.get_tensor_by_name('bert/encoder/layer_0/attention/self/Reshape:0').shape.as_list()[2]
        self.head_size = self.hid_size // self.num_heads
        self.eps = self.weights_dict['bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y:0']
        self.crude_gelu = crude_gelu
        self.layer_norm_dtype = tf.float16 if aggressive_fp16_cast else tf.float32
        sess.close()

    def encoder(self, tensor, bias_tensor):
        tensor = tf.reshape(tensor, [self.batch_size * self.seq_len, self.hid_size])
        for layer_id in range(24):
            mid_layer_name = 'layer_{}'.format(layer_id)
            tensor = self.self_attention(tensor, bias_tensor, mid_layer_name)
            tensor = self.layer_norm(tensor, 'encoder/' + mid_layer_name + '/attention/output')
            tensor = self.fully_connected(tensor, mid_layer_name)
            tensor = self.layer_norm(tensor, 'encoder/' + mid_layer_name + '/output')
        logits = self.pooler_loss(tensor)
        return logits

    def fully_connected(self, input_tensor, layer_name):
        inter_kernel = self.weights_dict['bert/encoder/{}/intermediate/dense/kernel:0'.format(layer_name)]
        inter_bias = self.weights_dict['bert/encoder/{}/intermediate/dense/bias:0'.format(layer_name)]
        out_kernel = self.weights_dict['bert/encoder/{}/output/dense/kernel:0'.format(layer_name)]
        out_bias = self.weights_dict['bert/encoder/{}/output/dense/bias:0'.format(layer_name)]
        with tf.name_scope('bert/encoder/{}/fully_connected/intermediate/dense'.format(layer_name)):
            matmul = tf.matmul(input_tensor, inter_kernel.astype(self.dtype.as_numpy_dtype))
            bias_add = tf.nn.bias_add(matmul, inter_bias.astype(self.dtype.as_numpy_dtype))
            gelu = self.gelu_sigmoid(bias_add) if self.crude_gelu else self.gelu_tanh(bias_add)
        with tf.name_scope('bert/encoder/{}/fully_connected/output/dense'.format(layer_name)):
            matmul = tf.matmul(gelu, out_kernel.astype(self.dtype.as_numpy_dtype))
            bias_add = tf.nn.bias_add(matmul, out_bias.astype(self.dtype.as_numpy_dtype))
            output_tensor = bias_add + input_tensor
        return output_tensor

    def self_attention(self, input_tensor, bias_tensor, layer_name):
        query_kernel = self.weights_dict['bert/encoder/{}/attention/self/query/kernel:0'.format(layer_name)] * 0.125
        query_bias = self.weights_dict['bert/encoder/{}/attention/self/query/bias:0'.format(layer_name)] * 0.125
        key_kernel = self.weights_dict['bert/encoder/{}/attention/self/key/kernel:0'.format(layer_name)]
        key_bias = self.weights_dict['bert/encoder/{}/attention/self/key/bias:0'.format(layer_name)]
        value_kernel = self.weights_dict['bert/encoder/{}/attention/self/value/kernel:0'.format(layer_name)]
        value_bias = self.weights_dict['bert/encoder/{}/attention/self/value/bias:0'.format(layer_name)]
        output_kernel = self.weights_dict['bert/encoder/{}/attention/output/dense/kernel:0'.format(layer_name)]
        output_bias = self.weights_dict['bert/encoder/{}/attention/output/dense/bias:0'.format(layer_name)]
        with tf.name_scope('bert/encoder/{}/attention/self'.format(layer_name)):
            matmul = tf.matmul(input_tensor, query_kernel.astype(self.dtype.as_numpy_dtype))
            query = tf.nn.bias_add(matmul, query_bias.astype(self.dtype.as_numpy_dtype))
            query_r = tf.reshape(query, [self.batch_size, self.seq_len, self.num_heads, self.head_size])
            query_rt = tf.transpose(query_r, [0, 2, 1, 3])
            matmul = tf.matmul(input_tensor, key_kernel.astype(self.dtype.as_numpy_dtype))
            key = tf.nn.bias_add(matmul, key_bias.astype(self.dtype.as_numpy_dtype))
            key_r = tf.reshape(key, [self.batch_size, self.seq_len, self.num_heads, self.head_size])
            key_rt = tf.transpose(key_r, [0, 2, 1, 3])  # [b, n, l, h]
            query_key = tf.matmul(query_rt, key_rt, transpose_b=True)  # [b, n, lq, h] @ [b, n, lk, h] -> [b, n, lq, lk]
            bias_query_key = tf.add(query_key, bias_tensor)
            softmax_weights = tf.nn.softmax(bias_query_key)
            matmul = tf.matmul(input_tensor, value_kernel.astype(self.dtype.as_numpy_dtype))
            value = tf.nn.bias_add(matmul, value_bias.astype(self.dtype.as_numpy_dtype))
            value_r = tf.reshape(value, [self.batch_size, self.seq_len, self.num_heads, self.head_size])
            value_rt = tf.transpose(value_r, [0, 2, 3, 1])
            weighted_value_rt = tf.matmul(softmax_weights, value_rt, transpose_b=True)  # [b, n, lq, lk] @ [b, n, h, lv] -> [b, n, lq, h]
            weighted_value_r = tf.transpose(weighted_value_rt, [0, 2, 1, 3])  # [b, lq, n, h]
            weighted_value = tf.reshape(weighted_value_r, [self.batch_size * self.seq_len, self.hid_size])
        with tf.name_scope('bert/encoder/{}/attention/output'.format(layer_name)):
            matmul = tf.matmul(weighted_value, output_kernel.astype(self.dtype.as_numpy_dtype))
            unnorm_output = tf.nn.bias_add(matmul, output_bias.astype(self.dtype.as_numpy_dtype))
            output_tensor = tf.add(input_tensor, unnorm_output)
        return output_tensor

    def layer_norm(self, input_tensor, layer_name, force_float32=False):
        dtype = tf.float32 if force_float32 else self.layer_norm_dtype
        gamma = dtype.as_numpy_dtype(self.weights_dict['bert/{}/LayerNorm/gamma:0'.format(layer_name)])
        beta = dtype.as_numpy_dtype(self.weights_dict['bert/{}/LayerNorm/beta:0'.format(layer_name)])
        with tf.name_scope('bert/{}/LayerNorm'.format(layer_name)):
            input_tensor = tf.cast(input_tensor, dtype)
            mean = tf.reduce_mean(input_tensor, axis=[-1], keepdims=True, name='mean')
            residuals = tf.subtract(input_tensor, mean, name='residuals')
            var = tf.reduce_mean(residuals * residuals, axis=[-1], keepdims=True, name='var')
            rsqrt = tf.rsqrt(var + dtype.as_numpy_dtype(self.eps))
            norm_output = tf.multiply(residuals, rsqrt, name='normalized')
            output_tensor = norm_output * gamma + beta
            output_tensor = tf.cast(output_tensor, self.dtype)
        return output_tensor

    def pooler_loss(self, input_tensor):
        pooler_kernel = self.weights_dict['bert/pooler/dense/kernel:0']
        pooler_bias = self.weights_dict['bert/pooler/dense/bias:0']
        loss_kernel = self.weights_dict['output_weights:0'].T
        loss_bias = self.weights_dict['output_bias:0']
        with tf.name_scope('bert/pooler_loss'):
            reshape = tf.reshape(input_tensor, [self.batch_size, self.seq_len, self.hid_size])
            reshape_1 = tf.reshape(reshape[:, 0:1, :], [self.batch_size, self.hid_size])
            matmul = tf.matmul(reshape_1, pooler_kernel.astype(self.dtype.as_numpy_dtype))
            bias_add = tf.nn.bias_add(matmul, pooler_bias.astype(self.dtype.as_numpy_dtype))
            tanh = tf.tanh(bias_add)
            matmul = tf.matmul(tanh, loss_kernel.astype(self.dtype.as_numpy_dtype))
            output_tensor = tf.nn.bias_add(matmul, loss_bias.astype(self.dtype.as_numpy_dtype))
        return output_tensor

    def gelu_tanh(self, tensor):
        pow3 = 0.044714998453855515 * tensor * tensor * tensor + tensor
        shifted = (tf.tanh(0.7978845834732056 * pow3) + 1.0) * tensor
        return tf.multiply(shifted, 0.5)

    def gelu_sigmoid(self, tensor):
        return tf.sigmoid(1.702 * tensor) * tensor


if __name__ == '__main__':
    main()
