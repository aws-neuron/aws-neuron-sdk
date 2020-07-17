"""
Usage: python convert_graph_opt.py /path/to/graph_opt.pb /path/to/graph_opt_neuron.pb
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
import tensorflow.neuron as tfn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_pb_path', help='Input serialized GraphDef protobuf')
    parser.add_argument('output_pb_path', help='Ouput serialized GraphDef protobuf')
    parser.add_argument('--net_resolution', default='656x368', help='Network resolution in WxH format, e. g., --net_resolution=656x368')
    parser.add_argument('--debug_verify', action='store_true')
    args = parser.parse_args()
    dim_w, dim_h = args.net_resolution.split('x')
    dim_w = int(dim_w)
    dim_h = int(dim_h)
    graph_def = tf.GraphDef()
    with open(args.input_pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    if args.debug_verify:
        np.random.seed(0)
        feed_dict = {'image:0': np.random.rand(1, dim_h, dim_w, 3)}
        output_name = 'Openpose/concat_stage7:0'
        with tf.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name='')
            result_reference = sess.run(output_name, feed_dict)

    preprocessing_ops = {'preprocess_divide', 'preprocess_divide/y', 'preprocess_subtract', 'preprocess_subtract/y'}
    graph_def = nhwc_to_nchw(graph_def, preprocessing_ops)
    graph_def = inline_float32_to_float16(graph_def, preprocessing_ops)
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name='')
        no_fuse_ops = preprocessing_ops.union({'Openpose/concat_stage7'})
        infer_graph = tfn.graph_util.inference_graph_from_session(
            sess, shape_feed_dict={'image:0': [1, dim_h, dim_w, 3]}, output_tensors=['Openpose/concat_stage7:0'],
            no_fuse_ops=no_fuse_ops, compiler_args=['-O2'], dynamic_batch_size=True,
        )
    with open(args.output_pb_path, 'wb') as f:
        f.write(infer_graph.as_graph_def().SerializeToString())

    if args.debug_verify:
        with tf.Session(graph=infer_graph) as sess:
            result_compiled = sess.run(output_name, feed_dict)
        np.testing.assert_allclose(result_compiled, result_reference, rtol=1e-2, atol=1e-3)


def inline_float32_to_float16(graph_def, preprocessing_ops):
    float32_enum = tf.float32.as_datatype_enum
    float16_enum = tf.float16.as_datatype_enum
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
        if node.name in preprocessing_ops or node.op == 'Placeholder':
            cast_input_node_name = node.name
            continue
        if node.op == 'Const':
            if node.attr['dtype'].type == float32_enum:
                node.attr['dtype'].type = float16_enum
                tensor_def = node.attr['value'].tensor
                tensor_def.dtype = float16_enum
                if tensor_def.tensor_content:
                    const_np = np.frombuffer(tensor_def.tensor_content, dtype=np.float32).astype(np.float16)
                    tensor_def.tensor_content = const_np.tobytes()
                elif len(tensor_def.float_val):
                    const_np = np.array(tensor_def.float_val).astype(np.float16).view(np.uint16)
                    tensor_def.float_val[:] = []
                    tensor_def.half_val[:] = list(const_np)
                else:
                    raise NotImplementedError
        elif 'T' in node.attr and node.attr['T'].type == float32_enum:
            node.attr['T'].type = float16_enum
    for node in graph_def.node:
        if node.name == cast_input_node_name:
            node.name = '{}_PreCastFloat32ToFlot16'.format(node.name)
            input_node = node
            break
    cast_input_node = _gen_cast_node_def(cast_input_node_name, tf.float16, input_node)

    output_node = graph_def.node[-1]
    cast_output_node_name = output_node.name
    output_node.name = '{}_PreCastFloat16ToFlot32'.format(output_node.name)
    cast_output_node = _gen_cast_node_def(cast_output_node_name, tf.float32, output_node)

    preprocessing_ops.add(input_node.name)
    new_graph_def = tf.GraphDef()
    new_graph_def.node.extend(graph_def.node)
    new_graph_def.node.append(cast_input_node)
    new_graph_def.node.append(cast_output_node)
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(new_graph_def, name='')
    return graph.as_graph_def()


def nhwc_to_nchw(graph_def, preprocessing_ops):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    graph_def = graph.as_graph_def()
    node_name_to_node = {node.name: node for node in graph_def.node}
    for node in graph_def.node:
        if node.name in preprocessing_ops or node.op == 'Placeholder':
            transpose_input_node_name = node.name
            continue
        if node.op == 'Conv2D':
            node.attr['data_format'].s = b'NCHW'
            strides = node.attr['strides'].list.i
            strides[:] = [strides[0], strides[3], strides[1], strides[2]]
        elif node.op == 'BiasAdd':
            if node.name != 'probs/BiasAdd':
                node.attr['data_format'].s = b'NCHW'
        elif node.op == 'MaxPool':
            node.attr['data_format'].s = b'NCHW'
            ksize = node.attr['ksize'].list.i
            ksize[:] = [ksize[0], ksize[3], ksize[1], ksize[2]]
            strides = node.attr['strides'].list.i
            strides[:] = [strides[0], strides[3], strides[1], strides[2]]
        elif node.op in {'Concat', 'ConcatV2'}:
            node_axes = node_name_to_node[node.input[-1]]
            node_axes.attr['value'].tensor.int_val[:] = [1]
    for node in graph_def.node:
        if node.name == transpose_input_node_name:
            node.name = '{}_PreTransposeNHWC2NCHW'.format(node.name)
            input_node = node
            break
    transpose_input_node, transpose_input_perm_node = _gen_transpose_def(transpose_input_node_name, [0, 3, 1, 2], input_node)

    output_node = graph_def.node[-1]
    transpose_output_node_name = output_node.name
    output_node.name = '{}_PreTransposeNCHW2NHWC'.format(output_node.name)
    transpose_output_node, transpose_output_perm_node = _gen_transpose_def(transpose_output_node_name, [0, 2, 3, 1], output_node)

    preprocessing_ops.add(input_node.name)
    preprocessing_ops.add(transpose_input_perm_node.name)
    new_graph_def = tf.GraphDef()
    new_graph_def.node.extend(graph_def.node)
    new_graph_def.node.append(transpose_input_perm_node)
    new_graph_def.node.append(transpose_input_node)
    new_graph_def.node.append(transpose_output_perm_node)
    new_graph_def.node.append(transpose_output_node)
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(new_graph_def, name='')
    return graph.as_graph_def()


def _gen_cast_node_def(name, target_dtype, input_node):
    cast_node = tf.NodeDef(name=name, op='Cast')
    cast_node.input.append(input_node.name)
    cast_node.attr['DstT'].type = target_dtype.as_datatype_enum
    cast_node.attr['SrcT'].type = input_node.attr['T'].type
    cast_node.attr['Truncate'].b = False
    return cast_node


def _gen_transpose_def(name, perm, input_node):
    perm_node = tf.NodeDef(name='{}/perm'.format(name), op='Const')
    perm_node.attr['dtype'].type = tf.int32.as_datatype_enum
    tensor_def = perm_node.attr['value'].tensor
    tensor_def.dtype = tf.int32.as_datatype_enum
    tensor_def.tensor_shape.dim.append(TensorShapeProto.Dim(size=4))
    tensor_def.tensor_content = np.array(perm, dtype=np.int32).tobytes()
    transpose_node = tf.NodeDef(name=name, op='Transpose')
    transpose_node.input.append(input_node.name)
    transpose_node.input.append(perm_node.name)
    transpose_node.attr['T'].type = input_node.attr['T'].type
    transpose_node.attr['Tperm'].type = tf.int32.as_datatype_enum
    return transpose_node, perm_node


if __name__ == '__main__':
    main()
