# bert_no_model.py
import argparse
import tensorflow as tf
import tensorflow.neuron as tfn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_saved_model', required=True, help='Original SaveModel')
    parser.add_argument('--output_saved_model', required=True, help='Output SavedModel that runs on Inferentia')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sequence_length', type=int, default=128)
    args = parser.parse_args()
    pred = tf.contrib.predictor.from_saved_model(args.input_saved_model)
    model_shape_feed_dict = {}
    for key, value in pred.feed_tensors.items():
        shape = value.shape.as_list()
        if len(shape) > 0:
            shape[0] = args.batch_size
        if len(shape) > 1:
            shape[1] = args.sequence_length
        model_shape_feed_dict[key] = shape
    no_fuse_ops = [op.name for op in pred.graph.get_operations()]

    def force_fuse_condition(op_name):
        exclude_scopes = [
            'bert/encoder/strided_slice',
            'bert/encoder/ones',
            'bert/encoder/Reshape',
            'bert/encoder/Shape',
            'bert/encoder/Cast',
        ]
        for scope in exclude_scopes:
            if op_name == scope or op_name.startswith('{}/'.format(scope)):
                return False
        return op_name.startswith('bert/encoder') or op_name.startswith('bert/pooler')

    force_fuse_ops = [op.name for op in pred.graph.get_operations() if force_fuse_condition(op.name)]
    compilation_result = tfn.saved_model.compile(
        args.input_saved_model, args.output_saved_model,
        model_shape_feed_dict=model_shape_feed_dict,
        compiler_args=['-O2'],
        compiler_timeout=36000,
        no_fuse_ops=no_fuse_ops,
        force_fuse_ops=force_fuse_ops,
        minimum_segment_size=20,
        dynamic_batch_size=True,
    )
    print(compilation_result)


if __name__ == '__main__':
    main()
