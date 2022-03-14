# -*- coding: utf-8 -*-

"""
neuronperf.tensorflow
~~~~~~~~~~~~~~~~~~~~~~~
Provides TensorFlow support.
"""

import itertools
import logging
import os
import threading


from .. import benchmarking


log = logging.getLogger(__name__)
_lock = threading.Lock()


def _load_fn(model_file, **kwargs):
    with _lock:
        import tensorflow as tf

        if tf.__version__.startswith("1"):
            return tf.contrib.predictor.from_saved_model(model_file)
        else:
            import tensorflow.keras as keras

            return keras.models.load_model(model_file)


def _compile_fn(model, inputs, models_dir, model_name, **kwargs):
    import tensorflow as tf
    import tensorflow.neuron as tfn

    model_filename = os.path.join(models_dir, model_name)

    # NeuronPerf provides compiler_args as a dictionary, but framework expects a different format.
    compiler_args = kwargs.pop("compiler_args", {})

    if tf.__version__.startswith("1"):
        compiler_args_flattened = list(itertools.chain.from_iterable(compiler_args.items()))
        kwargs["compiler_args"] = compiler_args_flattened
        kwargs["model_feed_dict"] = inputs

        # For TF 1.x, the saved model path is expected instead of a loaded model.
        tfn.saved_model.compile(model, model_filename, **kwargs)
    else:
        if compiler_args:
            compiler_args_flattened = " ".join(
                ["{}={}".format(k, v) for k, v in compiler_args.items()]
            )
            os.environ["NEURON_CC_FLAGS"] = compiler_args_flattened
        else:
            os.environ["NEURON_CC_FLAGS"] = ""

        model_neuron = tfn.trace(model, inputs, **kwargs)
        model_neuron.save(model_filename)
    return model_filename


def compile(model, inputs, *args, **kwargs):
    return benchmarking.compile(_compile_fn, model, inputs, *args, **kwargs)


def benchmark(model_filename, inputs, *args, **kwargs):
    # Tensorflow-neuron is not currently fork safe, so we workaround this during benchmarking
    # by spawning a fresh interpreter session for each model we benchmark.
    if "multiinterpreter" in kwargs and not kwargs["multiinterpreter"]:
        log.warning(
            "Setting multiinterpreter=False is not safe with TensorFlow. Use at your own risk."
        )
    else:
        kwargs["multiinterpreter"] = True

    return benchmarking.benchmark(_load_fn, model_filename, inputs, *args, **kwargs)
