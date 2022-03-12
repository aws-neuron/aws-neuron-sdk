# -*- coding: utf-8 -*-

"""
neuronperf.mxnet
~~~~~~~~~~~~~~~~~~~~~~~
Provides Apache MXNet (Incubating) support.
"""

import contextlib
import functools
import os
import threading

# handle different API versions of mxnet
import mxnet as mx
from distutils.version import LooseVersion

if LooseVersion(mx.__version__) >= LooseVersion("1.8"):
    _mx_version = 1.8
    import mx_neuron as neuron
else:
    _mx_version = 1.5
    from mxnet.contrib import neuron

from .. import benchmarking


class _MXNetModelWrapper:
    def __init__(self, device_id, sym, args, aux):
        self.device_id = device_id
        self.sym = sym
        self.args = args
        self.aux = aux
        self.ctx = None
        self.exes = {}
        self.lock = threading.Lock()

    def __call__(self, inputs):
        # on the first inference, do prep work
        if not self.ctx:
            self.ctx = mx.neuron(self.device_id)

        # prepare inputs for model
        for k, v in inputs.items():
            inputs[k] = mx.nd.array(v)
        self.args.update(inputs)

        # obtain an executor for this thread
        thread_id = threading.get_ident()
        if thread_id not in self.exes:
            with self.lock:
                exe = self.sym.bind(
                    ctx=self.ctx, args=self.args, aux_states=self.aux, grad_req="null"
                )
            self.exes[thread_id] = exe
        else:
            exe = self.exes[thread_id]

        # run inference
        outputs = exe.forward(**inputs)
        mx.nd.waitall()
        return outputs[0]


@contextlib.contextmanager
def change_dir(new_dir):
    old_dir = os.getcwd()
    os.chdir(os.path.join(old_dir, new_dir))
    try:
        yield
    finally:
        os.chdir(old_dir)


def _load_fn(model_filename, **kwargs):
    device_id = kwargs.get("device_id", 0)
    sym, args, aux = mx.model.load_checkpoint(model_filename, 0)
    return _MXNetModelWrapper(device_id, sym, args, aux)


def _compile_fn(model, example_inputs, models_dir, model_name, **kwargs):
    _sym, _args, _aux = model
    model_filename = os.path.join(models_dir, model_name)
    compiler_args = kwargs.pop("compiler_args", {})

    # MXNet passes additional kwargs directly to compiler
    _sym, _args, _aux = neuron.compile(
        _sym,
        _args,
        _aux,
        example_inputs,
        **compiler_args,
    )

    with change_dir(models_dir):
        mx.model.save_checkpoint(model_name, 0, _sym, _args, _aux)
    return model_filename


def compile(model, inputs, *args, **kwargs):
    return benchmarking.compile(_compile_fn, model, inputs, *args, **kwargs)


def benchmark(model_filename, inputs, *args, **kwargs):
    env_setup_fn = kwargs.pop("env_setup_fn", lambda *_: None)

    # Use a custom setup function to handle MXNet concurrency requirements.
    @functools.wraps(env_setup_fn)
    def _env_setup_fn(id, config):
        workers_per_model = str(config["workers_per_model"])
        os.environ["MXNET_CPU_TEMP_COPY"] = workers_per_model
        os.environ["MXNET_EXEC_NUM_TEMP"] = workers_per_model
        os.environ["MXNET_CPU_WORKER_NTHREADS"] = workers_per_model
        os.environ["MXNET_MP_WORKER_NTHREADS"] = workers_per_model

        # Remember to call any additional custom setup provided.
        env_setup_fn(id, config)

    kwargs["env_setup_fn"] = _env_setup_fn

    return benchmarking.benchmark(_load_fn, model_filename, inputs, *args, **kwargs)
