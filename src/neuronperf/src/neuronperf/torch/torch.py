# -*- coding: utf-8 -*-

"""
neuronperf.torch
~~~~~~~~~~~~~~~~~~~~~~~
Provides PyTorch support.
"""

import functools
import itertools
import logging
import math
import os
import types

import torch

from .. import benchmarking


log = logging.getLogger(__name__)


def _compile_fn(model, example_inputs, models_dir, model_name, **kwargs):
    import torch_neuron

    """Compiles a model for Neuron."""
    model_filename = os.path.join(models_dir, "{}.pt".format(model_name))
    model.eval()

    # NeuronPerf provides compiler_args as a dictionary, but framework expects a different format.
    compiler_args = kwargs.get("compiler_args", {})
    compiler_args_flattened = list(itertools.chain.from_iterable(compiler_args.items()))
    kwargs["compiler_args"] = compiler_args_flattened

    model_neuron = torch.neuron.trace(
        model,
        example_inputs,
        **kwargs,
    )
    model_neuron.save(model_filename)
    return model_filename


def _load_fn(model_filename, **kwargs):
    import torch_neuron

    model = torch.jit.load(model_filename)
    model.eval()
    return model


def _class_load_fn(model_class, **kwargs):
    model = model_class()
    model.eval()
    return model


def compile(model, inputs, *args, **kwargs):
    return benchmarking.compile(_compile_fn, model, inputs, *args, **kwargs)


# See: https://pytorch.org/docs/stable/data.html#dataset-types
def _get_dataset_loader_fn(dataset, loop):
    def _worker_init_fn(worker_id):
        # This function will be called for each worker by torch.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        dataset = worker_info.dataset  # the dataset copy in this worker process
        per_worker = int(math.ceil(len(dataset) / float(num_workers)))
        start = worker_id * per_worker
        end = min(start + per_worker, len(dataset))
        log.debug(
            "worker_id={}, num_workers={}, per_worker={}, start={}, end={}".format(
                worker_id, num_workers, per_worker, start, end
            )
        )

        # We monkey-patch the dataset __iter__ function to support a multi-worker config.
        def _iter(self, start, end, loop):
            if loop:
                return itertools.cycle(range(start, end))
            else:
                return iter(range(start, end))

        __iter__ = functools.partial(_iter, start, end, loop)
        dataset.__iter__ = types.MethodType(__iter__, dataset)

    def dataset_loader_fn(dataset, num_workers):
        return iter(
            torch.utils.data.DataLoader(
                dataset, num_workers=num_workers, worker_init_fn=_worker_init_fn
            )
        )

    return dataset_loader_fn


def benchmark(model_filename, inputs, *args, dataset_inputs=False, loop_dataset=False, **kwargs):
    # These functions may need to be overridden or wrapped, depending upon config requested.
    load_fn = _load_fn
    setup_fn = kwargs.get("setup_fn", lambda *args, **kwargs: None)
    preprocess_fn = kwargs.get("preprocess_fn", lambda *args: (*args,))

    # If cuda is requested, ensure it's available and provide smart wrappers for CUDA device loading.
    device_type = kwargs.get("device_type", None)
    use_cuda = device_type and ("cuda" in device_type.lower() or "gpu" == device_type.lower())
    if use_cuda:
        if not torch.cuda.is_available():
            raise ValueError(
                "You requested CUDA benchmarking, but torch is unable to locate a CUDA device."
            )

        # Must use multiinterpreter for CUDA.
        if "multiinterpreter" in kwargs and not kwargs["multiinterpreter"]:
            log.warning(
                (
                    "You set multiinterpreter to False, but it is required for safe CUDA benchmarking.\n"
                    "Your preference has been overridden so that benchmarking may continue."
                )
            )
        kwargs["multiinterpreter"] = True

        # If we received a non-string, use class-based load function
        if not isinstance(model_filename, str):
            # In GPU benchmarking, a model class is expected. This line is for clarity.
            model_class = model_filename
            if not isinstance(model_class, type):
                raise TypeError("GPU benchmarking expects a model class to be provided instead of a filename.")

            # We must also know the name of the file to import from, so that serialization can succeed.
            import inspect

            try:
                model_class_file = inspect.getfile(model_class)
                kwargs["model_class_file"] = model_class_file
                kwargs["model_class_name"] = model_class.__name__
            except:
                raise ValueError(
                    (
                        "Your model class must be defined in a Python module so that it can be serialized properly.\n"
                        "Please add your model to a simple Python file along with any required imports."
                    )
                )

            @functools.wraps(_class_load_fn)
            def load_fn(*args, **kwargs):
                return _class_load_fn(model_class, **kwargs)

            # Now swap the class object for its name so the benchmarker still receives a string.
            model_filename = model_class.__name__

        # Wrap setup_fn so that it moves the model to CUDA device.
        @functools.wraps(setup_fn)
        def _setup_fn(id, config, model):
            setup_fn(id, config, model)
            model.to("cuda")

        kwargs["setup_fn"] = _setup_fn

        # Wrap preprocess_fn with one that moves inputs to CUDA.
        @functools.wraps(preprocess_fn)
        def _preprocess_fn(*inputs):
            inputs = preprocess_fn(*inputs)
            for input in inputs:
                input.to("cuda")
            return (*inputs,)

        kwargs["preprocess_fn"] = _preprocess_fn

    # When custom datasets are used, a loader function will need to be available in subprocesses.
    dataset_loader_fn = None
    if dataset_inputs:
        dataset_loader_fn = _get_dataset_loader_fn(example_inputs, loop_dataset)
    kwargs["dataset_loader_fn"] = dataset_loader_fn

    with torch.no_grad():
        return benchmarking.benchmark(
            load_fn,
            model_filename,
            inputs,
            *args,
            **kwargs,
        )
