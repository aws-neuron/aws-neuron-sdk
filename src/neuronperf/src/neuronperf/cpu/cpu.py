# -*- coding: utf-8 -*-

"""
neuronperf.cpu
~~~~~~~~~~~~~~~~~~~~~~~
Provides CPU support.
"""

import functools
import logging

from .. import benchmarking


log = logging.getLogger(__name__)


class DummyModel:
    def __call__(self, x):
        x *= 5
        x += 3
        return x


def benchmark(model_class, inputs, *args, **kwargs):
    if not isinstance(model_class, type):
        raise TypeError("For CPU benchmarking, you must provide a class to instantiate.")

    device_type = kwargs.pop("device_type", "cpu")
    multiinterpreter = kwargs.pop("multiinterpreter", False)
    if multiinterpreter:
        log.warning(
            "CPU + multiinterpreter is not yet fully supported. You need to provide a custom load_fn that can import your class and instantiate it."
        )

    # Create a custom load_fn that instantiates the model.
    def load_fn(*args, **kwargs):
        return model_class()

    kwargs["device_type"] = device_type
    kwargs["multiinterpreter"] = multiinterpreter

    return benchmarking.benchmark(
        load_fn,
        model_class.__name__,
        inputs,
        *args,
        **kwargs,
    )
