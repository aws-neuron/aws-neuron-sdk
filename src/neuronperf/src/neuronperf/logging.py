# -*- coding: utf-8 -*-

"""
neuronperf.logging
~~~~~~~~~~~~~~~~~~~~~~~
Provides logging utility functions.
"""

import logging


FORMAT_STRING = '%(levelname)s:%(name)s - %(message)s'


def _get_stream_handlers(level = logging.DEBUG):
    formatter = logging.Formatter(FORMAT_STRING)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    return [sh]
