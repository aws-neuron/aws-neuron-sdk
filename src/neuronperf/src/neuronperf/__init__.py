# -*- coding: utf-8 -*-

"""
NeuronPerf Library
~~~~~~~~~~~~~~~~~~

A library for benchmarking machine learning models on accelerators.

:copyright: (c) 2022 Amazon Inc.
:license: See LICENSE.
"""

from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __author__, __author_email__, __license__
from .__version__ import __copyright__

# setup logging first
import logging

_log_level = logging.DEBUG
log = logging.getLogger(__name__)
log.setLevel(_log_level)

from .logging import _get_stream_handlers

for handler in _get_stream_handlers(_log_level):
    log.addHandler(handler)

from .benchmarking import compile, benchmark, set_verbosity
from .cpu import cpu
from .cpu.cpu import DummyModel
from .reporting import CSV_COLS, PRINT_COLS, get_reports, print_reports, write_csv, write_json
from .timing import timestamp_convert, Timer
