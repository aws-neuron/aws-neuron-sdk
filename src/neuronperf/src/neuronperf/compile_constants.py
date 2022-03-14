# -*- coding: utf-8 -*-

"""
neuronperf.compile_constants
~~~~~~~~~~~~~~~~~~~~~~~
Holds constants used at compile time.
"""

NEURONCORE_PIPELINE_CORES = "--neuroncore-pipeline-cores"
FAST_MATH = "--fast-math"
FAST_MATH_OPTIONS = {
    0: "none",
    1: "fp32-cast-matmult no-fast-relayout",
    2: "fp32-cast-matmult",
    3: "all",
}
