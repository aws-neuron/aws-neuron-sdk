"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki

# NKI_EXAMPLE_0_BEGIN NKI_EXAMPLE_1_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_0_END NKI_EXAMPLE_1_END

########################################################################
# NOTE: if you modify this file, make sure to update neuron_isa.py file with
# NOTE: the correct line numbers under .. nki_example:: directive
########################################################################

@nki.jit(mode="simulation")
# NKI_EXAMPLE_0_BEGIN
############################################################################
# Example 1: Simple 2D transpose (HBM->SB)
############################################################################
def nki_dma_transpose_2d_hbm2sb(a):
  b = nisa.dma_transpose(a)
  return b
# NKI_EXAMPLE_0_END

@nki.jit(mode="simulation")
# NKI_EXAMPLE_1_BEGIN
############################################################################
# Example 2: Simple 2D transpose (SB->SB)
############################################################################
def nki_dma_transpose_2d_sb2sb(a):
  a_sb = nl.load(a)
  b = nisa.dma_transpose(a_sb)
  return b
# NKI_EXAMPLE_1_END
