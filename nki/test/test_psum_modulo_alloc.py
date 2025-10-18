# NKI_EXAMPLE_0_BEGIN
from typing import Optional, Tuple
from functools import reduce
from operator import mul
import unittest

def num_elems(shape):
  return reduce(mul, shape, 1)

def linearize(shape, indices):
  return sum(i * num_elems(shape[dim+1:]) for dim, i in enumerate(indices))

def modulo_allocate_func(base, allocate_shape, scale):
  def func(indices):
    if not allocate_shape:
      # default shape is always (1, 1, ...)
      allocate_shape_ = (1, ) * len(indices)
    else:
      allocate_shape_ = allocate_shape
    mod_idx = tuple(i % s for i, s in zip(indices, allocate_shape_))
    return linearize(shape=allocate_shape_, indices=mod_idx) * scale + base
  return func

def mod_alloc(base_addr: int, *, 
               base_bank: Optional[int] = 0,
               num_bank_tiles: Optional[Tuple[int]] = (),
               base_partition: Optional[int] = 0,
               num_par_tiles: Optional[Tuple[int]] = (),
               num_free_tiles: Optional[Tuple[int]] = ()):
  def psum_modulo_alloc_func(idx, pdim_size, fdim_size):
    # partial bank allocation is not allowed
    return (modulo_allocate_func(base_bank, num_bank_tiles, 1)(idx),
          modulo_allocate_func(base_partition, num_par_tiles, pdim_size)(idx),
          modulo_allocate_func(base_addr, num_free_tiles, fdim_size)(idx))
  return psum_modulo_alloc_func

# NKI_EXAMPLE_0_END

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
import numpy as np
nki_jit = nki.trace


@nki_jit
def allocated_loop_transpose(a_ptr, tp_ptr):
  
  N, M = a_ptr.shape

  _M, _N = tp_ptr.shape
  assert _N == N and _M == M

  N0, N1 = N // 128, 128
  M0, M1 = M // 128, 128

  ix0 = nl.arange(0, M1)[:, None]
  iy0 = nl.arange(0, N1)[None, :]

  identity = nl.shared_constant(np.identity(n=128, dtype=np.int8), dtype=nl.bfloat16)

  for n0 in nl.affine_range(N0):
    for m0 in nl.affine_range(M0):
      ix0 = nl.arange(0, 128)[:, None]
      iy0 = nl.arange(0, 128)[None, :]
      a_local = nl.ndarray((nl.par_dim(N1), M1), dtype=a_ptr.dtype, 
                           buffer=ncc.sbuf.mod_alloc(base_addr=1024))
      a_local[ix0, iy0] = nl.load(a_ptr[n0 * N1 + ix0, m0 * M1 + iy0])

      identity_load = nl.ndarray((nl.par_dim(128), 128), dtype=a_ptr.dtype, buffer=ncc.sbuf.mod_alloc(base_addr=0))
      identity_load[ix0, iy0] = nl.load(identity, dtype=a_ptr.dtype)

      a_local_transpose = nl.ndarray((nl.par_dim(M1), N1), dtype=a_ptr.dtype,
                                     buffer=ncc.psum.alloc(mod_alloc(base_addr=0)))
      a_local_transpose[ix0, iy0] = nisa.nc_matmul(a_local[ix0, iy0], identity_load)

      a_t_sbuf = nl.ndarray((nl.par_dim(N1), M1), dtype=a_ptr.dtype,
                                     buffer=ncc.sbuf.mod_alloc(base_addr=2048))
      a_t_sbuf[ix0, iy0] = nl.copy(a_local_transpose[ix0, iy0])

      nl.store(tp_ptr[m0 * 128 + ix0, n0 * 128 + iy0], value=a_t_sbuf[ix0, iy0])

class TestNkiPSUMModuloAllocation(unittest.TestCase):
  def test_simulate_kernel(self):
    np.random.seed(0)
    a = np.random.random_sample([2048, 1024]).astype(np.float32) * 100
    b = np.ndarray(shape=(1024, 2048), dtype=np.float32)

    nki.simulate_kernel(allocated_loop_transpose, a, b)

    self.assertTrue(np.allclose(b, np.transpose(a)))

