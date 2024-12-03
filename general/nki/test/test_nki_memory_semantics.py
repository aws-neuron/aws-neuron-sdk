import unittest
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np

# NKI_EXAMPLE_0_BEGIN
@nki.jit(mode='simulation')
def simple_demo_kernel(a_ptr):
  
  B, N, M = a_ptr.shape

  a_loaded = nl.ndarray((B, nl.par_dim(N), M), dtype=a_ptr.dtype, buffer=nl.sbuf)
  exp_out =  nl.ndarray((B, nl.par_dim(N), M), dtype=a_ptr.dtype, buffer=nl.sbuf)
  out_ptr = nl.ndarray((B, nl.par_dim(N), M), dtype=a_ptr.dtype, buffer=nl.shared_hbm)

  for b in nl.affine_range(B):
    a_loaded[b] = nl.load(a_ptr[b])
    exp_out[b] = nl.exp(a_loaded[b])
    nl.store(out_ptr[b], value=exp_out[b])

  return out_ptr
# NKI_EXAMPLE_0_END

class TestNkiMemorySemantics(unittest.TestCase):
  def test_simulate_kernel(self):
    np.random.seed(0)
    a = np.random.random_sample([4, 128, 512]).astype(np.float32)
    
    result = simple_demo_kernel(a)

    self.assertTrue(np.allclose(result, np.exp(a)))
