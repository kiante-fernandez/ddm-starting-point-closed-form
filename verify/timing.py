"""Closed-form cost per evaluation, single thread, 10 000 RTs at one parameter set.
Same parameter set as verify/timing.R.  Needs the compiled kernel (cythonize -3 -i src/ddm_kernel.pyx)."""
import sys; sys.path.insert(0, "src")
import numpy as np, timeit
import ddm_kernel as K, ddm_fast as F
t = np.random.default_rng(1).uniform(0.1, 3, 10000)
a, v, w1, w2, sv, st0 = 1.2, 1.0, 0.4, 0.6, 1.0, 0.2
us = lambda fn: 1e6 * min(timeit.repeat(fn, number=1, repeat=5)) / len(t)
print(f"closed form, compiled  {'six-parameter density':38s} {us(lambda: K.g_full_sz(t, v, sv, a, w1, w2)):8.2f} us/eval")
print(f"closed form, numpy     {'six-parameter density + five gradients':38s} {us(lambda: F.grad_full_sz(t, v, sv, a, w1, w2)):8.2f} us/eval")
print(f"closed form, compiled  {'seven-parameter density':38s} {us(lambda: K.f7(t, v, sv, a, w1, w2, 0.0, st0)):8.2f} us/eval")
