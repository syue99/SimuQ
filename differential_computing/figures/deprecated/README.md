# Deprecated figures

- `psr_fd_device_gradient_p15.png`, `psr_fd_device_gradient_p50.png` —
  single-point device-gradient comparisons. **Superseded by**
  `../psr_fd_device_gradient_multipt.png`. They were evaluated at one operating
  point that sat near a gradient zero, where the `1/g_ideal` instability plus the
  `|g|` scaling faked a γ-dependence in the FD error. The multi-point
  RELATIVE-error version shows the true, γ-INDEPENDENT δ/ε control-resolution
  effect (the FD disadvantage is control resolution, not decoherence).

- `psr_fd_device_descent_p15.png` (residual-pinned least-squares: self-consistent
  targets pin the fixed point at theta* for BOTH estimators, so it doesn't isolate
  the FD failure) and `psr_fd_vqe_descent_p50.png` (linear VQE, honest but modest,
  min on the box edge). The finite-shot panel (`../psr_fd_device_finite_shot.png`)
  subsumes the descent point (PSR converges, FD floors), so these are retired.
