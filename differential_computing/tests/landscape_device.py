"""
landscape_device.py — Fig 3 data (device-target FD trap WITH control error δ).

Device-target (no rescale): the estimand is the EXACT device gradient ∇C_noisy.
Raw PSR computes it exactly (theorem) and is ε-free.  Finite-difference is trapped
by BOTH arms of ε on the sharp noisy landscape:
  large ε  -> truncation: the secant of the (sharp) noisy landscape has the WRONG
             sign vs its own tangent ∇C_noisy;
  small ε  -> the setpoint control error δ (∼ resolution r) is amplified by 1/ε
             (a floor that finite shots CANNOT remove), on top of shot noise.
So no ε is reliable, while raw PSR sits at the shot-limited floor.

Cost = <Z0>_noisy(x) under dephasing, T/T2*=0.5.  Caches landscape_device_data.json.
Run: conda run -n qec_pg python differential_computing/tests/landscape_device.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
from simuq import QSystem, Qubit
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T, T2 = 2.5, 5.0                    # T/T2* = 0.5
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
N_SHOTS, R, N_SAMPLE, POOL = 9000, 5000, 48, 800
R_CTRL = 0.05                       # control resolution: floors ε and sets δ~N(0,r)
WRONG_FRAC = 0.20
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def Hsimuq():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def fmaker(runner):
    H, var = Hsimuq()
    expfn = runner.make_expectation_fn(PSI0, OBS)
    return lambda x: expfn([[H.set_parameterizedHam({"x": float(x)}), T]])


def compute():
    rng = np.random.default_rng(0)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    fn = fmaker(noisy)                                   # <Z0>_noisy(x)  (device landscape)

    def gnoisy(x, h=1e-3):
        return (fn(x + h) - fn(x - h)) / (2 * h)         # device gradient ∇C_noisy

    # pick x*: moderate device gradient AND large-ε FD-of-noisy sign-flips (truncation)
    xs = np.linspace(0.25, 1.55, 120)
    cands = []
    for x in xs:
        g = gnoisy(x)
        if 0.15 < abs(g) < 0.55:
            fd06 = (fn(x + 0.6) - fn(x - 0.6)) / 1.2
            if np.sign(fd06) != np.sign(g):
                cands.append((float(x), abs(fd06 - g)))
    cands.sort(key=lambda c: -c[1])
    x_star = cands[0][0] if cands else float(xs[np.argmin(np.abs(
        np.array([abs(gnoisy(x)) for x in xs]) - 0.3))])
    g_real = float(gnoisy(x_star)); sgn = np.sign(g_real)
    print(f"x*={x_star:.3f}  device gradient ∇C_noisy = {g_real:+.4f}")

    def shots(exact, n):
        return 2.0 * rng.binomial(int(max(1, n)),
                                  0.5 * (1 + np.clip(exact, -1, 1)), size=R) / max(1, n) - 1

    # FD RMSE vs ε, with shots + per-trial control error δ~N(0,r) on the ± setpoints
    eps_grid = np.geomspace(0.03, 1.4, 22)
    fd_rmse, fd_wrong = [], []
    for eps in eps_grid:
        dp = rng.normal(0, R_CTRL, R); dm = rng.normal(0, R_CTRL, R)
        vp = np.array([fn(x_star + eps + d) for d in dp[:400]])   # sub-sample δ evals
        vm = np.array([fn(x_star - eps + d) for d in dm[:400]])
        ip = rng.integers(0, len(vp), R); im = rng.integers(0, len(vm), R)
        n = N_SHOTS // 2
        fp = 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(vp[ip], -1, 1))) / n - 1
        fm = 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(vm[im], -1, 1))) / n - 1
        est = (fp - fm) / (2 * eps)
        fd_rmse.append(float(np.sqrt(np.mean((est - g_real) ** 2))))
        fd_wrong.append(float(np.mean(np.sign(est) != sgn)))

    # raw PSR = exact device gradient (ε-free); shot-sampled from a deterministic-τ pool
    H, var = Hsimuq()
    orig = np.random.rand; np.random.rand = lambda k: (np.arange(k) + 0.5) / k
    try:
        progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                             diff_var=var, value=x_star)
    finally:
        np.random.rand = orig
    pexp = noisy.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]; b = len(H_tot) // 2
    em = np.array([pexp(H_tot[2 * i]) for i in range(b)])
    ep = np.array([pexp(H_tot[2 * i + 1]) for i in range(b)])
    n_per = int(max(1, round(N_SHOTS / (2 * N_SAMPLE))))
    idx = rng.integers(0, len(em), size=(R, N_SAMPLE))
    fm = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(em[idx], -1, 1))) / n_per - 1
    fpb = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(ep[idx], -1, 1))) / n_per - 1
    raw = (T / N_SAMPLE) * float(ug) * np.sum(fm - fpb, axis=1)
    psr_rmse = float(np.sqrt(np.mean((raw - g_real) ** 2)))
    psr_slope = float(np.mean(raw))
    print(f"raw PSR (device grad) mean={psr_slope:+.4f} rmse={psr_rmse:.4f}; "
          f"FD best rmse={min(fd_rmse):.4f}")

    # landscape + noisy secants for panel A
    gx = np.linspace(0.0, 1.55, 150)
    y_noisy = [fn(x) for x in gx]
    z0 = fn(x_star)
    secants = [dict(eps=e, fm=fn(x_star - e), fp=fn(x_star + e)) for e in [0.15, 0.3, 0.45, 0.6]]
    # a representative small-ε realization corrupted by δ (wrong sign)
    sl_small = None
    for _ in range(40):
        d1, d2 = rng.normal(0, R_CTRL), rng.normal(0, R_CTRL)
        nn = N_SHOTS // 2
        fp = 2 * rng.binomial(nn, 0.5 * (1 + np.clip(fn(x_star + 0.05 + d1), -1, 1))) / nn - 1
        fmm = 2 * rng.binomial(nn, 0.5 * (1 + np.clip(fn(x_star - 0.05 + d2), -1, 1))) / nn - 1
        cand = (fp - fmm) / 0.10
        if np.sign(cand) != sgn:
            sl_small = float(cand); break
    if sl_small is None:
        sl_small = float(cand)

    return dict(T=T, T2=T2, N_SHOTS=N_SHOTS, r_ctrl=R_CTRL,
                x_star=x_star, g_real=g_real, z0=float(z0), sl_small=sl_small,
                psr_slope=psr_slope, psr_rmse=psr_rmse,
                gx=list(map(float, gx)), y_noisy=y_noisy, secants=secants,
                eps_grid=list(map(float, eps_grid)), fd_rmse=fd_rmse, fd_wrong=fd_wrong)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "landscape_device_data.json")
    d = compute()
    json.dump(d, open(cache, "w"), default=float)
    print(f"cached: {cache}")


if __name__ == "__main__":
    main()
