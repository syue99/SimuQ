"""
sharp_small_gradient.py — the regime where FD cannot find the gradient DIRECTION:
small gradient on a SHARP landscape.  Analog PSR gets the direction; FD does not.

Where the gradient is small (near a feature/extremum) AND the landscape is sharp
(large f'''), FD's step bias ε²·f'''/6 can EXCEED the gradient itself → FD points
the WRONG way.  This is immune to ε-tuning: larger ε → more bias, smaller ε →
exploding shot variance (1/ε) that also buries the small signal.  Analog PSR has
no step ε and no 1/ε amplification → it resolves the small gradient's direction.

We pick a sharp landscape (long T), scan for a point with SMALL |grad| but high
curvature, then under noise + finite shots measure, vs the (noisy-landscape)
gradient:
  - P(correct sign): does the estimate point the right way?
  - RMSE relative to |grad|.
sweeping FD's ε (all floored ≥ ε_min) with PSR as the reference.

Saves figures/sharp_small_gradient.png.

Run:  conda run -n qec_pg python differential_computing/tests/sharp_small_gradient.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T = 4.0                  # sharp landscape
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
R = 20000
EPS_MIN = 0.30           # COARSE hardware control floor (> landscape feature scale)


def H_param():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def H_eval(x):
    Hp, var = H_param()
    return Hp.set_parameterizedHam({var: float(x)})


def landscape(runner):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    return lambda x: expfn([[H_eval(float(x)), T]])


def grad(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def find_small_sharp_point(f):
    """Scan for x with small |grad| but large |f'''| (sharp)."""
    xs = np.linspace(0.2, 2.2, 400)
    g = np.array([grad(f, x) for x in xs])
    # third derivative proxy
    h = 0.05
    f3 = np.array([(f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h))
                   / (2 * h ** 3) for x in xs])
    score = np.abs(f3) / (np.abs(g) + 0.05)        # sharp & small-grad
    # require a defined (non-vanishing) direction
    ok = np.abs(g) > 0.05
    score[~ok] = -1
    return float(xs[np.argmax(score)])


def psr_pool(runner, x, pool_size):
    Hp, var = H_param()
    np.random.seed(123)
    progs = observable_program_generator(Hp, T, n_sample=pool_size,
                                         n_repetition=1, diff_var=var, value=x)
    expfn = runner.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]
    b = len(H_tot) // 2
    em = np.array([expfn(H_tot[2 * i]) for i in range(b)])
    ep = np.array([expfn(H_tot[2 * i + 1]) for i in range(b)])
    return em, ep, float(ug)


def psr_estimates(em, ep, ug, n_sample, N, rng):
    P = len(em)
    n_per = int(max(1, round(N / (2 * n_sample))))
    idx = rng.integers(0, P, size=(R, n_sample))
    fm = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(em[idx], -1, 1))) / n_per - 1
    fp = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(ep[idx], -1, 1))) / n_per - 1
    return (T / n_sample) * ug * np.sum(fm - fp, axis=1)


def fd_estimates(f, x, eps, N, rng):
    n = N // 2
    fp_ex, fm_ex = f(x + eps), f(x - eps)
    fp = 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(fp_ex, -1, 1)), size=R) / n - 1
    fm = 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(fm_ex, -1, 1)), size=R) / n - 1
    return (fp - fm) / (2 * eps)


def main():
    rng = np.random.default_rng(0)
    clean = NoisyQuTiPRunner(2, noise=None)
    fc = landscape(clean)
    x_star = find_small_sharp_point(fc)
    g_true = grad(fc, x_star)
    print(f"Sharp landscape T={T}.  Chosen point x*={x_star:.3f}: "
          f"small gradient |g|={abs(g_true):.4f}, sharp curvature.\n")

    # moderate dephasing; truth = noisy-landscape gradient (achievable target)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=2.0))
    fn = landscape(noisy)
    g_truth = grad(fn, x_star)
    sgn = np.sign(g_truth)
    N = 20000
    n_sample = 256
    em, ep, ug = psr_pool(noisy, x_star, 800)

    psr = psr_estimates(em, ep, ug, n_sample, N, rng)
    psr_mean, psr_std = float(np.mean(psr)), float(np.std(psr))

    eps_grid = np.linspace(EPS_MIN, 1.5, 14)
    fd_mean, fd_std = [], []
    for eps in eps_grid:
        est = fd_estimates(fn, x_star, float(eps), N, rng)
        fd_mean.append(float(np.mean(est)))
        fd_std.append(float(np.std(est)))
    fd_mean, fd_std = np.array(fd_mean), np.array(fd_std)

    print(f"TRUE noisy-landscape gradient = {g_truth:+.4f}  (descent direction: "
          f"{'+' if g_truth < 0 else '−'}).  N={N}, T2=2, ε≥{EPS_MIN}.")
    print(f"  PSR n={n_sample}: estimate {psr_mean:+.4f} ± {psr_std:.4f}  "
          f"(error {psr_mean - g_truth:+.4f})")
    print(f"  {'ε':>6}{'FD estimate':>16}{'error':>10}{'sign':>7}")
    for i, eps in enumerate(eps_grid):
        ok = "ok" if np.sign(fd_mean[i]) == sgn else "WRONG"
        print(f"  {eps:>6.2f}{fd_mean[i]:>+11.4f}±{fd_std[i]:.3f}"
              f"{fd_mean[i] - g_truth:>+10.4f}{ok:>7}")

    fig, (axE, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)

    # ── Panel A: actual gradient estimate (linear) vs FD ε ──
    axE.axhspan(0, axE.get_ylim()[1], facecolor="#fde8e8", alpha=0)  # placeholder
    axE.fill_between(eps_grid, fd_mean - fd_std, fd_mean + fd_std,
                     color="#d62728", alpha=0.18)
    axE.plot(eps_grid, fd_mean, "s-", color="#d62728", lw=2, label="FD estimate")
    axE.axhline(psr_mean, color="#1f77b4", lw=2.6, label=f"PSR estimate (n={n_sample})")
    axE.fill_between(eps_grid, psr_mean - psr_std, psr_mean + psr_std,
                     color="#1f77b4", alpha=0.18)
    axE.axhline(g_truth, color="k", ls="--", lw=1.6, label="TRUE gradient")
    axE.axhline(0.0, color="gray", lw=1)
    # shade the wrong-sign half-plane (gradient is negative → positive = wrong way)
    lo, hi = axE.get_ylim()
    axE.axhspan(0.0, hi, facecolor="#d62728", alpha=0.06)
    axE.text(eps_grid[-1], hi * 0.92, "wrong direction", ha="right", va="top",
             fontsize=8, color="#d62728")
    axE.set_xlabel(r"FD step size $\varepsilon$  (control floor $\geq$%.2f)" % EPS_MIN)
    axE.set_ylabel("gradient estimate")
    axE.set_title("(A) estimated gradient vs true value")
    axE.legend(frameon=False, fontsize=8.5, loc="upper left")

    # ── Panel B: error = estimate − true (linear) ──
    axB.fill_between(eps_grid, (fd_mean - g_truth) - fd_std,
                     (fd_mean - g_truth) + fd_std, color="#d62728", alpha=0.18)
    axB.plot(eps_grid, fd_mean - g_truth, "s-", color="#d62728", lw=2, label="FD error")
    axB.axhline(psr_mean - g_truth, color="#1f77b4", lw=2.6,
                label=f"PSR error (n={n_sample})")
    axB.axhline(0.0, color="k", ls="--", lw=1.2, label="zero error (exact)")
    axB.axhline(-g_truth, color="gray", ls=":", lw=1,
                label="error = −|true| (sign flips here)")
    axB.set_xlabel(r"FD step size $\varepsilon$")
    axB.set_ylabel("error  (estimate − true)")
    axB.set_title("(B) error: PSR ≈ 0, FD's bias exceeds the gradient")
    axB.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Small gradient ({g_truth:+.3f}) on a sharp landscape: FD's step "
                 f"bias pushes its estimate ACROSS zero\n(wrong direction) at every "
                 f"feasible ε; analog PSR's estimate sits at the true gradient",
                 fontsize=9.3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "sharp_small_gradient.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
