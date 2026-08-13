"""
build_Floop_real.py — SEC6_FOLLOWUP C2: F-loop with the REAL estimators.

Replaces the surrogate F-loop (build_Floop.py, PSR/NSR = exact grad + Gaussian) which
C2 rejects for the headline loop. The descent is driven by the ACTUAL sampled estimators:
  - PSR : real kick branches from observable_program_generator, branch readouts through
          NoisyQuTiPRunner INCLUDING the T4 kick gate-error channel (so PSR carries its
          ~0.028 digital gate bias, C3), finite-shot sampled.
  - NSR : real stochastic (n, σ) Nyquist sampler along each component's tangent
          A_i = ∂_θi H, sampling the emulated noisy landscape. No inserted op → gate-error
          immune (C3). The sampler visits only 32 distinct shift points, computed once.
  - FD  : real noisy secant + control δ + shots (surrogate allowed only for the FD ε-grid
          side variants — here all FD arms are the real secant).

Cost O = (1/P) Σ_i Z_iZ_{i+1} ∈ [-1,1] (mean bond parity). All Z_iZ_{i+1} are DIAGONAL,
so one Z-basis shot draws a bitstring giving every bond at once: the finite-shot model
samples basis states ~ diag(ρ) and averages (1/P)ΣZZ — the correct model for a summed
diagonal cost (a single [-1,1] binomial cannot represent a [-P,P] sum).
Multi-parameter: for component i, fix the other couplings by partial set_parameterizedHam
(keeps t_i symbolic), then differentiate t_i.

PROBE mode times one gradient of each method and exits; use it to choose (seeds, iters, P)
under the wall-clock budget (reduce seeds → iters → P per C2), then set PROBE=False.
Run: conda run -n qec_pg python differential_computing/tests/build_Floop_real.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel
from observable_program_generator import observable_program_generator
from nyquist_shift import tangent_hamiltonian, bandwidth_K

# ── config ──
PROBE = False                # True: time one gradient of each method and exit
P = 4
NQ = P + 1
G_FIELD, T = 1.0, 1.5
T2 = T / 0.15
DELTA = 0.02                 # control setpoint error δ (T4 best-guess)
GATE_2Q = 1.0e-3             # T4 kick gate error — INCLUDED for PSR (C2/C3)
COH_FRAC = 0.5
B = 1000                     # quantum executions per gradient
N_POOL = 16                  # PSR deterministic-τ pool (branches)
MAXN = 16                    # NSR series truncation (n = 0..MAXN-1)
BOX = 2.5
LAM = 0.3                    # amplitude regularizer λ: C = ⟨O⟩ + λ/2·|θ|² (interior min,
                             # keeps NSR's dominant shift feasible). ∇reg = λθ is an EXACT
                             # classical add-on to the SAMPLED ∇⟨O⟩ (no shots) — same for all.
ITERS = int(os.environ.get("FLOOP_ITERS", 60))
SEEDS = int(os.environ.get("FLOOP_SEEDS", 20))
ETA = 0.25                   # learning rate (fixed; chosen from noiseless descent on C)
EPS_STAR = 0.25
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
C_PSR, C_NSR, C_FD = "#0072B2", "#009E73", "#D55E00"


def build(nq, p):
    qs = QSystem(); q = [Qubit(qs) for _ in range(nq)]
    syms = [sp.Symbol(f"t{i}") for i in range(p)]
    H = sum(syms[i] * q[i].Z * q[i + 1].Z for i in range(p)) + G_FIELD * sum(q[i].X for i in range(nq))
    return H, [str(s) for s in syms]


def zz_vector(nq, p):
    """(1/P)Σ_i z_i z_{i+1} for each of the 2^nq Z-basis states (qutip qubit 0 = MSB)."""
    v = np.zeros(2 ** nq)
    for k in range(2 ** nq):
        bits = [(k >> (nq - 1 - q)) & 1 for q in range(nq)]
        z = [1 - 2 * b for b in bits]
        v[k] = sum(z[i] * z[i + 1] for i in range(p)) / p
    return v


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    H, names = build(NQ, P)
    zzv = zz_vector(NQ, P)
    psi0 = qp.tensor([qp.basis(2, 0)] * NQ)
    noise_g = NoiseModel(n_qubits=NQ, T2=T2, gate_error_2q=GATE_2Q, gate_coherent_frac=COH_FRAC)
    noise_0 = NoiseModel(n_qubits=NQ, T2=T2)
    probs_g = NoisyQuTiPRunner(NQ, noise=noise_g).make_probs_fn(psi0)   # gate ON (PSR branches)
    probs_0 = NoisyQuTiPRunner(NQ, noise=noise_0).make_probs_fn(psi0)   # gate OFF (landscape)

    def cost_probs(theta, probsfn):
        th = np.clip(theta, -BOX, BOX)
        return probsfn([[H.set_parameterizedHam({names[i]: float(th[i]) for i in range(P)}), T]])

    def Obs(theta):                                  # exact noisy ⟨O⟩ (gate off)
        return float(zzv @ cost_probs(theta, probs_0))

    def Creg(theta):                                 # regularized descent objective
        th = np.clip(theta, -BOX, BOX)
        return Obs(theta) + 0.5 * LAM * float(th @ th)

    def shot_mean(probs, n, rng):                    # n-shot diagonal readout of ⟨O⟩
        idx = rng.choice(len(probs), size=int(max(1, n)), p=probs)
        return float(np.mean(zzv[idx]))

    def partial_H(theta, i):
        th = np.clip(theta, -BOX, BOX)
        return H.set_parameterizedHam({names[k]: float(th[k]) for k in range(P) if k != i})

    def _shift(theta, i, s):
        t = np.clip(theta, -BOX, BOX).copy(); t[i] += s; return t

    # ── real PSR gradient (per component; gate error included via probs_g) ──
    def psr_grad(theta, rng):
        g = np.zeros(P)
        nper = int(max(1, round((B / P) / (2 * N_POOL))))
        for i in range(P):
            Hi = partial_H(theta, i)
            orig = np.random.rand; np.random.rand = lambda k: (np.arange(k) + 0.5) / k
            try:
                progs = observable_program_generator(Hi, T, n_sample=N_POOL, n_repetition=1,
                                                     diff_var=names[i], value=float(np.clip(theta, -BOX, BOX)[i]),
                                                     short_kick=True)
            finally:
                np.random.rand = orig
            H_tot, ug, _ = progs[0]; nb = len(H_tot) // 2
            fm = np.array([shot_mean(probs_g(H_tot[2 * j]), nper, rng) for j in range(nb)])
            fp = np.array([shot_mean(probs_g(H_tot[2 * j + 1]), nper, rng) for j in range(nb)])
            g[i] = (T / nb) * float(ug) * float(np.sum(fm - fp))
        return g

    # ── real NSR gradient (per component; stochastic Nyquist sampler, gate-error immune) ──
    ns = np.arange(MAXN); pw = 1.0 / (ns + 0.5) ** 2; pw /= pw.sum()

    def nsr_grad(theta, rng):
        g = np.zeros(P)
        nshot = int(max(1, round(B / P)))
        for i in range(P):
            Hi = partial_H(theta, i)
            _, A = tangent_hamiltonian(Hi, names[i], float(np.clip(theta, -BOX, BOX)[i]))
            K = bandwidth_K(A, T); L1 = 2 * np.pi * K
            # the sampler visits only 2*MAXN distinct shifts s = σ(n+0.5)/(2K); compute
            # their readout probs ONCE, then draw shots from the cache (no per-shot mesolve).
            shifts = {}
            for sg in (-1.0, 1.0):
                for n in range(MAXN):
                    s = sg * (n + 0.5) / (2 * K)
                    shifts[(n, sg)] = cost_probs(_shift(theta, i, s), probs_0)
            n_draw = rng.choice(ns, size=nshot, p=pw); sig = rng.choice([-1.0, 1.0], size=nshot)
            vals = np.array([shot_mean(shifts[(int(nn), ss)], 1, rng) for nn, ss in zip(n_draw, sig)])
            g[i] = float(np.mean(L1 * ((-1.0) ** n_draw) * sig * vals))
        return g

    # ── real FD gradient (noisy secant + δ + shots) ──
    def fd_grad(theta, eps, rng):
        g = np.zeros(P); nper = max(1, B // (2 * P))
        for i in range(P):
            dp, dm = rng.normal(0, DELTA), rng.normal(0, DELTA)
            pp = cost_probs(_shift(theta, i, eps + dp), probs_0)
            pm = cost_probs(_shift(theta, i, -(eps - dm)), probs_0)
            g[i] = (shot_mean(pp, nper, rng) - shot_mean(pm, nper, rng)) / (2 * eps)
        return g

    theta0 = np.array([0.7, -0.5, 0.6, -0.4])[:P]

    def grad_obs_exact(theta, h=1e-3):
        return np.array([(Obs(_shift(theta, i, h)) - Obs(_shift(theta, i, -h))) / (2 * h) for i in range(P)])

    if PROBE:
        rng = np.random.default_rng(0)
        print(f"=== COST PROBE: P={P} ({NQ}q), N_POOL={N_POOL}, MAXN={MAXN}, B={B} ===")
        print(f"exact ∇⟨O⟩ (fine FD) = {np.round(grad_obs_exact(theta0),4)}")
        for name, fn in [("PSR", lambda: psr_grad(theta0, rng)),
                         ("NSR", lambda: nsr_grad(theta0, rng)),
                         ("FD ", lambda: fd_grad(theta0, EPS_STAR, rng))]:
            t0 = time.perf_counter(); g = fn(); dt = time.perf_counter() - t0
            print(f"{name}: 1 gradient = {dt:6.2f}s   g={np.round(g,4)}")
        t0 = time.perf_counter(); c = Creg(theta0); print(f"Creg(θ0)={c:+.4f}  ({time.perf_counter()-t0:.3f}s/eval)")
        return

    # ── θ* = interior basin min of the regularized noisy cost (noiseless GD from θ0) ──
    cache = os.path.join(FIGDIR, "F_loop_real_curves.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        theta_star = z["theta_star"]; C_star = float(z["C_star"])
        print(f"θ*(cached)={np.round(theta_star,3)}  C*={C_star:.4f}")
    else:
        th = theta0.copy()
        for _ in range(300):
            th = np.clip(th - 0.1 * (grad_obs_exact(th) + LAM * th), -BOX, BOX)
        theta_star = th; C_star = float(Creg(theta_star))
        print(f"θ*(interior)={np.round(theta_star,3)}  Obs*={Obs(theta_star):+.4f}  C*={C_star:.4f}")

    # ── descent with the REAL estimators (regularizer λθ = exact classical add-on) ──
    def descend(grad_fn, seed):
        srng = np.random.default_rng(seed)                       # shared per-seed start jitter
        theta = np.clip(theta0 + srng.normal(0, 0.08, P), -BOX, BOX)
        nrng = np.random.default_rng(seed * 131 + 17)            # sampling noise
        traj, ascents = [Creg(theta) - C_star], 0
        for _ in range(ITERS):
            g = grad_fn(theta, nrng) + LAM * np.clip(theta, -BOX, BOX)
            new = np.clip(theta - ETA * g, -BOX, BOX)
            if Creg(new) > Creg(theta) + 1e-3:
                ascents += 1
            theta = new
            traj.append(Creg(theta) - C_star)
        return np.array(traj), ascents

    methods = {
        "PSR": (lambda th, r: psr_grad(th, r), C_PSR, "-"),
        "NSR": (lambda th, r: nsr_grad(th, r), C_NSR, "--"),
        "FD 1x": (lambda th, r: fd_grad(th, EPS_STAR, r), C_FD, "-"),
        "FD 0.3x": (lambda th, r: fd_grad(th, 0.3 * EPS_STAR, r), "#E69F00", ":"),
        "FD 3x": (lambda th, r: fd_grad(th, 3 * EPS_STAR, r), "#7b1fa2", "-."),
    }
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        results = {lab: dict(curves=z[f"c_{i}"], ascents=int(z[f"a_{i}"]),
                             color=methods[lab][1], ls=methods[lab][2]) for i, lab in enumerate(methods)}
        print(f"loaded cached curves from {cache}")
    else:
        results = {}
        for lab, (gf, c, ls) in methods.items():
            t0 = time.perf_counter()
            curves, asc = [], 0
            for s in range(SEEDS):
                tc, a = descend(gf, 2000 + s); curves.append(tc); asc += a
            curves = np.array(curves)
            results[lab] = dict(curves=curves, ascents=asc, color=c, ls=ls)
            print(f"{lab:8s}: final median (C-C*)={np.median(curves[:,-1]):.4f}  ascents={asc}  "
                  f"({time.perf_counter()-t0:.0f}s)")
        save = {"C_star": C_star, "theta_star": theta_star}
        for i, lab in enumerate(methods):
            save[f"c_{i}"] = results[lab]["curves"]; save[f"a_{i}"] = results[lab]["ascents"]
        np.savez(cache, **save); print(f"cached -> {cache}")

    # ── plot ──
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    xexec = np.arange(ITERS + 1) * B
    order = [k for k in results if k != "FD 0.3x"] + (["FD 0.3x"] if "FD 0.3x" in results else [])
    for lab in order:
        r = results[lab]; cur = np.maximum(r["curves"], 1e-4)
        med = np.median(cur, 0); lo = np.percentile(cur, 25, 0); hi = np.percentile(cur, 75, 0)
        lw = 1.4 if lab == "FD 0.3x" else 1.8
        ax.semilogy(xexec, med, r["ls"], color=r["color"], lw=lw, label=lab,
                    zorder=5 if lab == "FD 0.3x" else 3)
        ax.fill_between(xexec, lo, hi, color=r["color"], alpha=0.12)
    ax.set_xlabel("cumulative quantum executions")
    ax.set_ylabel(r"$C_{\rm noisy}(\theta_t)-C_{\rm noisy}(\theta^*)$")
    ax.set_title(f"F-loop (REAL estimators) — TFIM P={P} descent, emulated noisy cost "
                 f"($T/T_2^*$=0.15, $B$={B}/grad)", fontsize=8.5)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right"); ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"F_loop_real.{e}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    json.dump({lab: dict(final_med=float(np.median(r["curves"][:, -1])), ascents=r["ascents"])
               for lab, r in results.items()} | {"C_star": C_star, "theta_star": theta_star.tolist(),
               "P": P, "B": B, "eta": ETA, "lam": LAM, "eps_star": EPS_STAR, "gate_2q": GATE_2Q},
              open(os.path.join(FIGDIR, "F_loop_real.json"), "w"), indent=2, default=float)
    print("wrote F_loop_real.pdf/.png/.json")
    print(f"DATA NOTE (C2): F-loop with the REAL sampled estimators. TFIM P={P} ({NQ}q), cost "
          f"C=⟨(1/P)ΣZZ⟩+λ/2|θ|² (λ={LAM}; interior θ*), diagonal-readout shots (basis-state "
          f"sampling of diag ρ). PSR = real kick branches through the noisy runner INCLUDING the "
          f"T4 kick gate error (short-kick, N_POOL={N_POOL}) → carries the ~0.028 digital bias; "
          f"NSR = real stochastic Nyquist sampler (gate-error immune; its n≥1 tail shifts clip at "
          f"the amplitude box near θ* — the headroom/certificate cost). FD = real noisy secant + "
          f"δ={DELTA} + shots. B={B} exec/grad, η={ETA}, {ITERS} iters, {SEEDS} seeds median+IQR, "
          f"θ0+N(0,0.08) shared start. Gate rate, δ, T2* = T4 best-guess (flagged). Surrogate "
          f"(build_Floop.py) retired.")


if __name__ == "__main__":
    main()
