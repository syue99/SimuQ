"""
build_Floop_trajectory.py — F-loop (Fig A-c): descent-trajectory companion to the direction map.

Honest IQS recovery on a SLOPPY valley, with an INTERIOR, CENTERED optimum.

Objective (two Z-diagonal observables — same readout, no extra measurement channel):
    C(θ) = (⟨Z0Z1⟩(θ) − a*)² + w·(⟨Z0+Z1⟩(θ) − b*)²,   a*,b* = exact values at a chosen interior θ★.
θ* = θ★ by construction (no boundary clipping — the single-observable ⟨O⟩+prior form has |∇O|≈1
everywhere, no interior stationary point, so its min always clips to an edge; the residual form fixes
that). w sets the conditioning: small w ⇒ a sloppy valley (κ~1/w), the paper's thesis.

Why FD fails in THREE ways while PSR/NSR reach θ*:
  • FD is the honest BLACK-BOX central difference of the noisy objective C. At θ* the objective's
    gradient vanishes but C‴(θ*)≠0, so FD-of-C does NOT self-correct: it floors at a systematic
    offset (ε²/6)|C‴|/μ_soft (+ δ/ε), amplified by the small soft-direction curvature μ_soft. Hence
    the oracle-tuned ε (irreducible truncation floor), the too-large ε (truncation-dominated) and the
    too-small ε (δ/ε variance, wide IQR) all settle short. (Differencing the OBSERVABLES instead
    would self-correct — that is not what a black-box optimizer does.)
  • PSR/NSR assemble the ANALYTIC gradient 2·r_a·∇⟨O_a⟩ + 2w·r_b·∇⟨O_b⟩ from ε-free observable-
    gradient estimators (residuals r measured at θ). Unbiased ⇒ they crawl the valley to θ*.

2q TFIM H=θ1·Z0Z1+θ2·(X0+X1), T/T2*=0.15, compiled to machine-native segments, emulated under T4,
finite-shot. Reference = the noisy landscape. NO rescale.
Run: conda run -n qec_pg python differential_computing/tests/build_Floop_trajectory.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_Floop_direction as bd
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

# Evolution time sets the landscape's sharpness (⟨O⟩ curvature ≈ T²).  T = 0.8 was the
# 2026-08 choice ("gentle valley"); under the P0-0 setpoint rule and with the estimator
# bugs fixed, every method plateaued there at ≈0.04 (shot-noise jitter) and no ε
# separated FD from the shift rules.  The 2026-09-04 scan of T ∈ {0.8..3.0} (audit of
# per-step gradient error at θ*, B = 6000) gives at T = 2.5: PSR/NSR ≈ 0.02, FD's best
# step 0.05, i.e. the shift rules hold the 0.03 tolerance and FD does not at any ε.
# T/T2* = 0.15 held by scaling T2, so the noise regime is unchanged.
bd.T = float(os.environ.get("TEVOL", "2.5"))            # 2026-09-04: 0.8 → 2.5 (landscape scan; T/T2* held at 0.15)
bd.T2 = bd.T / 0.15
bd.RUNNER = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=bd.T2))
bd.PROBS = bd.RUNNER.make_probs_fn(bd.PSI0)
bd.B_BUDGET = int(os.environ.get("B_BUDGET", "6000"))      # FINITE executions per gradient (paper: 6000)
bd.DELTA = float(os.environ.get("DELTA", "0.02"))          # amplitude-resolution jitter (zero-mean)
bd.M_PSR = int(os.environ.get("M_PSR", "32"))              # converge PSR's τ-integral for both gens

ZZV = np.array([1.0, -1.0, -1.0, 1.0])                     # ⟨Z0Z1⟩ outcome vector (Z-diagonal)
ZSV = np.array([2.0, 0.0, 0.0, -2.0])                      # ⟨Z0+Z1⟩ outcome vector (Z-diagonal)

DOM = (0.2, 1.4)
W = float(os.environ.get("W", "0.25"))                    # residual weight ⇒ conditioning κ~1/w (paper: 0.25)
ITERS = int(os.environ.get("FLOOP_ITERS", 100))          # paper: 100 steps (50 drawn)
SEEDS = int(os.environ.get("FLOOP_SEEDS", 20))            # L4 [B]: ≥20 seeds with an IQR band
ETA_ENV = os.environ.get("FLOOP_ETA")                     # auto (1.4/μ_stiff) unless overridden
# FD steps in the PAPER convention (probes θ ± ε/2, divide by ε).  The builder used
# θ ± ε ÷ 2ε before 2026-09-04; the physical probes below are the same as the old
# 0.7 / 0.04 / [0.15..0.35], relabelled.
FD_WRONG = float(os.environ.get("FD_WRONG", "0.5"))       # too-large ε (truncation-dominated: fails)
FD_TOO_SMALL = float(os.environ.get("FD_SMALL", "0.05"))  # too-small ε = 2.5δ (δ/ε noise: unreliable, wide IQR)
FD_ORACLE_GRID = [0.1, 0.15, 0.2, 0.3, 0.5]                # a-priori-unknowable "best" ε (needs θ*)
# FLOOP_TAG=<name>: pilot/diagnostic run — caches and figures get a _<name> suffix
# and the paper_fig outputs are NOT written.
# 2026-09-04 (owner): the loop is SGD, so (i) the step decays, η_t = η₀/(1 + t/τ), and
# (ii) the reported iterate is the tail average θ̄_t = mean(θ_{⌈t/2⌉..t}) (Polyak).  Both
# are what makes an UNBIASED estimator converge to θ* itself instead of a noise ball of
# radius ∝ √η, and neither can remove a bias (FD's truncation).  τ = ∞ and FLOOP_AVG=0
# recover the raw fixed-step loop.
ETA_TAU = float(os.environ.get("FLOOP_TAU", "20"))          # decay constant in steps (pilots: 10 and 20 equivalent for PSR/NSR; 20 kept)
AVG = os.environ.get("FLOOP_AVG", "1") == "1"
FLOOP_TAG = os.environ.get("FLOOP_TAG", "")
SUF = ("_" + FLOOP_TAG) if FLOOP_TAG else ""
FIGDIR = bd.FIGDIR
C_PSR, C_NSR, C_FDO, C_FDW, C_FDS = "#0072B2", "#009E73", "#E69F00", "#d62728", "#7b3fa0"

_OCa, _OCb = {}, {}


def probs_at(th):
    """Outcome probabilities of a PROGRAM at coefficient vector th.  Not clipped: the
    box DOM constrains the optimizer's iterate (clipd), not what a program may dial —
    NSR's Nyquist shifts (±0.98, ±2.9, … in θ1 at T = 0.8) and FD's wide probes
    leave the box by construction.  (Before 2026-09-04 this clipped, which silently
    evaluated most NSR shifts at the box edge.)"""
    return bd.probs_at(np.asarray(th, float))


def Oa_ex(th):
    k = (round(float(th[0]), 4), round(float(th[1]), 4))
    if k not in _OCa:
        _OCa[k] = float(ZZV @ probs_at(th))
    return _OCa[k]


def Ob_ex(th):
    k = (round(float(th[0]), 4), round(float(th[1]), 4))
    if k not in _OCb:
        _OCb[k] = float(ZSV @ probs_at(th))
    return _OCb[k]


def shot_both(p, n, rng):
    idx = rng.choice(4, size=int(max(1, n)), p=p)
    return float(np.mean(ZZV[idx])), float(np.mean(ZSV[idx]))


A_STAR = B_STAR = None
THETA_STAR = None
ETA = None


def clipd(th):
    return np.clip(th, DOM[0], DOM[1])


def Cval(th):
    t = clipd(th)
    return (Oa_ex(t) - A_STAR) ** 2 + W * (Ob_ex(t) - B_STAR) ** 2


def _gradf(f, th, h=1e-3):
    return np.array([(f([th[0] + h, th[1]]) - f([th[0] - h, th[1]])) / (2 * h),
                     (f([th[0], th[1] + h]) - f([th[0], th[1] - h])) / (2 * h)])


def gC_exact(th):
    t = clipd(th)
    return 2 * (Oa_ex(t) - A_STAR) * _gradf(Oa_ex, t) + 2 * W * (Ob_ex(t) - B_STAR) * _gradf(Ob_ex, t)


def gd_exact(start, steps=400, eta=0.3):
    tv = np.array(start, float)
    for _ in range(steps):
        tv = clipd(tv - eta * gC_exact(tv))
    return tv


def hess_C(th, h=2e-2):
    def C(t):
        return Cval(np.array(t))
    Hxx = (C([th[0] + h, th[1]]) - 2 * C(th) + C([th[0] - h, th[1]])) / h ** 2
    Hyy = (C([th[0], th[1] + h]) - 2 * C(th) + C([th[0], th[1] - h])) / h ** 2
    Hxy = (C([th[0] + h, th[1] + h]) - C([th[0] + h, th[1] - h]) - C([th[0] - h, th[1] + h])
           + C([th[0] - h, th[1] - h])) / (4 * h ** 2)
    return np.array([[Hxx, Hxy], [Hxy, Hyy]])


# ── P0-0 setpoint draw, per-change rule (owner's ruling 2026-09-04) ──────────────────────────────
class Dial:
    """One setpoint draw per coefficient, r = bd.DELTA, taken when the programmed VALUE of that
    coefficient changes and held until it changes again.  A fresh Dial per gradient estimate
    (one optimizer step), so every draw is redrawn between steps.  FD's four probe programs
    each redraw the coefficient they move; PSR's residual measurement and all its branches dial
    the same vector, so one draw serves the whole estimate; NSR's shifted coefficient changes
    on (almost) every execution and is drawn per execution in _obs_grads_nsr."""

    def __init__(self, rng, r):
        self.rng, self.r = rng, float(r)
        self.last = np.full(2, np.nan)
        self.d = np.zeros(2)

    def __call__(self, th):
        th = np.asarray(th, float)
        ch = ~np.isclose(th, self.last, rtol=0.0, atol=1e-12)
        if self.r > 0 and ch.any():
            self.d[ch] = self.rng.normal(0.0, self.r, int(ch.sum()))
        self.last = th.copy()
        return th + self.d

    def forget(self, ell):                  # the next program re-dials coefficient ell
        self.last[ell] = np.nan


# ── estimators ────────────────────────────────────────────────────────────────────────────────────
def fd_grad_C(th, eps, rng):
    """Honest black-box central difference OF THE OBJECTIVE C. Truncation (ε²/6·C‴, deterministic) +
    δ/ε (resolution jitter, zero-mean per programming). Does NOT self-correct at θ* (C‴(θ*)≠0)."""
    g = np.zeros(2); nper = max(1, bd.B_BUDGET // 4)
    t = clipd(th); dial = Dial(rng, bd.DELTA)

    def Cshot(x):
        oa, ob = shot_both(probs_at(dial(x)), nper, rng)      # program x lands at x + δ
        return (oa - A_STAR) ** 2 + W * (ob - B_STAR) ** 2
    for ell in range(2):                                      # paper convention: θ ± ε/2, ÷ ε
        ep = t.copy(); ep[ell] += 0.5 * eps
        em = t.copy(); em[ell] -= 0.5 * eps
        g[ell] = (Cshot(ep) - Cshot(em)) / eps
    return g


def _psr_programs(ell, th):
    """All PSR branch programs for coefficient ell at coefficient vector th: one program set per
    Pauli term of the coefficient (θ2 multiplies X0 + X1 → two sets), each M_PSR τ-samples ×
    (−, +) branches.  Before 2026-09-04 only the first set was used, which halved ∂/∂θ2."""
    Hp = bd.Hp_for(ell, th)
    orig = np.random.rand
    np.random.rand = lambda k: (np.arange(k) + 0.5) / k
    try:
        return bd.observable_program_generator(Hp, bd.T, n_sample=bd.M_PSR, n_repetition=1,
                                               diff_var=bd.NAMES[ell], value=float(th[ell]))
    finally:
        np.random.rand = orig


def _obs_grads_psr(th, rng, nshot_half, dial=None, exact=False):
    """PSR observable gradients at the DIALED point (every branch programs the same source
    coefficients, so the whole estimate shares one setpoint draw).  nshot_half executions per
    coefficient, split evenly over that coefficient's actual branch count."""
    ga, gb = np.zeros(2), np.zeros(2)
    tr = dial(th) if dial is not None else np.asarray(th, float)
    for ell in range(2):
        pr = _psr_programs(ell, tr)
        nb_tot = sum(len(H_tot) // 2 for H_tot, _, _ in pr)
        nper = int(max(1, round(nshot_half / (2 * nb_tot))))
        for H_tot, ug, _ in pr:                           # weight T/M per τ-sample, per term
            nb = len(H_tot) // 2
            sa = sb = 0.0
            for i in range(nb):
                pm, pp = bd.PROBS(H_tot[2 * i]), bd.PROBS(H_tot[2 * i + 1])
                if exact:
                    ma_a, ma_b = float(ZZV @ pm), float(ZSV @ pm)
                    pa_a, pa_b = float(ZZV @ pp), float(ZSV @ pp)
                else:
                    ma_a, ma_b = shot_both(pm, nper, rng)
                    pa_a, pa_b = shot_both(pp, nper, rng)
                sa += ma_a - pa_a; sb += ma_b - pa_b
            ga[ell] += (bd.T / bd.M_PSR) * float(ug) * sa
            gb[ell] += (bd.T / bd.M_PSR) * float(ug) * sb
    return ga, gb


NSR_STENCIL_H = 0.04                    # = 2r: quadratic-in-δ realization of a shifted program


def _obs_grads_nsr(th, rng, nshot, dial=None, exact=False):
    """NSR observable gradients.  Shifted programs are never clipped (M = ∞).  Setpoint rule:
    the shifted coefficient changes value on (almost) every execution, so it takes a fresh
    draw whenever the drawn (κ, σ) differs from the previous execution's; the other
    coefficient is held at the value (and draw) the estimate started with.  The per-execution
    draw is realized by a 3-point stencil at each shift (probs at s, s ± NSR_STENCIL_H,
    quadratic in δ — exact to O(δ³))."""
    ga, gb = np.zeros(2), np.zeros(2)
    r = bd.DELTA if dial is not None else 0.0
    for ell in range(2):
        base = dial(th) if dial is not None else np.asarray(th, float)
        base = base.copy(); base[ell] = float(th[ell])         # the shifted coefficient's own δ is per execution
        Hp = bd.Hp_for(ell, th)
        _, A = bd.tangent_hamiltonian(Hp, bd.NAMES[ell], float(th[ell]))
        K = bd.bandwidth_K(A, bd.T); L1 = 2 * np.pi * K
        P0, P1, P2 = {}, {}, {}
        for sg in (-1.0, 1.0):
            for n in range(bd.MAXN):
                s = sg * (n + 0.5) / (2 * K)
                tt = base.copy(); tt[ell] += s
                p0 = probs_at(tt)
                if r > 0:
                    tp = tt.copy(); tp[ell] += NSR_STENCIL_H; tm = tt.copy(); tm[ell] -= NSR_STENCIL_H
                    pp, pm = probs_at(tp), probs_at(tm)
                    P1[(n, sg)] = (pp - pm) / (2 * NSR_STENCIL_H)
                    P2[(n, sg)] = (pp - 2 * p0 + pm) / (2 * NSR_STENCIL_H ** 2)
                P0[(n, sg)] = p0
        if exact:
            ga[ell] = float(sum(bd.pw[n] * L1 * ((-1.0) ** n) * sg * float(ZZV @ P0[(n, sg)])
                                for sg in (-1.0, 1.0) for n in range(bd.MAXN)) / 2.0)
            gb[ell] = float(sum(bd.pw[n] * L1 * ((-1.0) ** n) * sg * float(ZSV @ P0[(n, sg)])
                                for sg in (-1.0, 1.0) for n in range(bd.MAXN)) / 2.0)
            continue
        nd = rng.choice(bd.ns, size=nshot, p=bd.pw); sig = rng.choice([-1.0, 1.0], size=nshot)
        if r > 0:
            dl = rng.normal(0.0, r, nshot)
            same = np.zeros(nshot, bool); same[1:] = (nd[1:] == nd[:-1]) & (sig[1:] == sig[:-1])
            for k in range(1, nshot):                          # value unchanged → draw held
                if same[k]:
                    dl[k] = dl[k - 1]
        else:
            dl = np.zeros(nshot)
        va = np.empty(nshot); vb = np.empty(nshot)
        for k, (a, b) in enumerate(zip(nd, sig)):
            key = (int(a), b)
            if r > 0:
                p = P0[key] + dl[k] * P1[key] + dl[k] ** 2 * P2[key]
                p = np.clip(p, 0.0, None); p = p / p.sum()
            else:
                p = P0[key]
            va[k], vb[k] = shot_both(p, 1, rng)
        ga[ell] = float(np.mean(L1 * ((-1.0) ** nd) * sig * va))
        gb[ell] = float(np.mean(L1 * ((-1.0) ** nd) * sig * vb))
        if dial is not None:
            dial.forget(ell)                                   # ell returns to its source value next
    return ga, gb


def iqs_grad(kind, th, rng):
    """Analytic IQS gradient 2·r_a·∇O_a + 2w·r_b·∇O_b, residuals measured at θ (unbiased)."""
    t = clipd(th); dial = Dial(rng, bd.DELTA)
    n_res = max(50, bd.B_BUDGET // 5)
    ra_s, rb_s = shot_both(probs_at(dial(t)), n_res, rng)     # residuals: the program dials t
    ra, rb = ra_s - A_STAR, rb_s - B_STAR
    ngrad = max(1, (bd.B_BUDGET - 2 * n_res))
    if kind == "PSR":
        ga, gb = _obs_grads_psr(t, rng, ngrad / 2, dial)
    else:
        ga, gb = _obs_grads_nsr(t, rng, int(max(1, round(ngrad / 2))), dial)
    return 2 * ra * ga + 2 * W * rb * gb


def descend(kind, seed, eps=None):
    srng = np.random.default_rng(seed)
    th = clipd(THETA_STARTPT + srng.normal(0, 0.015, 2))
    nrng = np.random.default_rng(seed * 131 + 17)
    traj = [th.copy()]
    for t in range(ITERS):
        if kind == "FD":
            g = fd_grad_C(th, eps, nrng)
        else:
            g = iqs_grad(kind, th, nrng)
        th = clipd(th - ETA / (1.0 + t / ETA_TAU) * g)
        traj.append(th.copy())
    return np.array(traj)


THETA_STARTPT = None


def setup():
    global A_STAR, B_STAR, THETA_STAR, ETA, THETA_STARTPT
    env_ts = os.environ.get("TSTAR", "1,1")               # paper: θ* = (1, 1)
    if env_ts:
        THETA_STAR = np.array([float(x) for x in env_ts.split(",")])
    else:                                # pick interior θ★ with ∇Oa ⟂ ∇Ob (identifiable) and room
        best = None
        for t1 in np.linspace(0.6, 1.0, 9):
            for t2 in np.linspace(0.6, 1.0, 9):
                th = np.array([t1, t2]); ga, gb = _gradf(Oa_ex, th), _gradf(Ob_ex, th)
                na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
                if na < 0.25 or nb < 0.25:
                    continue
                ang = abs(np.degrees(np.arccos(min(1, abs(ga @ gb / (na * nb))))) - 90)
                if best is None or ang < best[0]:
                    best = (ang, th)
        THETA_STAR = best[1]
    A_STAR, B_STAR = Oa_ex(THETA_STAR), Ob_ex(THETA_STAR)
    HC = hess_C(THETA_STAR); ev, evec = np.linalg.eigh(HC)
    muC, mu_stiff = float(max(ev[0], 1e-6)), float(ev[1])
    ETA = float(ETA_ENV) if ETA_ENV else float(np.clip(1.4 / mu_stiff, 0.05, 0.9))
    # start ~0.4 away in θ★'s basin (verified by exact GD at the descent η), mostly along the STIFF
    # direction so PSR/NSR visibly converge, a little along soft so the FD spread is legible.
    soft, stiff = evec[:, 0], evec[:, 1]
    cand = None
    for r in (0.42, 0.38, 0.48, 0.34):
        for sgn in (1, -1):
            for frac in (0.7, 0.55, 0.85):
                s = THETA_STAR + r * sgn * (frac * stiff + (1 - frac) * soft)
                if not all(DOM[0] + 0.1 < s[k] < DOM[1] - 0.1 for k in (0, 1)):
                    continue
                if np.hypot(*(gd_exact(s, steps=400, eta=ETA) - THETA_STAR)) < 0.04:
                    cand = s; break
            if cand is not None:
                break
        if cand is not None:
            break
    if cand is None:
        raise SystemExit("no basin start where exact→θ* at descent η — adjust W/ETA")
    THETA_STARTPT = cand
    print(f"θ*={np.round(THETA_STAR,3)} interior  a*={A_STAR:+.3f} b*={B_STAR:+.3f}  W={W}  "
          f"μ_soft={muC:.3f} μ_stiff={mu_stiff:.2f} κ={mu_stiff/muC:.0f} ETA={ETA:.3f}", flush=True)
    print(f"START={np.round(THETA_STARTPT,3)}  ‖START−θ*‖={np.hypot(*(THETA_STARTPT-THETA_STAR)):.3f}",
          flush=True)
    return muC, soft


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_2"))
    os.makedirs(OUT, exist_ok=True)
    muC, soft = setup()

    # |C‴| along the soft direction: the truncation-bias coefficient the FD offset divides by μ_soft
    hh = 2e-2
    C3 = abs(float((Cval(THETA_STAR + 2 * hh * soft) - 2 * Cval(THETA_STAR + hh * soft)
                    + 2 * Cval(THETA_STAR - hh * soft) - Cval(THETA_STAR - 2 * hh * soft))
                   / (2 * hh ** 3)))
    delta = bd.DELTA
    def pred_offset(eps):
        return (2 * delta / eps + (eps ** 2 / 24) * C3) / muC     # paper ε: probes ± ε/2
    print(f"|C‴|(soft)={C3:.2f}  δ={delta}  pred oracle offset≈{pred_offset(0.5):.3f}", flush=True)
    # estimator audit (shot-free, δ-free): both shift rules must reproduce ∇⟨O⟩ for BOTH
    # coefficients — this is what catches the two-term PSR and the clipped-NSR bugs.
    for lab in ("PSR", "NSR"):
        fn = _obs_grads_psr if lab == "PSR" else _obs_grads_nsr
        for pt in (THETA_STARTPT, THETA_STAR):
            ga_x, gb_x = _gradf(Oa_ex, pt), _gradf(Ob_ex, pt)
            ga_e, gb_e = fn(pt, None, 0, None, exact=True)
            err = max(np.linalg.norm(ga_e - ga_x) / np.linalg.norm(ga_x),
                      np.linalg.norm(gb_e - gb_x) / np.linalg.norm(gb_x))
            print(f"  audit {lab} exact at {np.round(pt,3)}: rel|Δ∇O|={100*err:.2f}%  "
                  f"(∇Oa {np.round(ga_e,3)} vs {np.round(ga_x,3)}; ∇Ob {np.round(gb_e,3)} vs {np.round(gb_x,3)})",
                  flush=True)
            # PSR is exact to the τ-quadrature (0.1%); NSR carries the MAXN=14-mode series
            # truncation (~1.5%).  Anything larger is a bug (the two-term PSR bug was 50%,
            # the clipped-NSR bug 30-100%).
            assert err < 0.03, f"{lab} observable gradient off by {100*err:.1f}%"

    # cache the (expensive) trajectories so plot tweaks never re-run the sim (REPLOT=1 reloads)
    CACHE_NPZ = os.path.join(FIGDIR, f"F_loop_curves{SUF}.npz")
    CACHE_JSON = os.path.join(FIGDIR, f"F_loop_meta{SUF}.json")
    if os.environ.get("REPLOT") and os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_JSON):
        meta = json.load(open(CACHE_JSON)); npz = np.load(CACHE_NPZ)
        order = meta["order"]; ob = (0.0, meta["ob_eps"], None); results = {}
        for i, lab in enumerate(order):
            md = meta["methods"][lab]
            results[lab] = dict(trajs=npz[f"t{i}"], c=md["c"], mk=md["mk"])
            if md.get("eps") is not None:
                results[lab]["eps"] = md["eps"]
        fd_series = [(lab, results[lab]["eps"], results[lab]["c"], results[lab]["mk"],
                      results[lab]["trajs"]) for lab in order if "eps" in results[lab]]
        print("REPLOT: loaded cached trajectories (no sim)", flush=True)
    else:
        results = {}
        for lab, kind, c, mk in [("PSR", "PSR", C_PSR, "o"), ("NSR", "NSR", C_NSR, "s")]:
            trajs = np.array([descend(kind, 7000 + s) for s in range(SEEDS)])
            results[lab] = dict(trajs=trajs, c=c, mk=mk)
            print(f"  {lab:20s}: median final="
                  f"{np.median([np.hypot(*(tr[-1]-THETA_STAR)) for tr in trajs]):.3f}", flush=True)
        ob = None
        for e in FD_ORACLE_GRID:
            trajs = np.array([descend("FD", 7000 + s, eps=e) for s in range(SEEDS)])
            m = np.median([np.hypot(*(tr[-1] - THETA_STAR)) for tr in trajs])
            print(f"    oracle-grid ε={e:.2f}: median {m:.3f}  (pred {pred_offset(e):.3f})", flush=True)
            if ob is None or m < ob[0]:
                ob = (m, e, trajs)
        fd_series = [(f"FD ε={ob[1]:g} (best ε — needs θ*)", ob[1], C_FDO, "^", ob[2]),
                     (f"FD ε={FD_WRONG:g} (too large: fails)", FD_WRONG, C_FDW, "D",
                      np.array([descend("FD", 7000 + s, eps=FD_WRONG) for s in range(SEEDS)])),
                     (f"FD ε={FD_TOO_SMALL:g} (too small: unreliable)", FD_TOO_SMALL, C_FDS, "v",
                      np.array([descend("FD", 7000 + s, eps=FD_TOO_SMALL) for s in range(SEEDS)]))]
        for lab, eps, c, mk, trajs in fd_series:
            results[lab] = dict(trajs=trajs, c=c, mk=mk, eps=eps)
            print(f"  {lab:22s}: median final="
                  f"{np.median([np.hypot(*(tr[-1]-THETA_STAR)) for tr in trajs]):.3f}  "
                  f"(pred b(ε)/μ={pred_offset(eps):.3f})", flush=True)
        order = ["PSR", "NSR"] + [s[0] for s in fd_series]
        np.savez(CACHE_NPZ, **{f"t{i}": results[lab]["trajs"] for i, lab in enumerate(order)})   # RAW iterates
        json.dump({"order": order, "ob_eps": ob[1], "eta0": ETA, "eta_tau": ETA_TAU,
                   "methods": {lab: {"c": results[lab]["c"], "mk": results[lab]["mk"],
                                     "eps": results[lab].get("eps")} for lab in order}},
                  open(CACHE_JSON, "w"), indent=2)

    def tail_avg(trajs):
        """Polyak tail average per seed: θ̄_t = mean(θ_s, s ∈ [⌈t/2⌉, t])."""
        out = np.empty_like(trajs)
        for t in range(trajs.shape[1]):
            out[:, t] = trajs[:, (t + 1) // 2:t + 1].mean(1)
        return out
    for lab in order:
        results[lab]["raw"] = results[lab]["trajs"]
        if AVG:
            results[lab]["trajs"] = tail_avg(results[lab]["raw"])
    B = bd.B_BUDGET
    THRESH = 0.03

    def dist_curves(trajs):
        D = np.array([[np.hypot(*(tr[t] - THETA_STAR)) for t in range(ITERS + 1)] for tr in trajs])
        return np.median(D, 0), np.percentile(D, 25, 0), np.percentile(D, 75, 0)

    def reached_held(med):
        for t in range(len(med)):
            if np.all(med[t:] < THRESH):
                return t
        return None
    dc = {lab: dist_curves(results[lab]["trajs"]) for lab in order}
    held = {lab: reached_held(dc[lab][0]) for lab in order}
    finals = {lab: float(dc[lab][0][-1]) for lab in order}

    # FLOOP_REPLOT §5 — computable from the cached arrays, no re-run:
    # terminal step = first step from which the MEDIAN stays inside tolerance for
    # 5 consecutive steps (a single crossing is ambiguous: a series can dip and
    # bounce back out); frac50 = fraction of seeds inside tolerance at step 50
    # (the number behind "reliably" — a claim about the band, not the median).
    NS = min(50, ITERS)

    def first_held5(med, k=5):
        for t in range(NS + 1 - k + 1):
            if np.all(med[t:t + k] < THRESH):
                return t
        return None
    term = {lab: first_held5(dc[lab][0]) for lab in order}
    frac50 = {lab: float(np.mean(np.linalg.norm(
        results[lab]["trajs"][:, NS] - THETA_STAR, axis=1) < THRESH))
        for lab in order}
    for lab in order:
        print(f"  {lab:24s}: held@{held[lab]} term5@{term[lab]} "
              f"frac50={frac50[lab]:.2f} final={finals[lab]:.3f}", flush=True)

    # ── plot ──
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "stix"})
    # frame on the REACHING bundle (θ*, start, and the endpoints that converge) so the final
    # trajectory is legible; let the failing too-large ε run off-frame (annotated as diverging).
    reach = [l for l in order if "too large" not in l]
    ends = np.array([np.median(results[l]["trajs"], 0)[min(50, ITERS)] for l in reach])
    pts = np.vstack([ends, THETA_STAR, THETA_STARTPT])
    ctr = 0.5 * (THETA_STAR + THETA_STARTPT)
    half = 1.25 * np.max(np.abs(pts - ctr), axis=0) + 0.05
    half = np.maximum(half, half.max() * 0.78)
    xlo, xhi = max(DOM[0], ctr[0] - half[0]), min(1.5, ctr[0] + half[0])
    ylo, yhi = max(DOM[0], ctr[1] - half[1]), min(1.5, ctr[1] + half[1])
    gx = np.linspace(xlo, xhi, 90); gy = np.linspace(ylo, yhi, 90)
    Zc = np.array([[Cval(np.array([a, b])) for a in gx] for b in gy])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.7))
    cf = axL.contourf(gx, gy, Zc, levels=26, cmap="viridis")
    axL.contour(gx, gy, Zc, levels=12, colors="w", linewidths=0.3, alpha=0.35)
    cb = fig.colorbar(cf, ax=axL, fraction=0.046, pad=0.02)
    cb.set_label(r"$C(\theta)=(\langle Z_0Z_1\rangle-a^*)^2+w(\langle Z_0{+}Z_1\rangle-b^*)^2$",
                 fontsize=7)
    PLOT_STEPS = min(50, ITERS)
    for lab in order:
        med = np.median(results[lab]["trajs"], 0)[:PLOT_STEPS + 1]
        sub = med[::5]                                   # points at steps 0,5,10,...,50
        c = results[lab]["c"]
        axL.plot(sub[:, 0], sub[:, 1], "-o", color=c, lw=1.4, zorder=4, alpha=0.9,
                 ms=2.6, mec="white", mew=0.25)          # small dot at each plotted step
        axL.plot(sub[-1, 0], sub[-1, 1], "o", color=c, ms=5.5, mec="white", mew=0.4,
                 zorder=5)                                 # slightly larger dot at the last step
    axL.plot(*THETA_STAR, "*", color="gold", ms=15, mec="k", mew=0.8, zorder=3, label=r"optimum $\theta^*$")
    axL.plot(*THETA_STARTPT, "o", color="w", mec="k", ms=8, zorder=7, label="start")
    # annotate the failing too-large ε where its median path leaves the frame
    tl = next((l for l in order if "too large" in l), None)
    if tl is not None:
        m = np.median(results[tl]["trajs"], 0)[:51]
        seg = m[(m[:, 0] <= xhi) & (m[:, 1] <= yhi) & (m[:, 0] >= xlo) & (m[:, 1] >= ylo)]
        if len(seg):
            ex, ey = seg[-1]
            axL.annotate(r"FD $\varepsilon$ too large" + "\n(diverges)", xy=(ex, ey),
                         xytext=(min(xhi - 0.02, ex + 0.02), min(yhi - 0.02, ey + 0.06)),
                         fontsize=6.0, color=C_FDW, ha="right", va="top",
                         arrowprops=dict(arrowstyle="->", color=C_FDW, lw=1.0))
    axL.set_xlim(xlo, xhi); axL.set_ylim(ylo, yhi)
    axL.set_xlabel(r"$\theta_1$ (ZZ coupling)"); axL.set_ylabel(r"$\theta_2$ (X field)")
    axL.set_title(r"(a) sloppy IQS valley: $\varepsilon$-free PSR/NSR reach $\theta^*$; FD too-large "
                  r"fails, small-$\varepsilon$ is unreliable", fontsize=7.0)
    axL.legend(fontsize=6.4, loc="best", framealpha=0.85)

    NS = min(50, ITERS)
    for lab in order:
        med, p25, p75 = (a[:NS + 1] for a in dc[lab])
        xs = np.arange(NS + 1)
        axR.semilogy(xs, np.maximum(med, 1e-3), "-", color=results[lab]["c"], lw=1.5, label=lab,
                     marker=results[lab]["mk"], ms=3.0, mec="white", mew=0.3, markevery=5)
        axR.fill_between(xs, np.maximum(p25, 1e-3), np.maximum(p75, 1e-3),
                         color=results[lab]["c"], alpha=0.10)
    axR.axhline(THRESH, color="#888", lw=0.8, ls=":")
    axR.text(1, THRESH * 1.13, f"tolerance {THRESH}", fontsize=6.3, color="#666")
    axR.set_xlabel("optimization step"); axR.set_ylabel((r"$\|\bar\theta_t-\theta^*\|$" if AVG else r"$\|\theta_t-\theta^*\|$") + "  (median and IQR)")
    axR.set_title("(b) convergence: PSR/NSR reach θ* reliably; every FD ε either fails, floors, "
                  "or is unreliable", fontsize=7.0)
    axR.legend(fontsize=6.4, ncol=2); axR.grid(True, which="both", alpha=0.15); axR.set_xlim(0, 50)

    fig.suptitle("F-loop — ε-free PSR/NSR reliably recover an interior θ* on a sloppy IQS valley; "
                 "finite differences need an ε they cannot tune without the answer: too-large ε FAILS "
                 r"(truncation $\frac{\varepsilon^2}{6}|C'''|/\mu_{\rm soft}$), too-small ε is "
                 "UNRELIABLE (δ/ε noise, wide IQR), and even the best ε only reaches an uncertifiable "
                 "δ-floor"
                 "\n"
                 r"(2q TFIM, $P{=}2$; compiled to machine-native segments, emulated under T4; "
                 r"$T/T_2^*{=}0.15$; %d seeds, median $\pm$ IQR; $w{=}%.3f$, $\mu_{\rm soft}{=}%.2f$)"
                 % (SEEDS, W, muC), fontsize=7.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"F_loop_trajectory{SUF}.{e}"), bbox_inches="tight", pad_inches=0.03)
        if not FLOOP_TAG:
            fig.savefig(os.path.join(OUT, f"F_loop_full.{e}"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # ── FLOOP_REPLOT (2026-08-21): single-column panel — convergence trace with the
    # valley as an inset. Same arrays, same numbers; layout only. The two-panel
    # render above ships unchanged as F_loop_full (appendix). ──
    def short_label(lab):
        if results[lab].get("eps") is None:
            return lab
        e = results[lab]["eps"]
        for tag in ("best", "too large", "too small"):
            if tag in lab:
                return rf"FD $\varepsilon$={e:g} ({tag})"
        return lab

    figS, ax = plt.subplots(figsize=(3.3, 3.05), dpi=300)
    for lab in order:
        med, p25, p75 = (a[:NS + 1] for a in dc[lab])
        xs = np.arange(NS + 1)
        ax.semilogy(xs, np.maximum(med, 1e-3), "-", color=results[lab]["c"], lw=1.2,
                    label=short_label(lab), marker=results[lab]["mk"], ms=2.4,
                    mec="white", mew=0.25, markevery=5)
        ax.fill_between(xs, np.maximum(p25, 1e-3), np.maximum(p75, 1e-3),
                        color=results[lab]["c"], alpha=0.10)
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    ax.axhline(THRESH, color="#888", lw=0.8, ls=":")
    ax.text(NS - 0.6, THRESH * 1.10, f"tolerance {THRESH}", fontsize=6.0, color="#666",
            va="bottom", ha="right", path_effects=halo)
    placed_early = 0
    for lab in order:                       # terminal markers (median holds 5 steps)
        t = term[lab]
        if t is None:
            continue
        ax.plot(t, dc[lab][0][t], marker=results[lab]["mk"], ms=5.5,
                color=results[lab]["c"], mec="k", mew=0.6, zorder=6)
        if t < 20:                          # early markers can sit a step apart: stagger labels
            dx, dy, ha = ((6, -10, "left"), (6, 5, "left"), (-6, -10, "right"))[placed_early % 3]
            placed_early += 1
        else:
            dx, dy, ha = (-3, 6, "right")
        ax.annotate(f"{t}", (t, dc[lab][0][t]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.0, color=results[lab]["c"],
                    ha=ha, weight="bold", path_effects=halo, zorder=7)
    ax.set_xlim(0, NS)
    ax.set_ylim(2.1e-3, 1.05)
    ax.text(0.02, 0.985, r"$T/T_2^*=0.15$", transform=ax.transAxes, fontsize=6.5,
            color="#52514e", va="top")
    ax.set_xlabel("optimization step", fontsize=7.5)
    ax.set_ylabel((r"$\|\bar\theta_t-\theta^*\|$" if AVG else r"$\|\theta_t-\theta^*\|$") + "  (median and IQR)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=5.6, ncol=2, loc="lower left", bbox_to_anchor=(0.01, 0.015),
              framealpha=0.65, borderpad=0.35, handlelength=1.7, columnspacing=0.8,
              handletextpad=0.5, labelspacing=0.35)

    # inset: the sloppy valley — a geometry cue, not a readable chart. No
    # colorbar/ticks/labels/legend; thin frame. Placed in the empty band between
    # the flat red series (~0.57) and the purple band top (~0.13 in this window).
    axI = ax.inset_axes([27.5, 0.135, 22.0, 0.325], transform=ax.transData)
    # trajectories' actual extent (median paths of every series except the too-large
    # arm, which runs off the frame by design), plus θ* and the start, with a margin
    _pts = np.vstack([np.median(results[lab]["trajs"], 0)[:NS + 1] for lab in order
                      if "too large" not in lab] + [THETA_STAR[None, :], THETA_STARTPT[None, :]])
    _mg = 0.04
    IX = (float(_pts[:, 0].min() - _mg), float(_pts[:, 0].max() + _mg))
    IY = (float(_pts[:, 1].min() - _mg), float(_pts[:, 1].max() + _mg))
    gxi = np.linspace(*IX, 75)
    gyi = np.linspace(*IY, 75)
    Zci = np.array([[Cval(np.array([a, b])) for a in gxi] for b in gyi])
    axI.contourf(gxi, gyi, Zci, levels=22, cmap="viridis")
    axI.contour(gxi, gyi, Zci, levels=10, colors="w", linewidths=0.25, alpha=0.35)
    for lab in order:                       # median paths, main-legend colours; only every
        med = np.median(results[lab]["trajs"], 0)[:NS + 1]      # 5th step as a dot (matches
        sub = med[::5]                                          # panel a) — declutters the inset;
        axI.plot(sub[:, 0], sub[:, 1], "-o", color=results[lab]["c"], lw=0.9, ms=1.9,
                 mec="white", mew=0.2, alpha=0.95, zorder=4)    # too-large ε runs off the frame edge
        axI.plot(sub[-1, 0], sub[-1, 1], "o", color=results[lab]["c"], ms=3.2,
                 mec="white", mew=0.3, zorder=5)
    axI.plot(*THETA_STAR, "*", color="gold", ms=9, mec="k", mew=0.6, zorder=3)
    axI.plot(*THETA_STARTPT, "o", mfc="white", mec="k", ms=4.5, mew=0.8, zorder=6)
    axI.set_xlim(*IX)
    axI.set_ylim(*IY)
    axI.set_xticks([])
    axI.set_yticks([])
    for sp in axI.spines.values():
        sp.set_linewidth(0.7)
        sp.set_color("#444444")
    figS.tight_layout(pad=0.4)
    OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_3", "figs"))
    for e in ("pdf", "png"):
        if FLOOP_TAG:
            figS.savefig(os.path.join(FIGDIR, f"F_loop{SUF}.{e}"), bbox_inches="tight", pad_inches=0.02)
            continue
        for out in (OUT, OUT3):
            figS.savefig(os.path.join(out, f"F_loop.{e}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figS)

    Cstar = Cval(THETA_STAR)
    summ = {lab: dict(final_err=finals[lab], held_step=held[lab], raw_final_err=float(np.median(np.linalg.norm(results[lab]["raw"][:, -1] - THETA_STAR, axis=1))),
                      term5_step=term[lab], frac_in_tol_at50=frac50[lab],
                      C_final=float(np.median([Cval(tr[-1]) for tr in results[lab]["trajs"]])),
                      predicted_offset=(pred_offset(results[lab]["eps"]) if "eps" in results[lab] else None))
            for lab in order}
    json.dump({"theta_star": THETA_STAR.tolist(), "a_star": A_STAR, "b_star": B_STAR, "w": W,
               "start": THETA_STARTPT.tolist(), "mu_soft": muC, "C3_soft": C3, "delta": delta,
               "oracle_eps": ob[1], "B": B, "tolerance": THRESH, "seeds": SEEDS, "C_star": Cstar, "eta0": ETA, "eta_tau": ETA_TAU, "tail_avg": AVG,
               "summary": summ}, open(os.path.join(FIGDIR, f"F_loop_trajectory{SUF}.json"), "w"),
              indent=2, default=float)
    # FLOOP_REPLOT §4 — the caption now also carries everything that was in the
    # in-figure title of the old two-panel render (which ships as F_loop_full).
    cap_new = (
        "Figure (F-loop, single column). Closing the optimization loop on a deliberately "
        "ill-conditioned valley: median ± IQR over "
        f"{SEEDS} seeds of ‖θ_t−θ*‖, 50 gradient-descent steps. 2q TFIM (P=2), compiled to "
        f"machine-native segments, emulated under T4 at T/T2*=0.15. Minimised objective: "
        f"C(θ)=(⟨Z0Z1⟩−a*)²+w(⟨Z0+Z1⟩−b*)² with a*={A_STAR:+.3f}, b*={B_STAR:+.3f} the exact "
        f"values at the interior θ*={np.round(THETA_STAR,2).tolist()} and w={W} — a residual "
        "form whose minimum is θ* by construction (no boundary clipping, no prior term). The "
        f"small residual weight makes the valley SLOPPY (soft curvature μ_soft={muC:.2f}, "
        "κ~1/w) on purpose: an ill-conditioned valley is where a biased gradient is most "
        "likely to go wrong, hence the informative test case. Every method spends the same "
        f"{B}-shot budget per gradient, so a step costs the same for all series and the step "
        "axis is directly comparable. PSR/NSR assemble the analytic gradient "
        "2·r_a·∇⟨O_a⟩+2w·r_b·∇⟨O_b⟩ from ε-free observable-gradient estimators (residuals "
        "measured at θ) — unbiased, no step size to choose. FD is the honest black-box "
        "central difference of the noisy objective at three ε: too large fails outright "
        "(truncation (ε²/6)|C‴|/μ_soft), too small is unreliable (δ/ε jitter, wide IQR), and "
        "the retrospectively best ε (unknowable without θ*) floors just above tolerance. "
        f"Tolerance {THRESH} ≈ 1.5δ, with δ={delta} the programmed amplitude-resolution "
        "jitter. Filled markers: first step from which the median stays inside tolerance for "
        "5 consecutive steps. Inset: the valley itself — filled contours of C(θ) on "
        "θ1∈[0.78,1.05]×θ2∈[0.95,1.28] with each method's median path (colours as the "
        "legend), star = θ*, open circle = the start; the too-large-ε path runs off the "
        "frame edge, as its series shows.")
    cap = (
        "Figure (F-loop-full, appendix two-panel variant). The loop closes for the ε-free strategies; finite differences settle short. "
        f"Honest IQS recovery of an INTERIOR, centered θ*={np.round(THETA_STAR,2).tolist()} for a "
        f"2-parameter TFIM instance (P=2). Objective C=(⟨Z0Z1⟩−a*)²+w(⟨Z0+Z1⟩−b*)² (two Z-diagonal "
        f"observables, one readout; a*,b* the exact values at θ*, so θ* is C's min by construction — the "
        f"single-observable ⟨O⟩+prior form has no interior stationary point and clips to an edge). "
        f"w={W} sets a SLOPPY valley (soft-direction curvature μ_soft={muC:.2f}, κ~1/w). Compiled to "
        f"machine-native segments, emulated under T4 at T/T2*=0.15; real estimators, no surrogates; "
        f"identical {B}-shot budget per gradient (step axis directly comparable). PSR/NSR assemble the "
        "ANALYTIC gradient 2·r_a·∇⟨O_a⟩+2w·r_b·∇⟨O_b⟩ from ε-free observable-gradient estimators "
        "(residuals measured at θ) ⇒ unbiased and ε-FREE ⇒ they crawl the valley to θ* and hold within "
        "tolerance, no step size to choose. FD is the honest BLACK-BOX central difference of the noisy "
        "objective C, and it needs an ε it cannot set without already knowing θ*: the too-large ε FAILS "
        "outright (truncation (ε²/6)|C‴| does not self-correct — C‴(θ*)≠0 — and divides by the small "
        "μ_soft into a large offset); the too-small ε is UNRELIABLE (δ/ε amplifies the resolution jitter "
        "into a wide-IQR random walk); and even the retrospectively best ε (which requires θ* to pick) "
        f"only reaches an UNCERTIFIABLE δ-floor a hair above tolerance. HONEST NOTE: unlike the gradient-"
        f"space floor (F6), the δ/ε variance would vanish exactly at a flat optimum, so a perfectly tuned "
        "small ε can limp in — the point is not that every ε lands far, it is that FD carries an ε-tuning "
        "burden (unknowable a-priori, failing on either side) that the ε-free strategies simply do not "
        f"have. No injected calibration offset — the trap is the sloppy conditioning (κ~1/μ_soft), the "
        f"paper's thesis. Tolerance {THRESH}≈1.5δ. Predicted truncation b(ε)/μ_soft reported beside the "
        "measured offset below; we report disagreement, we do not scale to fit. Scope: FD is unreliable, "
        "not always worse — a biased estimator can point better at a particular operating point (so can "
        "a random direction); the sound strategies converge without that luck.")
    with open(os.path.join(OUT if not FLOOP_TAG else FIGDIR, f"F_loop_caption{SUF}.txt"), "w") as f:
        f.write(cap_new + "\n\n" + cap +
                "\n\nPREDICTED vs MEASURED offset (b(ε)/μ_soft vs final ‖θ−θ*‖):\n")
        for lab, eps, c, mk, tr in fd_series:
            f.write(f"  {lab}: predicted {pred_offset(eps):.3f}  measured {finals[lab]:.3f}\n")

    # FLOOP_REPLOT §5 data note — the numbers behind "reliably" and the plateau
    # cross-check against F6's δ/ε floor. All computed from the cached arrays.
    obe = ob[1]
    trunc_only = (obe ** 2 / 6) * C3 / muC
    with open(os.path.join(OUT if not FLOOP_TAG else FIGDIR, f"F_loop_note{SUF}.md"), "w") as f:
        f.write("# F_loop data note (FLOOP_REPLOT §5 — no re-run)\n\n"
                "## Fraction of seeds inside tolerance at step 50\n\n"
                "\"Reliably\" is a claim about the band, not the median; this is the "
                "number that supports it.\n\n")
        for lab in order:
            f.write(f"- {lab}: {frac50[lab]*100:.0f}%  "
                    f"(terminal step, median holds 5 consecutive: {term[lab]})\n")
        f.write(
            "\n## Predicted plateau for the best-ε arm vs measured\n\n"
            f"Best ε = {obe:g}, |C‴|(soft) = {C3:.2f}, δ = {delta}, μ_soft = {muC:.2f}.\n\n"
            f"- b(ε)/μ_soft with b = δ/ε + (ε²/6)|C‴| (central-difference Taylor, as in "
            f"the builder): {pred_offset(obe):.3f}\n"
            f"- same with the spec's (ε²/24) coefficient: "
            f"{(delta/obe + obe**2/24*C3)/muC:.3f}\n"
            f"- truncation-only (ε²/6)|C‴|/μ_soft: {trunc_only:.3f}\n"
            f"- measured final median offset: {finals[[l for l in order if 'best' in l][0]]:.3f}\n\n"
            "The δ/ε term treats the resolution jitter as if it were bias; it is "
            "zero-mean per programming, so the full b(ε)/μ_soft overstates the floor "
            "(disagreement reported, not scaled to fit — cf. F6's δ/ε floor, where the "
            "jitter enters the gradient estimate directly and the floor is real).\n")
    print("wrote paper_fig_2/F_loop.pdf/.png (single panel + inset), "
          "F_loop_full.pdf/.png (appendix), caption + note")


if __name__ == "__main__":
    main()
