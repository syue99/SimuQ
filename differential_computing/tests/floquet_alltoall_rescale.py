"""
floquet_alltoall_rescale.py — does the rescale survive a Floquet-compiled
uniform all-to-all program?  (schedule-aware correction test, n=4)

TARGET (source program): H(θ) = θ·Σ_{i<j} Z_iZ_j (6 pairs, uniform all-to-all)
+ Σ X_i, evolved for logical time T.  No light cone exists (every qubit is one
strength-1 hop from the observable) — and the tweezer platform can't hold all
6 pairs at interaction distance simultaneously, so the COMPILED program is a
FLOQUET schedule: per period, an X slice then the 6 pair slots executed
sequentially at the gate zone, with transport ramps between them.  Parameter θ
lives in the pair-slot amplitudes J_p = θ·Δ/τ_slot (fixed slot duration).

THE POINT: noise integrates along the WALL-CLOCK schedule (ramps included),
which is ~6-7x longer than T.  A source-level rescale (ideal continuous target,
exposure T) must undercorrect.  The schedule-aware rescale computes the same
first-order Lindblad integral along the COMPILED ideal schedule with the
ledger's wall-clock durations — the correction as a compiler pass.

Compared quantities (exact expectations, no shots, deterministic):
  g_target   : ideal gradient of the CONTINUOUS source program (Floquet-error ref)
  g_sched    : ideal gradient of the compiled schedule  (= ground truth here)
  g_noisy    : fine-ε FD on the noisy schedule (= λ_exact·g_sched)
  PSR raw    : per-slot midpoint kick insertions (lemma check: ≈ g_noisy)
  raw·f_src  : naive source-level rescale (analytic_rescale on continuous H, T)
  raw·f_sched: schedule-aware rescale (first-order integral along the schedule)

Run:  conda run -n qec_pg python differential_computing/tests/floquet_alltoall_rescale.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analytic_rescale as ar

N = 4
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]   # 6 pairs
T_EFF = 1.5          # logical evolution time of the source program
NF = 4               # Floquet periods
DELTA = T_EFF / NF   # logical time per period
TAU_SLOT = 0.1       # μs — fixed pair-slot duration (amplitude carries θ)
T_RAMP = 0.25        # μs — AOD transport ramp between pair configurations
T2 = 40.0            # μs — dephasing on ALL wall-clock segments
GAMMA = 1.0 / (2.0 * T2)   # NoiseModel convention: T2-only → Γ = 1/(2·T2)

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()


def emb(op, i):
    return qp.tensor([op if k == i else I2 for k in range(N)])


HX = sum(emb(X, i) for i in range(N))
ZZ = {(i, j): emb(Z, i) * emb(Z, j) for (i, j) in PAIRS}
H0 = 0.0 * emb(Z, 0)
OBS = ZZ[(0, 1)]
PSI0 = qp.tensor([qp.basis(2, 0)] * N)
C_OPS = [np.sqrt(GAMMA) * emb(Z, i) for i in range(N)]


def H_target(th):
    return th * sum(ZZ.values()) + HX


def build_schedule(th):
    """Compiled Floquet schedule: list of (H, duration, tag).
    tag = 'x' | 'ramp' | ('zz', i, j).  Wall-clock includes transport."""
    segs = []
    J = th * DELTA / TAU_SLOT
    for _ in range(NF):
        segs.append((HX, DELTA, "x"))
        for (i, j) in PAIRS:
            segs.append((H0, T_RAMP, "ramp"))
            segs.append((J * ZZ[(i, j)], TAU_SLOT, ("zz", i, j)))
    return segs


def wall_clock(segs):
    return sum(d for _, d, _ in segs)


# ── ideal (noiseless) evolution ───────────────────────────────────────────────

def ideal_expect(segs):
    psi = PSI0
    for H, d, _ in segs:
        psi = (-1j * H * d).expm() * psi
    return float(qp.expect(OBS, psi).real)


def g_of(f, th, h=1e-3):
    return (f(th + h) - f(th - h)) / (2 * h)


# ── noisy evolution (density matrix; kick segments are noiseless unitaries) ──

def noisy_expect(segs, kick=None):
    """kick = (position_index, U_kick): apply noiseless unitary AFTER segment
    `position_index` in the (already split) segment list."""
    rho = PSI0 * PSI0.dag()
    for idx, (H, d, _) in enumerate(segs):
        if d > 0:
            res = qp.mesolve(H, rho, [0.0, float(d)], c_ops=C_OPS)
            rho = res.states[-1]
        if kick is not None and idx == kick[0]:
            U = kick[1]
            rho = U * rho * U.dag()
    return float(qp.expect(OBS, rho).real)


# ── PSR raw on the schedule (per-slot midpoint insertion, deterministic) ─────

def psr_raw(th):
    """d<O>/dθ = Σ_slots Δ·(f₋ − f₊), kick inserted at each zz slot midpoint.
    Kick = exp(-i·(1+s·3/4)π·Z_iZ_j), noiseless (compiled CZ)."""
    segs = build_schedule(th)
    grad = 0.0
    for idx, (H, d, tag) in enumerate(segs):
        if not (isinstance(tag, tuple) and tag[0] == "zz"):
            continue
        i, j = tag[1], tag[2]
        split = (segs[:idx] + [(H, d / 2, tag), (H, d / 2, tag)] + segs[idx + 1:])
        fs = {}
        for s in (+1, -1):
            U = (-1j * (1 + s * 0.75) * np.pi * ZZ[(i, j)]).expm()
            fs[s] = noisy_expect(split, kick=(idx, U))
        grad += DELTA * (fs[-1] - fs[+1])
    return grad


# ── schedule-aware first-order correction ────────────────────────────────────

def dO_dGamma_schedule(th, pts_per_seg=6):
    """d<O>/dΓ|₀ integrated along the IDEAL compiled schedule (wall-clock):
    Σ_i ∫ [<χ_i(t)|O|χ_i(t)> − <O>(end)] dt, χ_i = U(end←t) Z_i U(t←0) ψ0."""
    segs = build_schedule(th)
    Us = [(-1j * H * d).expm() for H, d, _ in segs]
    # suffix propagators S[k] = U_{N-1}···U_k  (from segment k to the end)
    S = [None] * (len(segs) + 1)
    S[len(segs)] = qp.qeye([2] * N)
    for k in range(len(segs) - 1, -1, -1):
        S[k] = S[k + 1] * Us[k]
    psi_end = S[0] * PSI0
    O_end = float(qp.expect(OBS, psi_end).real)
    Zs = [emb(Z, i) for i in range(N)]

    total = 0.0
    psi_start = PSI0
    for k, (H, d, _) in enumerate(segs):
        ts = (np.arange(pts_per_seg) + 0.5) / pts_per_seg * d   # midpoint grid
        acc = 0.0
        for t in ts:
            psit = (-1j * H * t).expm() * psi_start
            tail = S[k + 1] * (-1j * H * (d - t)).expm()
            for Zi in Zs:
                chi = tail * (Zi * psit)
                acc += float(qp.expect(OBS, chi).real) - O_end
        total += acc * d / pts_per_seg
        psi_start = Us[k] * psi_start
    return total


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "floquet_alltoall_rescale_data.json")

    if os.path.exists(cache):
        r = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        segs0 = build_schedule(0.5)
        T_wall = wall_clock(segs0)
        print(f"n={N} all-to-all, {len(PAIRS)} pairs, NF={NF}, "
              f"T_logical={T_EFF}, T_wall={T_wall:.2f} μs, T2={T2} "
              f"(exposure ratio {T_wall / T2:.3f} vs source {T_EFF / T2:.4f})")

        # operating point: moderate |g| on the ideal schedule
        f_sched = lambda th: ideal_expect(build_schedule(th))
        cands = np.arange(0.1, 1.21, 0.05)
        gs_scan = [(float(th), g_of(f_sched, float(th))) for th in cands]
        th_star, g_sched = max(gs_scan, key=lambda p: abs(p[1]))
        print(f"θ* = {th_star:.2f}  g_sched = {g_sched:+.4f}")

        f_target = lambda th: float(qp.expect(
            OBS, (-1j * H_target(th) * T_EFF).expm() * PSI0).real)
        g_target = g_of(f_target, th_star)

        f_noisy = lambda th: noisy_expect(build_schedule(th))
        g_noisy = g_of(f_noisy, th_star)          # FD best (ε→0, ∞ shots)
        lam_exact = g_noisy / g_sched

        g_raw = psr_raw(th_star)                   # lemma check vs g_noisy

        # (a) naive source-level rescale: continuous target H, exposure T_EFF
        s_src = ar.lambda_slope(H_target, OBS, PSI0, T_EFF, N,
                                z_sites=range(N), theta=th_star, n_grid=120)
        f_src = ar.rescale_factor(s_src, T_EFF, T2)
        lam_src = 1.0 / f_src

        # (b) schedule-aware: first-order integral along the compiled schedule
        h = 1e-3
        dD = (dO_dGamma_schedule(th_star + h) -
              dO_dGamma_schedule(th_star - h)) / (2 * h)
        f_schd = float(np.exp(-GAMMA * dD / g_sched))
        lam_schd = 1.0 / f_schd

        # self-check: dD vs numeric small-Γ slope of the noisy gradient
        global C_OPS
        gam_chk = 1e-3
        C_OPS_save = C_OPS
        C_OPS = [np.sqrt(gam_chk) * emb(Z, i) for i in range(N)]
        g_chk = g_of(lambda th: noisy_expect(build_schedule(th)), th_star)
        C_OPS = C_OPS_save
        dD_num = (g_chk - g_sched) / gam_chk
        print(f"self-check d(dg)/dΓ: analytic {dD:+.4f} vs numeric {dD_num:+.4f}")

        r = dict(theta=th_star, T_wall=T_wall, T_eff=T_EFF, T2=T2,
                 g_target=g_target, g_sched=g_sched, g_noisy=g_noisy,
                 g_raw=g_raw, lam_exact=lam_exact,
                 lam_src=lam_src, lam_schd=lam_schd,
                 est_raw=g_raw, est_src=g_raw * f_src, est_schd=g_raw * f_schd,
                 dD=dD, dD_num=dD_num)
        json.dump(r, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    gs = r["g_sched"]
    rel = lambda x: abs(x - gs) / abs(gs)
    print(f"\nθ*={r['theta']:.2f}  T_wall={r['T_wall']:.2f} vs T_logical={r['T_eff']}")
    print(f"g_target(continuous) = {r['g_target']:+.4f}   "
          f"g_sched(Floquet ideal) = {gs:+.4f}   (coherent Floquet gap "
          f"{abs(r['g_target'] - gs) / abs(gs):.3f})")
    print(f"λ_exact = {r['lam_exact']:.4f}   λ_src-pred = {r['lam_src']:.4f}   "
          f"λ_sched-pred = {r['lam_schd']:.4f}")
    print(f"lemma check: PSR raw {r['g_raw']:+.4f} vs FD fine-ε {r['g_noisy']:+.4f}")
    rows = [("PSR raw / FD best", rel(r["est_raw"])),
            ("naive source rescale", rel(r["est_src"])),
            ("schedule-aware rescale", rel(r["est_schd"]))]
    for name, v in rows:
        print(f"  {name:>24}: rel bias {v:.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    names = [n for n, _ in rows]
    vals = [max(v, 1e-4) for _, v in rows]
    colors = ["#9e9e9e", "#d62728", "#1f77b4"]
    ax.bar(names, vals, color=colors, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("relative gradient bias  |estimate − g_sched| / |g_sched|")
    for k, v in enumerate(vals):
        ax.text(k, v * 1.15, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title(f"Uniform all-to-all (n=4), Floquet-compiled: correct along\n"
                 f"the COMPILED schedule, not the source program\n"
                 f"(T_wall={r['T_wall']:.1f} μs vs T={r['T_eff']};  "
                 f"λ: exact {r['lam_exact']:.3f}, source-pred {r['lam_src']:.3f}, "
                 f"schedule-pred {r['lam_schd']:.3f})",
                 fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    out = os.path.join(figdir, "floquet_alltoall_rescale.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
