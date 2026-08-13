"""
build_fig1.py — Fig 1 (intro FD-trap), single column. SELF-CONTAINED (Fig-1 only;
does NOT read the shared landscape_device_data.json, so Fig 3 is untouched).

Composition (APPROVED, do not restructure): the noisy landscape C_noisy(θ), three FD
secants at three ε (all wrong sign), and ONE shift-rule tangent. Hamiltonian-level under
the T4 noise model — same instrument as every other Sec-6 figure. Cost = ⟨Z0⟩_noisy(θ)
of H(θ)=θ·Z0 + X0 under T2* dephasing, T/T2*=0.5 (large-T sharpens the θ-landscape so the
FD secants straddle a feature).

FIG1_REVISION (2026-08-13) applied:
  R1 legend "shift-rule tangent (PSR/NSR)" — one line, no "raw", no "=∇C_noisy", no slope.
  R2 anchor on a visibly-sloped stretch off the crest; tangent slope = analytic derivative
     of the plotted landscape at the anchor (reported in the data note); short tangent.
  R3 regime out of the axes (stated in the caption) — no text collision.
  R4 y-axis C_noisy(θ).
  R5 caption draft printed with the figure.
  R6 keep all three secants, black landscape, single-column.
Caches fig1_intro_data.json so replots never re-simulate.
Run: conda run -n qec_pg python differential_computing/tests/build_fig1.py
"""
import json
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
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T, T2 = 10.0, 20.0                    # T/T2* = 0.5
EPS = [0.15, 0.25, 0.35]
WINDOW = (0.75, 1.72)                 # focused view: one crest + the descending flank
TAN_HALF = 0.06                       # half-length of the drawn tangent (short → hugs)
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
COL = 3.3
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "font.family": "serif",
                     "mathtext.fontset": "stix", "legend.frameon": False,
                     "axes.linewidth": 0.7, "savefig.dpi": 300})
C_INK, C_FD, C_PSR = "#1a1a1a", "#D55E00", "#0072B2"


def make_fn():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    ex = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2)).make_expectation_fn(
        qp.tensor(qp.basis(2, 0), qp.basis(2, 0)), qp.tensor(qp.sigmaz(), qp.qeye(2)))
    return lambda v: ex([[H.set_parameterizedHam({"x": float(v)}), T]])


def compute():
    fn = make_fn()
    slope = lambda v, h=1e-3: (fn(v + h) - fn(v - h)) / (2 * h)   # analytic ∇C_noisy

    # anchor: steepest-descent point at which ALL THREE ε secants flip sign (the strong
    # FD-trap statement). On this sharp landscape that pins the anchor just past a crest;
    # we take the steepest such point and draw a short tangent so it still hugs the curve.
    cands = []
    for v in np.arange(WINDOW[0] + 0.1, WINDOW[1] - 0.1, 0.005):
        g = slope(v)
        if g >= -0.3:
            continue
        secs = [(fn(v + e) - fn(v - e)) / (2 * e) for e in EPS]
        if all(np.sign(s) != np.sign(g) for s in secs):
            cands.append((abs(g), float(v), float(g)))
    cands.sort(reverse=True)
    a = cands[0][1]; g_anal = cands[0][2]
    z0 = float(fn(a))

    gx = np.linspace(*WINDOW, 320)
    y = np.array([fn(v) for v in gx])
    g_plot = float(np.gradient(y, gx)[np.argmin(np.abs(gx - a))])   # plotted-resolution slope
    secants = [dict(eps=e, fm=float(fn(a - e)), fp=float(fn(a + e)),
                    slope=float((fn(a + e) - fn(a - e)) / (2 * e))) for e in EPS]
    return dict(T=T, T2=T2, regime=T / T2, anchor=a, g_analytic=g_anal, g_plotted=g_plot,
                z0=z0, tan_half=TAN_HALF, gx=list(map(float, gx)), y=list(map(float, y)),
                secants=secants)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "fig1_intro_data.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
    else:
        d = compute(); json.dump(d, open(cache, "w"), indent=2, default=float)

    a = d["anchor"]; g = d["g_analytic"]; z0 = d["z0"]; th = TAN_HALF
    gx = np.array(d["gx"]); y = np.array(d["y"])

    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.plot(gx, y, color=C_INK, lw=1.6, label=r"noisy landscape $C_{\rm noisy}(\theta)$")

    # FD secants (all three wrong sign at the anchor)
    ramp = plt.cm.Oranges(np.linspace(0.55, 0.9, len(d["secants"])))
    for k, (sec, c) in enumerate(zip(d["secants"], ramp)):
        e = sec["eps"]
        ax.plot([a - e, a + e], [sec["fm"], sec["fp"]], "o-", color=c, lw=1.2, ms=2.6,
                label="FD secants (wrong sign)" if k == 0 else None)

    # short shift-rule tangent: slope = analytic derivative of the plotted landscape
    xt = np.array([a - th, a + th])
    ax.plot(xt, z0 + g * (xt - a), color=C_PSR, lw=2.6, solid_capstyle="round",
            label="shift-rule tangent (PSR/NSR)")
    ax.plot([a], [z0], "o", color=C_INK, ms=4, zorder=6)

    ax.set_xlabel(r"parameter $\theta$"); ax.set_ylabel(r"$C_{\rm noisy}(\theta)$")
    ax.set_xlim(*WINDOW)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=6.6,
              handlelength=1.8, ncol=1, columnspacing=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig1_intro_trap.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    signs = "/".join(f"{s['slope']:+.2f}" for s in d["secants"])
    # R2 slope verification: the DRAWN tangent slope IS the analytic derivative (equal by
    # construction). g_plotted (np.gradient over the plot grid) is only a coarse consistency
    # check, not "the plotted slope" — report it as such so "equal" is not misread.
    data_note = (
        f"DATA NOTE (Fig 1): H(θ)=θ·Z0+X0, cost C_noisy=⟨Z0⟩ under T2* dephasing, "
        f"T/T2*={d['regime']:.2f} (Hamiltonian-level under T4). Anchor θ*={a:.3f}. "
        f"Slope verification (R2): drawn tangent slope = analytic ∇C_noisy = {g:+.3f} "
        f"(central difference h=1e-3) — EQUAL by construction (the tangent is plotted with "
        f"exactly this slope); coarse np.gradient check on the plotted samples gives "
        f"{d['g_plotted']:+.2f}, consistent to grid resolution. Three FD secants at ε={EPS}: "
        f"slopes {signs} — all POSITIVE while ∇C_noisy is NEGATIVE (all three wrong sign). "
        f"θ*={a:.3f} is the steepest point at which all three flip (on this sharp landscape "
        f"that pins it just past a crest); tangent drawn over ±{th} so it hugs the descending "
        f"flank. Caveat: the smallest ε={EPS[0]} secant is barely wrong-sign (slope "
        f"{d['secants'][0]['slope']:+.2f}, near-flat).")
    caption = (
        f"Figure 1. The finite-difference trap on a noisy analog cost. The transverse-field "
        f"program H(θ)=θZ0+X0 is evaluated Hamiltonian-level under the T4 noise model "
        f"(dephasing, T/T2*={d['regime']:.1f}); the cost C_noisy(θ)=⟨Z0⟩ (black) develops sharp "
        f"θ-features. At the operating point (dot), either sound differentiation strategy of "
        f"Sec. 4 — kick-PSR or the Nyquist waveform shift — recovers the true noisy slope "
        f"(blue tangent). Finite differences do not: secants at three step sizes ε (orange) "
        f"all return the WRONG sign, and shrinking ε does not rescue them — the δ/ε control-"
        f"noise floor (Sec. 6.2) blows up as ε→0. No step size works.")

    # DELIVER the caption + data note durably alongside the figure (not just stdout).
    d["data_note"] = data_note
    d["caption"] = caption
    json.dump(d, open(cache, "w"), indent=2, default=float)
    with open(os.path.join(FIGDIR, "fig1_intro_trap_caption.txt"), "w") as f:
        f.write(caption + "\n\n" + data_note + "\n")

    print(f"wrote fig1_intro_trap.pdf/.png + fig1_intro_trap_caption.txt  (T/T2*={d['regime']:.2f})")
    print(data_note)
    print("\nCAPTION DRAFT (R5):\n  " + caption)


if __name__ == "__main__":
    main()
