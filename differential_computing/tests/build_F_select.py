"""
build_F_select.py — F-select (replaces F-phase; SEC6_FIGURE_REVISIONS_0821 §4).

ONE single-column figure on the program design space (P, k):
  P = # differentiated coefficients; k = # device-alphabet terms each
  coefficient touches (set by coefficient sharing in the program text).

Term alphabet = the device signature {X_a, Z_a, Z_aZ_b} (Eq. 5.2.1 family);
pools are ways of sharing coefficients inside it:
  general — X + Z + ZZ mixed (non-commuting), the fill + main contour;
  aligned — ZZ-only (mutually commuting, chi -> 1), appendix contour.
No XX/YY (chi depends on diam/touched-term structure, not Pauli identity;
Trotter-synthesised terms cost compilation, not shots). The chi = 1/m
telescoping mechanism is NOT drawable in this alphabet: its compressing
family Sum_j (Z_j - Z_{j+1}) has non-involutive generators (Assumption 4.7
fails — the "NSR-only" row), and commuting Pauli families are jointly
extremizable (chi = Theta(1)); covered by the theorem in 6.3 prose, not by
a synthesised fake pool.

Fill: min(N_PSR, N_NSR) executions-to-target (log, neutral gray bands —
hue is reserved for the winner washes: PSR blue #0072B2, NSR green #009E73,
the paper-wide strategy colors). Solid black contour: measured crossing.
The compile-time certificate prediction has no crossing on this plane
(caption/note finding, not a contour). Running TFIM instance of Fig. A
marked at (P=2, k=1) (per-bond theta_i); its global-theta rewrite at
(P=1, k=2) shows k is a property of the program text.

Hamiltonian level, no compilation, no noise (units R = 1, dt = 1;
target-independence: N_S = C_S^2 eps_t^-2 log(1/delta), so changing the
target rescales the colorbar and leaves the boundary fixed).

Outputs: figures/F_select_data.json (cache), figures/F_select.{png,pdf}
(main, general pool), figures/F_select_appendix.{png,pdf} (+aligned
contour), captions, F_select_data_note.md.

Run: conda run -n qec_pg python differential_computing/tests/build_F_select.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
CACHE = os.path.join(FIGDIR, "F_select_data.json")

NQ, T = 7, 1.0
PS = list(range(1, 21))
KS = list(range(1, 15))
SEEDS = 6
SIGMA_BOUND = np.sqrt(2.0)   # static per-branch bound: (1-f+^2)+(1-f-^2) <= 2

# ── quantum model (only for the MEASURED surfaces) ───────────────────────────


def build_alphabet():
    import qutip as qp
    X, Z, Iop = qp.sigmax(), qp.sigmaz(), qp.qeye(2)

    def op1(P, i):
        o = [Iop] * NQ
        o[i] = P
        return qp.tensor(o)

    def op2(P, i, Q, j):
        o = [Iop] * NQ
        o[i] = P
        o[j] = Q
        return qp.tensor(o)

    ops, labels = [], []
    for i in range(NQ):
        ops.append(op1(X, i)); labels.append(("X", i))
    for i in range(NQ):
        ops.append(op1(Z, i)); labels.append(("Z", i))
    for i, j in itertools.combinations(range(NQ), 2):
        ops.append(op2(Z, i, Z, j)); labels.append(("ZZ", i, j))
    return ops, labels


def branch_sigma(ops, U_half, O, states):
    """Measured mean per-branch shot std per alphabet term (operating point)."""
    import qutip as qp
    IF = qp.tensor([qp.qeye(2)] * NQ)
    sig = np.zeros(len(ops))
    for psi in states:
        st = U_half * psi
        for k, Gj in enumerate(ops):
            kp = (IF - 1j * Gj) / np.sqrt(2)
            km = (IF + 1j * Gj) / np.sqrt(2)
            fp = float(qp.expect(O, U_half * (kp * st)).real)
            fm = float(qp.expect(O, U_half * (km * st)).real)
            sig[k] += np.sqrt(max(0.0, (1 - fp ** 2) + (1 - fm ** 2)))
    return sig / len(states)


def draw_params(pool, k, P, labels, rng):
    """P parameters, each touching k alphabet terms with +-1 coefficients."""
    if pool == "general":
        cand = list(range(len(labels)))
    elif pool == "aligned":
        cand = [i for i, l in enumerate(labels) if l[0] == "ZZ"]
    else:
        raise ValueError(pool)
    params = []
    for _ in range(P):
        idx = rng.choice(cand, size=min(k, len(cand)), replace=False)
        params.append({int(j): float(rng.choice([-1.0, 1.0])) for j in idx})
    return params


def cell_costs(params, ops, sig):
    """Measured and predicted (static-certificate) execution counts."""
    G = len(ops)
    # NSR: sum over parameters of diam(A_l)^2 — measured (exact spectrum) and
    # predicted (certificate diam <= 2*Sum|v|).
    N_nsr = 0.0
    N_nsr_pred = 0.0
    for d in params:
        A = sum(c * ops[j] for j, c in d.items())
        e = A.eigenenergies()
        N_nsr += (e[-1] - e[0]) ** 2
        N_nsr_pred += (2.0 * sum(abs(c) for c in d.values())) ** 2
    # PSR with cross-parameter branch reuse: n_j = max_l S_l |c_j| sigma_j,
    # S_l = Sum_j' |c_j'| sigma_j'; predicted replaces sigma by sqrt(2).
    nj = np.zeros(G)
    nj_pred = np.zeros(G)
    for d in params:
        Sl = sum(abs(c) * sig[j] for j, c in d.items())
        Sl_pred = sum(abs(c) * SIGMA_BOUND for c in d.values())
        for j, c in d.items():
            nj[j] = max(nj[j], Sl * abs(c) * sig[j])
            nj_pred[j] = max(nj_pred[j], Sl_pred * abs(c) * SIGMA_BOUND)
    return N_nsr, 2.0 * nj.sum(), N_nsr_pred, 2.0 * nj_pred.sum()


def run_sweep():
    import qutip as qp
    rng0 = np.random.default_rng(0)
    ops, labels = build_alphabet()

    # operating point: alphabet-generated background + high-entropy states
    c = rng0.normal(size=len(ops)) * 0.4
    B = sum(ci * op for ci, op in zip(c, ops))
    U_half = (-1j * B * (T / 2)).expm()
    O = sum(ops[NQ + i] for i in range(NQ))     # Sum_a Z_a readout
    states = []
    for _ in range(4):
        v = rng0.normal(size=2 ** NQ) + 1j * rng0.normal(size=2 ** NQ)
        v /= np.linalg.norm(v)
        states.append(qp.Qobj(v.reshape(-1, 1), dims=[[2] * NQ, [1] * NQ]))
    sig = branch_sigma(ops, U_half, O, states)

    out = dict(meta=dict(NQ=NQ, T=T, Ps=PS, ks=KS, seeds=SEEDS,
                         sigma_mean=float(sig.mean()),
                         sigma_bound=float(SIGMA_BOUND),
                         alphabet=dict(X=NQ, Z=NQ, ZZ=NQ * (NQ - 1) // 2)))
    for pool in ("general", "aligned"):
        Zg = np.zeros((len(KS), len(PS)))       # measured log10(N_NSR/N_PSR)
        Zp = np.zeros((len(KS), len(PS)))       # predicted (static) log-ratio
        Fm = np.zeros((len(KS), len(PS)))       # log10 min(N) (measured)
        for a, k in enumerate(KS):
            for b, P in enumerate(PS):
                lr, lrp, lm = [], [], []
                for s in range(SEEDS):
                    rng = np.random.default_rng(97 * s + 3 * P + 11 * k)
                    params = draw_params(pool, k, P, labels, rng)
                    Nn, Npr, Nn_p, Npr_p = cell_costs(params, ops, sig)
                    lr.append(np.log10(Nn / max(Npr, 1e-12)))
                    lrp.append(np.log10(Nn_p / max(Npr_p, 1e-12)))
                    lm.append(np.log10(max(min(Nn, Npr), 1e-12)))
                Zg[a, b] = np.mean(lr)
                Zp[a, b] = np.mean(lrp)
                Fm[a, b] = np.mean(lm)
            print(f"  {pool}: k={k} done", flush=True)
        out[pool] = dict(Z=Zg.tolist(), Zpred=Zp.tolist(), logminN=Fm.tolist(),
                         nsr_win_frac=float((Zg < 0).mean()),
                         sign_disagree_frac=float(
                             (np.sign(Zg) != np.sign(Zp)).mean()))
    json.dump(out, open(CACHE, "w"), indent=1)
    print(f"cached: {CACHE}")


# ── figure ───────────────────────────────────────────────────────────────────

INK, SEC, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
C_PSR, C_NSR = "#0072B2", "#009E73"          # paper-wide strategy colors


def render(data, with_aligned, outname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    # neutral gray ramp for the cost fill — hue is reserved for the
    # winner washes, so magnitude never competes with identity
    grays = LinearSegmentedColormap.from_list(
        "costgray", ["#f4f3f0", "#43423f"])

    g = data["general"]
    Ps, ks = data["meta"]["Ps"], data["meta"]["ks"]
    Pg, Kg = np.meshgrid(Ps, ks)
    Z = gaussian_filter(np.array(g["Z"]), sigma=0.8)
    F = gaussian_filter(np.array(g["logminN"]), sigma=0.5)

    fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=300)
    fig.subplots_adjust(left=0.13, right=0.88, top=0.86, bottom=0.14)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRID)

    # discrete half-decade bands so a cell can be read off the colorbar
    lo = np.floor(F.min() * 2) / 2
    hi = np.ceil(F.max() * 2) / 2
    levels = np.arange(lo, hi + 0.25, 0.5)
    pc = ax.contourf(Pg, Kg, F, levels=levels, cmap=grays)
    cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("executions to target (best strategy)", fontsize=7, color=SEC)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"$10^{{{int(t)}}}$" for t in ticks])

    # transparent winner washes over the gray fill, split at the measured
    # crossing (solid black). The compile-time certificate prediction has
    # no crossing on this plane (it says PSR-or-tie everywhere: the 2Σ|v|
    # diameter certificate is loose for non-commuting mixed tangents) —
    # that finding lives in the caption and note, not as a degenerate
    # contour. The aligned-pool overlay, where the certificate is
    # near-tight, is in the appendix variant.
    ax.contourf(Pg, Kg, Z, levels=[-99.0, 0.0, 99.0],
                colors=[C_NSR, C_PSR], alpha=0.30, antialiased=True,
                zorder=3)
    ax.contour(Pg, Kg, Z, levels=[0.0], colors="k", linewidths=1.6, zorder=4)
    if with_aligned:
        Za = gaussian_filter(np.array(data["aligned"]["Z"]), sigma=0.8)
        ca = ax.contour(Pg, Kg, Za, levels=[0.0], colors=INK, linewidths=1.2,
                        linestyles="dashdot", zorder=4)
        ca.set(path_effects=[pe.withStroke(linewidth=2.6,
                                           foreground="#ffffff")])
        ax.annotate("dash-dot: aligned (ZZ-only) crossing —\n"
                    "alignment (χ→1) shrinks the NSR region",
                    xy=(0.29, 0.955), xycoords="axes fraction", fontsize=5.8,
                    color=INK, va="top", zorder=6,
                    bbox=dict(facecolor="#ffffff", alpha=0.75,
                              edgecolor="none", pad=1.5))

    # winner labels inside the washed regions, in the wash's own hue
    halo = [pe.withStroke(linewidth=2.0, foreground="#ffffff")]
    ax.text(0.13, 0.87, "NSR\nwins", transform=ax.transAxes, fontsize=8.5,
            weight="bold", color="#00654a", ha="center", va="center",
            path_effects=halo, zorder=5)
    ax.text(0.80, 0.13, "PSR\nwins", transform=ax.transAxes, fontsize=8.5,
            weight="bold", color="#00517e", ha="center", va="center",
            path_effects=halo, zorder=5)

    # running instance of Fig. A: per-bond theta_i -> (P=2, k=1); the same
    # physics with one shared global theta -> (P=1, k=2)
    ax.plot([2], [1], marker="*", ms=11, color="#eb6834",
            markeredgecolor=INK, markeredgewidth=0.6, zorder=6, clip_on=False)
    ax.annotate("TFIM instance (Fig. A)", xy=(2, 1), xytext=(8, 6),
                textcoords="offset points", fontsize=6.5, color="#a0451a")
    ax.plot([1], [2], marker="o", ms=5, markerfacecolor="none",
            markeredgecolor="#a0451a", markeredgewidth=1.0, zorder=6,
            clip_on=False)
    ax.annotate("same physics,\nglobal θ", xy=(1, 2), xytext=(6, 8),
                textcoords="offset points", fontsize=5.8, color="#a0451a")

    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("# differentiated coefficients  P", fontsize=8, color=SEC)
    ax.set_ylabel("# alphabet terms per coefficient  k", fontsize=8, color=SEC)
    ax.set_title("Strategy selection on the program design space\n"
                 "(device alphabet; Hamiltonian level, no compilation)",
                 fontsize=8, color=INK, pad=6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{outname}.{ext}"))
    plt.close(fig)


def certificate_forfeit(pool_data):
    """Max shot-cost factor forfeited by following the certificate where it
    disagrees with the measured winner. The certificate says PSR-or-tie on
    the whole sampled plane (Zpred >= -0.0 everywhere), so the divergent
    set is exactly the measured NSR-wins region."""
    Z = np.array(pool_data["Z"])
    Zp = np.array(pool_data["Zpred"])
    div = (Zp >= 0) & (Z < 0)
    return float(10 ** np.abs(Z[div]).max()) if div.any() else 1.0


def write_notes(data):
    g, al = data["general"], data["aligned"]
    meta = data["meta"]
    forfeit_g = certificate_forfeit(g)
    forfeit_a = certificate_forfeit(al)
    cap = (
        "F-select: which differentiation strategy needs fewer executions, on "
        "the program design space (P, k). P = number of differentiated "
        "coefficients; k = number of device-alphabet terms each coefficient "
        "touches — set by coefficient sharing in the program text, not by "
        "physics (per-bond θ_i gives k=1; one global θ over the same chain "
        "gives k=n−1). Alphabet = the device signature {X_a, Z_a, Z_aZ_b}; "
        "fill = min(N_PSR, N_NSR) to a fixed target (units R=1, Δτ=1; "
        "N_S = C_S²·ε_t⁻²·log(1/δ), so changing the target rescales the "
        "colorbar and leaves the boundary fixed). Shaded overlays mark the "
        "winner: blue = PSR needs fewer executions, green = NSR does; "
        "solid black: the measured crossing between them. The crossing "
        "computed at compile time from static "
        "certificates alone (C_NSR from the Assumption-4.4 spectral "
        "certificate 2Σ|v|; C_PSR from the coefficient expressions with the "
        "worst-case branch deviation √2) does NOT cross on this plane: for "
        "non-commuting mixed tangents the diameter certificate is loose, so "
        "the certificate-guided compiler picks PSR everywhere here — the "
        "safe direction (looseness costs shots, never bias, Remark B.1), "
        f"forfeiting at most {certificate_forfeit(g):.1f}× shots at any "
        "sampled point. Star: the TFIM instance "
        "of Fig. A (P=2 per-bond couplings, k=1); open circle: its "
        "global-coefficient rewrite. Operating point: high-entropy states, "
        f"mean per-branch shot std ⟨σ⟩≈{meta['sigma_mean']:.2f} "
        "(PSR-conservative; polarised states enlarge the PSR region). "
        "XX/YY terms are excluded: the selection constants depend on "
        "diam(A) and touched-term structure, not on which Pauli a term is, "
        "and Trotter-synthesised terms cost compilation, not shots. "
        f"NSR wins on {g['nsr_win_frac']*100:.0f}% of the sampled plane.\n")
    open(os.path.join(FIGDIR, "F_select_caption.txt"), "w").write(cap)

    note = f"""# F_select data note

Sweep: NQ={meta['NQ']}, T={meta['T']}, P in {meta['Ps'][0]}..{meta['Ps'][-1]},
k in {meta['ks'][0]}..{meta['ks'][-1]}, {meta['seeds']} seeds/cell (means of
log quantities). Alphabet: {meta['alphabet']['X']} X + {meta['alphabet']['Z']} Z
+ {meta['alphabet']['ZZ']} ZZ terms. Units R=1, dt=1 (caption states the
boundary is drawn in those units). No noise anywhere in this figure.

Cost model (executions to a fixed target, constants shared):
  N_NSR = Sum_l diam(A_l)^2          [measured: exact spectrum]
  N_PSR = 2 Sum_j max_l S_l|c_j|sig_j, S_l = Sum_j'|c_j'|sig_j'
          [measured: per-branch shot std sig_j at the operating point;
           cross-parameter branch reuse via the max]
Predicted (compile-time, static text only):
  diam -> 2 Sum|v| (Assumption 4.4 certificate), sig -> sqrt(2) (worst case).

## Certificate finding (the [B] predicted-vs-measured overlay)

The certificate-predicted surface (diam -> 2 Sum|v|, sig -> sqrt(2)) has
NO strict NSR region on either pool at this operating point — it says
PSR-or-tie everywhere (exact ties on the P=1 edge where terms cannot
overlap). The measured NSR-wins regions therefore ARE the divergence set:
general {g['nsr_win_frac']*100:.1f}% of cells, aligned
{al['nsr_win_frac']*100:.1f}%. The divergence is one-sided and safe
(Remark B.1: a loose certificate costs shots, never bias): following the
certificate forfeits at most {forfeit_g:.2f}x shots on the general pool
and {forfeit_a:.2f}x on the aligned pool — inside the 10x margin
everywhere, and mostly inside 2x. The looseness is the Assumption-4.4
diameter certificate: measured diam concentrates below 2 Sum|v| for
random-sign tangents (operator-norm concentration); the sqrt(2) branch
bound is nearly tight here (measured <sigma> = {meta['sigma_mean']:.2f}).
There is no meaningful predicted CROSSING to overlay, so the main figure
carries the finding in its caption instead of a degenerate contour; the
appendix draws the aligned pool's measured crossing (alignment, chi -> 1,
shrinks the NSR region — visible as the dash-dot boundary left of the
solid one).

Takeaway sentence for 6.3: the compiler can compute its choice before
running anything; where its certificate is loose the error is bounded
(<= {forfeit_g:.1f}x shots here) and always lands on the bias-free side.

NSR-win fraction (honesty rail): general {g['nsr_win_frac']*100:.1f}%,
aligned {al['nsr_win_frac']*100:.1f}% — NSR does win on a Pauli-only system
in the small-P, large-k corner (subextensive tangents).

Telescoping (chi = 1/m): NOT drawable in this alphabet. The compressing
family Sum_j (Z_j - Z_{{j+1}}) has non-involutive generators — precisely the
Assumption-4.7-fails, NSR-only regime — and commuting PAULI families are
jointly extremizable (chi = Theta(1); frustration gives constant factors,
never 1/m). This regime is covered by the theorem, not by measurement;
6.3 states it in one sentence together with the non-involutive table row.

The aligned (ZZ-only) crossing appears in the appendix variant
(F_select_appendix): coefficient alignment moves the boundary toward PSR
(chi -> 1 removes NSR's folding headroom).
"""
    open(os.path.join(FIGDIR, "F_select_data_note.md"), "w").write(note)


def main():
    if not os.path.exists(CACHE):
        run_sweep()
    data = json.load(open(CACHE))
    render(data, with_aligned=False, outname="F_select")
    render(data, with_aligned=True, outname="F_select_appendix")
    write_notes(data)
    g = data["general"]
    print(f"F_select written; NSR-win {g['nsr_win_frac']*100:.0f}%, "
          f"pred/meas sign agreement {(1-g['sign_disagree_frac'])*100:.0f}%")


if __name__ == "__main__":
    main()
