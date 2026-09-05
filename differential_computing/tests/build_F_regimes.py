"""
build_F_regimes.py — App G.4 (fig:regimes): who wins on the (p, q) plane for
three tangent families, on Figure 10's grid and with Figure 10's measurement.

Replaces phase_who_wins_3panel.py, which swept the OLD p in [1,20] x q in [1,14]
grid and called q "k".  Everything measured here is Figure 10's own machinery,
imported rather than re-implemented:

  operating point   selector_check.rebuild_operating_point() (verbatim
                    re-execution of build_F_select.run_sweep's background,
                    readout and 4 high-entropy states)
  cost model        build_F_select.cell_costs — N_NSR = sum_l diam(A_l)^2 with
                    the exact spectrum, N_PSR = 2 sum_j max_l S_l |c_j| sigma_j
                    (cross-parameter branch reuse)
  certificate       selector_check.omega_ac — Omega_AC = 2 sum_g ||v_g||_2 over
                    an anticommuting clique cover
  selector          eq:margin, choose NSR iff  Zpred_AC + log10(gamma(q)) < 0
                    with gamma(q) = min(1, GAMMA0/sqrt(q)), exponent PINNED to
                    sqrt(q) and GAMMA0 = 1.86 calibrated on Figure 10.

Three families (panels a/b/c):
  general      random tangents over the device signature {X_a, Z_a, Z_aZ_b}
  aligned      ZZ-only tangents (mutually commuting, chi -> 1)
  heisenberg   Heisenberg bonds XX+YY+ZZ per edge

ALPHABET NOTE, stated on the figure because it is a real caveat: the device
signature has no XX/YY, so the Heisenberg family is drawn from an EXTENDED
tangent alphabet (77 terms = 35 device + 21 XX + 21 YY).  Only panel (c)'s
tangents are extended; the background, the readout and the states stay on the
device alphabet, so panels (a) and (b) reproduce Figure 10's plane bit-exactly
(asserted in check_reproduces_fig10(), which passes at 2.2e-16).

Grouping gain, measured, because the naive expectation is wrong.  XX, YY and ZZ
on ONE bond mutually commute, so a tangent confined to a single bond (q <= 3)
groups into singletons and Omega_AC = Omega_L1 exactly.  Past one bond that
stops holding: terms on bonds SHARING a qubit anticommute (XX_01 vs YY_12
differ on site 1 alone), so cliques do form and the certificate tightens with q
— measured Omega_AC/Omega_L1 falls to 0.60 at q = 35.  The aligned family is
the one where grouping never helps (ZZ terms all commute: 1.000 at every q).

Outputs: figures/F_regimes_data.json (cache; delete or pass --recompute to
re-sweep), figures/F_regimes.{pdf,png} + paper_fig_3/figs/, and the per-family
selector table on stdout.

Run:      conda run -n qec_pg python differential_computing/tests/build_F_regimes.py
Replot:   ... build_F_regimes.py --replot          (never recomputes)
Probe:    ... build_F_regimes.py --probe           (times a 3-cell slice)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import build_F_select as bs
import selector_check as sc

FIGDIR = bs.FIGDIR
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))
CACHE = os.path.join(FIGDIR, "F_regimes_data.json")
BAL_CACHE = os.path.join(FIGDIR, "F_select_balanced_data.json")

PS = list(range(1, 11))          # p — differentiated coefficients (Fig 10 grid)
KS = list(range(1, 36))          # q — alphabet terms per coefficient
SEEDS = 6
GAMMA0 = 1.86                    # eq:margin constant, calibrated on Fig 10

FAMILIES = [
    ("general", "(a) general\ndevice signature "
                r"$\{X_a, Z_a, Z_aZ_b\}$"),
    ("aligned", "(b) aligned\nZZ-only tangents (commuting)"),
    ("heisenberg", "(c) Heisenberg bonds\n" + r"$XX{+}YY{+}ZZ$ per edge"),
]
C_PSR, C_NSR = "#0072B2", "#009E73"
INK, SEC, GRID, SURFACE = bs.INK, bs.SEC, bs.GRID, bs.SURFACE


# ── extended tangent alphabet (panel c only) ─────────────────────────────────

def build_extended_alphabet():
    """Device signature first (indices 0..34, identical to Fig 10), then the
    XX and YY bond operators appended.  Returns (ops, labels, edges).

    edges[(i, j)] = [idx_XX, idx_YY, idx_ZZ] — the Heisenberg triple per edge.
    """
    import itertools

    import qutip as qp
    ops, labels = bs.build_alphabet()          # X_a, Z_a, Z_aZ_b  (35 terms)
    X, Y, Iop = qp.sigmax(), qp.sigmay(), qp.qeye(2)

    def op2(P, i, Q, j):
        o = [Iop] * bs.NQ
        o[i] = P
        o[j] = Q
        return qp.tensor(o)

    zz_index = {}
    for idx, l in enumerate(labels):
        if l[0] == "ZZ":
            zz_index[(l[1], l[2])] = idx

    edges = {}
    pairs = list(itertools.combinations(range(bs.NQ), 2))
    xx_start = len(ops)
    for i, j in pairs:
        ops.append(op2(X, i, X, j))
        labels.append(("XX", i, j))
    for i, j in pairs:
        ops.append(op2(Y, i, Y, j))
        labels.append(("YY", i, j))
    for e, (i, j) in enumerate(pairs):
        edges[(i, j)] = [xx_start + e, xx_start + len(pairs) + e,
                         zz_index[(i, j)]]
    return ops, labels, edges


def label_to_xz(label):
    """Pauli bit-vectors for the extended alphabet (selector_check's, + XX/YY).

    Y = iXZ, so a Y site sets BOTH the x and the z bit.
    """
    if label[0] in ("X", "Z", "ZZ"):
        return sc.label_to_xz(label)
    bits = (1 << label[1]) | (1 << label[2])
    if label[0] == "XX":
        return (bits, 0)
    if label[0] == "YY":
        return (bits, bits)
    raise ValueError(label)


def draw_params(family, k, P, labels, edges, rng):
    """P tangents of q=k terms each, for one family.

    general / aligned delegate to build_F_select.draw_params VERBATIM (same
    candidate list, same rng call sequence), which is what makes panels (a)
    and (b) reproduce Figure 10 exactly.
    """
    if family in ("general", "aligned"):
        base = [l for l in labels if l[0] in ("X", "Z", "ZZ")]
        return bs.draw_params(family, k, P, base, rng)
    if family != "heisenberg":
        raise ValueError(family)
    edgelist = list(edges.keys())
    params = []
    for _ in range(P):
        d = {}
        for e in rng.permutation(len(edgelist)):
            for j in edges[edgelist[e]]:            # XX, YY, ZZ on this bond
                if len(d) >= k:
                    break
                d[int(j)] = 1.0
            if len(d) >= k:
                break
        params.append(d)
    return params


# ── sweep ────────────────────────────────────────────────────────────────────

def _cell(family, k, P, ops, labels, edges, xz, dense, sig, rng_restart):
    """One (p, q) cell: measured and certificate log-ratios, averaged over seeds."""
    meas, pac, pl1 = [], [], []
    ac_over_l1 = []
    for s in range(SEEDS):
        rng = np.random.default_rng(97 * s + 3 * P + 11 * k)
        params = draw_params(family, k, P, labels, edges, rng)

        N_true = N_l1 = N_ac = 0.0
        for d in params:
            idx = list(d.keys())
            w = [abs(d[j]) for j in idx]
            o_l1 = 2.0 * sum(w)
            o_ac = sc.omega_ac([xz[j] for j in idx], w, rng=rng_restart)
            A = sum(d[j] * dense[j] for j in idx)
            e = np.linalg.eigvalsh(A)
            o_tr = float(e[-1] - e[0])
            if not (o_l1 + 1e-9 >= o_ac >= o_tr - 1e-9):
                raise RuntimeError(f"certificate chain violated: family={family} "
                                   f"p={P} q={k} seed={s}: L1={o_l1} AC={o_ac} "
                                   f"true={o_tr}")
            N_true += o_tr ** 2
            N_l1 += o_l1 ** 2
            N_ac += o_ac ** 2
            ac_over_l1.append(o_ac / o_l1)

        # PSR side — line-identical to build_F_select.cell_costs
        nj = np.zeros(len(ops))
        nj_pred = np.zeros(len(ops))
        for d in params:
            Sl = sum(abs(c) * sig[j] for j, c in d.items())
            Sl_pred = sum(abs(c) * bs.SIGMA_BOUND for c in d.values())
            for j, c in d.items():
                nj[j] = max(nj[j], Sl * abs(c) * sig[j])
                nj_pred[j] = max(nj_pred[j], Sl_pred * abs(c) * bs.SIGMA_BOUND)
        N_psr = 2.0 * nj.sum()
        N_psr_pred = 2.0 * nj_pred.sum()

        meas.append(np.log10(N_true / N_psr))
        pac.append(np.log10(N_ac / N_psr_pred))
        pl1.append(np.log10(N_l1 / N_psr_pred))
    return (float(np.mean(meas)), float(np.mean(pac)), float(np.mean(pl1)),
            float(np.mean(ac_over_l1)))


def run_sweep(probe=False):
    ops, labels, edges = build_extended_alphabet()
    _o, _l, sig = sc.rebuild_operating_point()          # device-alphabet point
    # sigma for the appended XX/YY tangents, same background/readout/states
    import qutip as qp
    rng0 = np.random.default_rng(0)
    base_ops, _bl = bs.build_alphabet()
    c = rng0.normal(size=len(base_ops)) * 0.4
    B = sum(ci * op for ci, op in zip(c, base_ops))
    U_half = (-1j * B * (bs.T / 2)).expm()
    O = sum(base_ops[bs.NQ + i] for i in range(bs.NQ))
    states = []
    for _ in range(4):
        v = rng0.normal(size=2 ** bs.NQ) + 1j * rng0.normal(size=2 ** bs.NQ)
        v /= np.linalg.norm(v)
        states.append(qp.Qobj(v.reshape(-1, 1), dims=[[2] * bs.NQ, [1] * bs.NQ]))
    sig_ext = bs.branch_sigma(ops, U_half, O, states)
    if not np.allclose(sig_ext[:len(sig)], sig, atol=1e-12):
        raise RuntimeError("extended-alphabet sigma disagrees with Fig 10's "
                           "on the device terms — operating point drifted")

    xz = [label_to_xz(l) for l in labels]
    dense = [np.asarray(op.full()) for op in ops]
    rng_restart = np.random.default_rng(sc.RNG_RESTART)

    ks, ps = (KS[:3], PS[:1]) if probe else (KS, PS)
    out = dict(meta=dict(PS=ps, KS=ks, seeds=SEEDS, NQ=bs.NQ, T=bs.T,
                         gamma0=GAMMA0,
                         sigma_bound=float(bs.SIGMA_BOUND),
                         n_terms_device=len(sig),
                         n_terms_extended=len(ops),
                         sigma_mean_device=float(np.mean(sig)),
                         sigma_mean_extended=float(np.mean(sig_ext))))
    t0 = time.perf_counter()
    for family, _title in FAMILIES:
        Z = np.zeros((len(ks), len(ps)))
        Zac = np.zeros((len(ks), len(ps)))
        Zl1 = np.zeros((len(ks), len(ps)))
        Chi = np.zeros((len(ks), len(ps)))
        for a, k in enumerate(ks):
            for b, P in enumerate(ps):
                Z[a, b], Zac[a, b], Zl1[a, b], Chi[a, b] = _cell(
                    family, k, P, ops, labels, edges, xz, dense, sig_ext,
                    rng_restart)
            print(f"  {family}: q={k} done ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
        out[family] = dict(Z=Z.tolist(), Zpred_AC=Zac.tolist(),
                           Zpred_L1=Zl1.tolist(), ac_over_l1=Chi.tolist(),
                           nsr_win_frac=float((Z < 0).mean()))
    out["meta"]["sweep_seconds"] = time.perf_counter() - t0
    if not probe:
        json.dump(out, open(CACHE, "w"), indent=1)
        print(f"cached: {CACHE}  ({out['meta']['sweep_seconds']:.0f}s)")
    return out


# ── selector + checks ────────────────────────────────────────────────────────

def margin_column(ks):
    """eq:margin: log10 gamma(q), gamma(q) = min(1, GAMMA0/sqrt(q))."""
    q = np.asarray(ks, dtype=float)
    return np.log10(np.minimum(1.0, GAMMA0 / np.sqrt(q)))[:, None]


def selector_table(data):
    ks = data["meta"]["KS"]
    marg = margin_column(ks)
    rows = []
    for family, _t in FAMILIES:
        Z = np.array(data[family]["Z"])
        Zac = np.array(data[family]["Zpred_AC"])
        Zl1 = np.array(data[family]["Zpred_L1"])
        r = dict(family=family,
                 nsr_share=float((Z < 0).mean()),
                 ac_over_l1=float(np.mean(data[family]["ac_over_l1"])))
        for name, pred in (("L1", Zl1), ("AC", Zac), ("AC+margin", Zac + marg)):
            st = sc.selector_stats(Z, pred, name)
            r[f"agree_{name}"] = st["agreement"]
            r[f"maxforfeit_{name}"] = st["max_forfeit"]
            r[f"nsrchosen_{name}"] = st["nsr_frac_chosen"]
        # tie handling: cells where the certificate is an exact tie
        tie = np.abs(Zac + marg) <= 1e-9
        r["tie_cells"] = int(tie.sum())
        r["agree_ties_to_psr"] = float(
            ((Zac + marg < -1e-9) == (Z < 0)).mean())
        r["agree_ties_to_nsr"] = float(
            ((Zac + marg <= 1e-9) == (Z < 0)).mean())
        rows.append(r)
    return rows


def check_reproduces_fig10(data):
    """Panel (a) must reproduce Figure 10's published plane bit-exactly."""
    if not os.path.exists(BAL_CACHE):
        return None
    Zpub = np.array(json.load(open(BAL_CACHE))["general"]["Z"])
    Z = np.array(data["general"]["Z"])
    if Z.shape != Zpub.shape:
        return None
    return float(np.abs(Z - Zpub).max())


# ── figure ───────────────────────────────────────────────────────────────────

def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    ps, ks = data["meta"]["PS"], data["meta"]["KS"]
    Pg, Kg = np.meshgrid(ps, ks)
    marg = margin_column(ks)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]

    import build_F_select_balanced as bb          # Fig 10's colour scale and line styles
    plt.rcParams.update({"font.size": 7})
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.65), dpi=300, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    pc = None
    for ax, (family, title) in zip(axs, FAMILIES):
        Z = gaussian_filter(np.array(data[family]["Z"]), sigma=0.8)
        Zsel = gaussian_filter(np.array(data[family]["Zpred_AC"]) + marg,
                               sigma=0.8)
        ax.set_facecolor(SURFACE)
        pc = ax.contourf(Pg, Kg, np.clip(Z, -bb.RATIO_LIM + 1e-6, bb.RATIO_LIM - 1e-6),
                         levels=bb.RATIO_LEVELS, cmap=bb.ratio_cmap(), antialiased=True)
        ax.contour(Pg, Kg, Z, levels=[0.0], colors="k", linewidths=1.5, zorder=4)
        if Zsel.min() < 0 < Zsel.max():
            ax.contour(Pg, Kg, Zsel, levels=[0.0], colors="k",
                       linewidths=1.2, linestyles="dashed", zorder=4)
        if (np.array(data[family]["Z"]) < 0).mean() > 0.04:
            ax.text(0.30, 0.86, "NSR\nwins", transform=ax.transAxes,
                    color="#0f6b52", fontsize=8.5, weight="bold", ha="center",
                    path_effects=halo)
        if (np.array(data[family]["Z"]) > 0).mean() > 0.04:
            ax.text(0.80, 0.16, "PSR\nwins", transform=ax.transAxes,
                    color="#10507a", fontsize=8.5, weight="bold", ha="center",
                    path_effects=halo)
        ax.set_title(title, fontsize=7.2, color=INK)
        ax.set_xlabel("# differentiated coefficients  $p$", fontsize=7.2,
                      color=SEC)
        ax.tick_params(labelsize=6.5, colors=SEC)
        for s in ax.spines.values():
            s.set_color(GRID)
    axs[0].set_ylabel("# alphabet terms per coefficient  $q$", fontsize=7.2,
                      color=SEC)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color="k", lw=1.5, label="measured crossing"),
        Line2D([], [], color="k", lw=1.2, ls="dashed",
               label=r"compiler's selector: $\bar\Omega_{AC}$ with margin "
                     r"$\gamma(q)=\min(1,%.2f/\sqrt{q})$" % GAMMA0)],
        fontsize=6.4, frameon=False, loc="lower center", ncol=2,
        handlelength=2.0, borderpad=0.2, columnspacing=1.8,
        bbox_to_anchor=(0.5, -0.035))
    axs[2].text(0.985, 0.985,
                "panel (c) tangents leave the\ndevice signature (+XX, +YY)",
                transform=axs[2].transAxes, fontsize=5.6, color=SEC,
                ha="right", va="top", path_effects=halo)
    axs[1].text(0.985, 0.02, "Hamiltonian level, no noise",
                transform=axs[1].transAxes, fontsize=6.2, color=SEC,
                ha="right", va="bottom", path_effects=halo)

    fig.tight_layout(pad=0.4, rect=[0, 0.035, 0.925, 1])
    cax = fig.add_axes([0.935, 0.17, 0.011, 0.70])           # shared colour scale, own axis
    cb = fig.colorbar(pc, cax=cax, ticks=[-0.8, -0.4, 0.0, 0.4, 0.8])
    cb.set_label(r"$\log_{10}\,(N_{\rm NSR}\,/\,N_{\rm PSR})$", fontsize=7, color=SEC)
    cb.ax.tick_params(labelsize=6.5, colors=SEC)
    for out in (FIGDIR, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_regimes.{ext}"),
                        bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote F_regimes.pdf/.png -> {FIGDIR}, {OUT3}")


def main():
    args = sys.argv[1:]
    if "--probe" in args:
        t0 = time.perf_counter()
        out = run_sweep(probe=True)
        n_cells = len(out["meta"]["KS"]) * len(out["meta"]["PS"]) * len(FAMILIES)
        dt = time.perf_counter() - t0
        print(f"probe: {n_cells} cells in {dt:.1f}s -> full sweep "
              f"({len(KS) * len(PS) * len(FAMILIES)} cells) approx "
              f"{dt / n_cells * len(KS) * len(PS) * len(FAMILIES) / 60:.1f} min "
              "(q scales the per-cell cost, so this is a lower bound)")
        return
    if "--replot" in args or (os.path.exists(CACHE)
                             and "--recompute" not in args):
        data = json.load(open(CACHE))
        print(f"loaded cache {CACHE}")
    else:
        data = run_sweep()
    render(data)

    repro = check_reproduces_fig10(data)
    if repro is not None:
        print(f"\nFig 10 reproduction (panel a vs published plane): "
              f"max |dZ| = {repro:.2e}  "
              f"{'OK' if repro < 1e-12 else 'MISMATCH'}")
    print(f"\nselector on the (p,q) plane, gamma(q)=min(1,{GAMMA0}/sqrt(q)):")
    print(f"{'family':12s} {'NSR share':>10s} {'AC/L1':>7s} "
          f"{'agree L1':>9s} {'agree AC':>9s} {'agree AC+m':>11s} "
          f"{'max forfeit':>12s}")
    for r in selector_table(data):
        print(f"{r['family']:12s} {r['nsr_share']*100:9.1f}% "
              f"{r['ac_over_l1']:7.3f} {r['agree_L1']*100:8.1f}% "
              f"{r['agree_AC']*100:8.1f}% {r['agree_AC+margin']*100:10.1f}% "
              f"{r['maxforfeit_AC+margin']:11.2f}x")
    print("\ntie rule (certificate exactly 0 after the margin):")
    for r in selector_table(data):
        print(f"  {r['family']:12s} ties={r['tie_cells']:3d}  "
              f"ties->PSR {r['agree_ties_to_psr']*100:.1f}%  "
              f"ties->NSR {r['agree_ties_to_nsr']*100:.1f}%")


if __name__ == "__main__":
    main()
