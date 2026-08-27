"""
selector_check.py — selector check on the Fig. 9 (F_select balanced) plane.

Post-processing only: regenerates the exact per-cell tangents of the balanced
F-select sweep (deterministic seeds, build_F_select.draw_params verbatim) and
re-evaluates the *located* cost model (build_F_select.cell_costs, reused
verbatim / line-identical) with one substitution: the bandwidth certificate.

Certificates compared, per differentiated coefficient theta_l:
  Omega_L1  = 2 * Sum_j |v_j|                     (Assumption-4.4 term-wise L1)
  Omega_AC  = 2 * Sum_groups ||v_g||_2            (anticommutation-grouped)
  Omega_true= exact spectral diameter of A_l      (reference oracle ONLY)

Outputs into differential_computing/selector_check/:
  selector_check_percell.json / .csv   (per-(P,k,seed,l) table + per-cell aggregates)
  fig_selector_overlay.pdf/.png        (boundary overlay on the Fig. 9 plane)
  fig_selector_gamma.pdf/.png          (gamma offset diagnostic)
  selector_check_results.md            (verdict + stats; written by report step)

Run:  conda run -n qec_pg python -u differential_computing/tests/selector_check.py
Unit tests only:  ... selector_check.py --test
Replot from cache: ... selector_check.py --replot   (never recomputes)
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import build_F_select as bs

OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                      "selector_check"))
FIGDIR = bs.FIGDIR
BAL_CACHE = os.path.join(FIGDIR, "F_select_balanced_data.json")
PERCELL = os.path.join(OUTDIR, "selector_check_percell.json")

# balanced (Fig. 9) plane
PS = list(range(1, 11))
KS = list(range(1, 36))
SEEDS = 6
N_RANDOM_RESTARTS = 20
RNG_RESTART = 12345

# ── Pauli strings as (x, z) bitmasks; symplectic anticommutation ─────────────


def label_to_xz(label):
    """Alphabet label from build_F_select -> (x_bits, z_bits)."""
    if label[0] == "X":
        return (1 << label[1], 0)
    if label[0] == "Z":
        return (0, 1 << label[1])
    if label[0] == "ZZ":
        return (0, (1 << label[1]) | (1 << label[2]))
    raise ValueError(label)


def anticommute(a, b):
    """Symplectic form on Pauli bit-vectors: 1 iff the strings anticommute."""
    return (bin(a[0] & b[1]).count("1") + bin(a[1] & b[0]).count("1")) % 2 == 1


# ── anticommutation-grouped certificate ──────────────────────────────────────


def _greedy_cover(order, adj, weights):
    """Grow cliques by repeatedly adding the heaviest compatible term."""
    unassigned = set(order)
    groups = []
    for j in order:
        if j not in unassigned:
            continue
        unassigned.discard(j)
        clique = [j]
        compat = adj[j] & unassigned
        while compat:
            h = max(compat, key=lambda t: (weights[t], -t))
            clique.append(h)
            unassigned.discard(h)
            compat = compat & adj[h]
            compat.discard(h)
        groups.append(clique)
    return groups


def omega_ac(xz_list, weights, n_restarts=N_RANDOM_RESTARTS, rng=None,
             return_detail=False):
    """Omega_AC = 2 * Sum_g ||v_g||_2 over a pairwise-anticommuting clique
    cover; weight-aware greedy over several orderings, keep the best cover."""
    m = len(xz_list)
    w = np.abs(np.asarray(weights, dtype=float))
    keep = [j for j in range(m) if w[j] > 0 and (xz_list[j][0] | xz_list[j][1])]
    if not keep:
        return (0.0, dict(n_groups=0, max_group_size=0, restart_gain=1.0,
                          groups=[])) if return_detail else 0.0
    adj = {j: set() for j in keep}
    for a, b in itertools.combinations(keep, 2):
        if anticommute(xz_list[a], xz_list[b]):
            adj[a].add(b)
            adj[b].add(a)

    def score(groups):
        return 2.0 * sum(np.sqrt(sum(w[j] ** 2 for j in g)) for g in groups)

    deg_desc = sorted(keep, key=lambda j: (-len(adj[j]), -w[j], j))
    w_desc = sorted(keep, key=lambda j: (-w[j], -len(adj[j]), j))
    orders = [deg_desc, w_desc]
    if rng is None:
        rng = np.random.default_rng(RNG_RESTART)
    for _ in range(n_restarts):
        orders.append(list(rng.permutation(keep)))
    covers = [_greedy_cover(o, adj, w) for o in orders]
    scores = [score(c) for c in covers]
    baseline = min(scores[0], scores[1])          # deterministic orderings
    best_i = int(np.argmin(scores))
    best, best_score = covers[best_i], scores[best_i]
    if return_detail:
        return best_score, dict(
            n_groups=len(best),
            max_group_size=max(len(g) for g in best),
            restart_gain=float(baseline / best_score) if best_score > 0 else 1.0,
            groups=[[int(j) for j in g] for g in best])
    return best_score


# ── unit tests (guide section 1) ─────────────────────────────────────────────

_P1 = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex),
       "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0]).astype(complex)}


def _pauli_string(s):
    """'XZI' -> ((x,z) bitmask, dense matrix)."""
    x = z = 0
    M = np.array([[1.0 + 0j]])
    for i, ch in enumerate(s):
        if ch in "XY":
            x |= 1 << i
        if ch in "ZY":
            z |= 1 << i
        M = np.kron(M, _P1[ch])
    return (x, z), M


def _true_diam(terms, weights):
    A = sum(wj * M for (_, M), wj in zip(terms, weights))
    e = np.linalg.eigvalsh(A)
    return float(e[-1] - e[0])


def run_unit_tests():
    ok = True

    def check(name, cond, msg=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")
        ok = ok and cond

    # 1. single term
    t = [_pauli_string("XII")]
    ac = omega_ac([t[0][0]], [1.0])
    l1 = 2.0
    tr = _true_diam(t, [1.0])
    check("single term", np.allclose([ac, l1, tr], 2.0),
          f"AC={ac:.6f} L1={l1:.6f} true={tr:.6f} (expect 2)")

    # 2. q pairwise-anticommuting terms, equal v  (chain X1, Z1X2, Z1Z2X3, ...)
    for q in (2, 3, 5):
        strs = ["Z" * i + "X" + "I" * (q - 1 - i) for i in range(q)]
        terms = [_pauli_string(s) for s in strs]
        ac = omega_ac([t[0] for t in terms], [1.0] * q)
        tr = _true_diam(terms, [1.0] * q)
        check(f"{q} pairwise-AC terms", np.isclose(ac, 2 * np.sqrt(q))
              and np.isclose(tr, 2 * np.sqrt(q)),
              f"AC={ac:.4f} true={tr:.4f} (expect 2*sqrt({q})={2*np.sqrt(q):.4f}; "
              f"AC/L1={ac/(2*q):.4f}=q^-1/2)")

    # 3. all-commuting block (TFIM zz) -> singletons, AC == L1
    strs = ["ZZII", "IZZI", "IIZZ"]
    terms = [_pauli_string(s) for s in strs]
    ac, det = omega_ac([t[0] for t in terms], [1.0] * 3, return_detail=True)
    check("commuting TFIM zz", np.isclose(ac, 6.0) and det["max_group_size"] == 1,
          f"AC={ac:.4f} (expect 6, singletons)")

    # 4. Heisenberg bond XX+YY+ZZ: commuting pairs -> AC=L1=6, true diam 4
    terms = [_pauli_string(s) for s in ("XX", "YY", "ZZ")]
    ac = omega_ac([t[0] for t in terms], [1.0] * 3)
    tr = _true_diam(terms, [1.0] * 3)
    check("Heisenberg bond", np.isclose(ac, 6.0) and np.isclose(tr, 4.0),
          f"AC={ac:.4f} true={tr:.4f} (grouping NOT tight here: 6 vs 4)")

    # 5. frustrated triangle Z1Z2+Z2Z3+Z3Z1: commuting, true spectrum {3,-1}
    terms = [_pauli_string(s) for s in ("ZZI", "IZZ", "ZIZ")]
    ac = omega_ac([t[0] for t in terms], [1.0] * 3)
    tr = _true_diam(terms, [1.0] * 3)
    check("frustrated triangle", np.isclose(ac, 6.0) and np.isclose(tr, 4.0),
          f"AC={ac:.4f} true={tr:.4f} (grouping NOT tight: 6 vs 4)")

    print("unit tests:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ── the sweep: regenerate tangents, reuse the located cost model ─────────────


def rebuild_operating_point():
    """Verbatim re-execution of build_F_select.run_sweep's operating point."""
    import qutip as qp
    rng0 = np.random.default_rng(0)
    ops, labels = bs.build_alphabet()
    c = rng0.normal(size=len(ops)) * 0.4
    B = sum(ci * op for ci, op in zip(c, ops))
    U_half = (-1j * B * (bs.T / 2)).expm()
    O = sum(ops[bs.NQ + i] for i in range(bs.NQ))
    states = []
    for _ in range(4):
        v = rng0.normal(size=2 ** bs.NQ) + 1j * rng0.normal(size=2 ** bs.NQ)
        v /= np.linalg.norm(v)
        states.append(qp.Qobj(v.reshape(-1, 1), dims=[[2] * bs.NQ, [1] * bs.NQ]))
    sig = bs.branch_sigma(ops, U_half, O, states)
    return ops, labels, sig


def compute():
    os.makedirs(OUTDIR, exist_ok=True)
    ops, labels, sig = rebuild_operating_point()
    xz = [label_to_xz(l) for l in labels]
    dense = [np.asarray(op.full()) for op in ops]
    rng_restart = np.random.default_rng(RNG_RESTART)

    rows = []          # per (P, k, seed, l)
    cells = []         # per (P, k) aggregates
    spot = []          # spot-check vs bs.cell_costs verbatim
    for a, k in enumerate(KS):
        for b, P in enumerate(PS):
            per_seed = dict(meas=[], pl1=[], pac=[], nsr=[], psr=[],
                            nsr_l1=[], nsr_ac=[], psr_pred=[])
            for s in range(SEEDS):
                rng = np.random.default_rng(97 * s + 3 * P + 11 * k)
                params = bs.draw_params("general", k, P, labels, rng)

                # per-parameter certificates + oracle
                N_true = N_l1 = N_ac = 0.0
                for li, d in enumerate(params):
                    idx = list(d.keys())
                    w = [abs(d[j]) for j in idx]
                    o_l1 = 2.0 * sum(w)
                    o_ac, det = omega_ac([xz[j] for j in idx], w,
                                         rng=rng_restart, return_detail=True)
                    A = sum(d[j] * dense[j] for j in idx)
                    e = np.linalg.eigvalsh(A)
                    o_tr = float(e[-1] - e[0])
                    if not (o_l1 + 1e-9 >= o_ac >= o_tr - 1e-9):
                        raise RuntimeError(
                            f"certificate chain violated at P={P} k={k} s={s} "
                            f"l={li}: L1={o_l1} AC={o_ac} true={o_tr}")
                    if det["max_group_size"] > 2 * bs.NQ + 1:
                        raise RuntimeError("group larger than 2n+1")
                    N_true += o_tr ** 2
                    N_l1 += o_l1 ** 2
                    N_ac += o_ac ** 2
                    rows.append(dict(
                        P=P, k=k, seed=s, l=li,
                        Omega_L1=o_l1, Omega_AC=o_ac, Omega_true=o_tr,
                        chi_AC=o_tr / o_ac,
                        n_groups=det["n_groups"],
                        max_group_size=det["max_group_size"],
                        restart_gain=det["restart_gain"]))

                # PSR side — line-identical to bs.cell_costs
                G = len(ops)
                nj = np.zeros(G)
                nj_pred = np.zeros(G)
                for d in params:
                    Sl = sum(abs(c) * sig[j] for j, c in d.items())
                    Sl_pred = sum(abs(c) * bs.SIGMA_BOUND for c in d.values())
                    for j, c in d.items():
                        nj[j] = max(nj[j], Sl * abs(c) * sig[j])
                        nj_pred[j] = max(nj_pred[j],
                                         Sl_pred * abs(c) * bs.SIGMA_BOUND)
                N_psr = 2.0 * nj.sum()
                N_psr_pred = 2.0 * nj_pred.sum()

                per_seed["meas"].append(np.log10(N_true / N_psr))
                per_seed["pl1"].append(np.log10(N_l1 / N_psr_pred))
                per_seed["pac"].append(np.log10(N_ac / N_psr_pred))
                per_seed["nsr"].append(N_true)
                per_seed["psr"].append(N_psr)
                per_seed["nsr_l1"].append(N_l1)
                per_seed["nsr_ac"].append(N_ac)
                per_seed["psr_pred"].append(N_psr_pred)

                # verbatim spot-check on a deterministic subsample
                if (a * len(PS) + b + s) % 97 == 0:
                    rng2 = np.random.default_rng(97 * s + 3 * P + 11 * k)
                    params2 = bs.draw_params("general", k, P, labels, rng2)
                    Nn, Npr, Nn_p, Npr_p = bs.cell_costs(params2, ops, sig)
                    spot.append(dict(P=P, k=k, s=s,
                                     d_nsr=abs(Nn - N_true) / Nn,
                                     d_psr=abs(Npr - N_psr) / Npr,
                                     d_psr_pred=abs(Npr_p - N_psr_pred) / Npr_p))
            cells.append(dict(
                P=P, k=k,
                Z_meas=float(np.mean(per_seed["meas"])),
                Zpred_L1=float(np.mean(per_seed["pl1"])),
                Zpred_AC=float(np.mean(per_seed["pac"])),
                N_NSR_meas=per_seed["nsr"], N_PSR_meas=per_seed["psr"],
                N_NSR_L1=per_seed["nsr_l1"], N_NSR_AC=per_seed["nsr_ac"],
                N_PSR_pred=per_seed["psr_pred"]))
        print(f"k={k} done", flush=True)

    out = dict(meta=dict(PS=PS, KS=KS, seeds=SEEDS, NQ=bs.NQ, T=bs.T,
                         sigma_bound=float(bs.SIGMA_BOUND),
                         sigma_mean=float(np.mean(sig)),
                         n_restarts=N_RANDOM_RESTARTS),
               cells=cells, rows=rows, spot_check=spot)
    json.dump(out, open(PERCELL, "w"))
    # CSV of the per-(P,k,seed,l) table
    import csv
    with open(PERCELL.replace(".json", ".csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"wrote {PERCELL} (+.csv), {len(rows)} per-parameter rows")
    return out


# ── analysis, checks, figures ────────────────────────────────────────────────


def selector_stats(Zm, pred_log, name, tol=1e-9):
    """Agreement + forfeit stats for 'choose NSR iff pred_log < -tol'."""
    choice_nsr = pred_log < -tol
    meas_nsr = Zm < 0
    mism = choice_nsr != meas_nsr
    forf = 10.0 ** np.abs(Zm[mism]) if mism.any() else np.array([1.0])
    return dict(name=name,
                agreement=float(1.0 - mism.mean()),
                nsr_frac_chosen=float(choice_nsr.mean()),
                median_forfeit_mism=float(np.median(forf)),
                max_forfeit=float(forf.max()) if mism.any() else 1.0,
                mism_frac=float(mism.mean()))


def analyse(out):
    meta = out["meta"]
    nP, nK = len(meta["PS"]), len(meta["KS"])
    Zm = np.zeros((nK, nP))
    Zl1 = np.zeros((nK, nP))
    Zac = np.zeros((nK, nP))
    for c in out["cells"]:
        a, b = meta["KS"].index(c["k"]), meta["PS"].index(c["P"])
        Zm[a, b] = c["Z_meas"]
        Zl1[a, b] = c["Zpred_L1"]
        Zac[a, b] = c["Zpred_AC"]

    res = dict()

    # 5.1 reproduce the published plane
    bal = json.load(open(BAL_CACHE))
    Zpub = np.array(bal["general"]["Z"])
    res["repro_max_absdiff_Z"] = float(np.abs(Zm - Zpub).max())
    res["repro_nsr_share"] = float((Zm < 0).mean())
    res["published_nsr_share"] = float((Zpub < 0).mean())

    # published stats reproduction (builder's own convention)
    Zp_pub = np.array(bal["general"]["Zpred"])
    choice_nsr_b = Zp_pub < 0
    mism_b = choice_nsr_b != (Zpub < 0)
    forf_b = np.where(mism_b, 10.0 ** np.abs(Zpub), 1.0)
    res["published_builder_agreement"] = float(1 - mism_b.mean())
    res["published_builder_median_forfeit_mism"] = float(np.median(forf_b[mism_b]))
    res["published_builder_max_forfeit"] = float(forf_b.max())

    # 5.2: resolve 42 vs 64 — exact-tie cells resolved by fp sign noise
    tie = np.abs(Zl1) < 1e-12          # exact predicted tie (recomputed, exact)
    res["tie_cells_frac"] = float(tie.mean())
    res["tie_and_measured_nsr_frac"] = float((tie & (Zm < 0)).mean())
    res["tie_fp_sign_neg_frac_published"] = float((np.abs(Zp_pub) < 1e-10)
                                                  .mean())
    res["agreement_strict_psr"] = float((Zm > 0).mean())
    res["fp_lucky_agree_frac"] = float(((np.abs(Zp_pub) < 1e-10)
                                        & (Zp_pub < 0) & (Zpub < 0)).mean())

    # 5.3 constant-selector confirmation for L1
    res["L1_pred_min"] = float(10 ** Zl1.min())
    res["L1_pred_max"] = float(10 ** Zl1.max())
    res["L1_strict_nsr_cells"] = int((Zl1 < -1e-9).sum())

    # 5.4 chain: verified per-row during compute(); summarize chi
    rows = out["rows"]
    chi = np.array([r["chi_AC"] for r in rows])
    lac_l1 = np.array([r["Omega_AC"] / r["Omega_L1"] for r in rows])
    res["chi_AC_median"] = float(np.median(chi))
    res["chi_AC_min"] = float(chi.min())
    res["AC_over_L1_median"] = float(np.median(lac_l1))
    res["AC_over_L1_min"] = float(lac_l1.min())
    res["max_group_size"] = int(max(r["max_group_size"] for r in rows))
    res["restart_gain_max"] = float(max(r["restart_gain"] for r in rows))
    res["restart_gain_frac_gt1"] = float(np.mean(
        [r["restart_gain"] > 1 + 1e-12 for r in rows]))

    # gamma offset diagnostic
    G = 10.0 ** (Zm - Zac)
    res["gamma_median"] = float(np.median(G))
    res["gamma_iqr"] = [float(np.percentile(G, 25)), float(np.percentile(G, 75))]
    res["gamma_min"] = float(G.min())
    res["gamma_max"] = float(G.max())
    res["gamma_maxmin"] = float(G.max() / G.min())
    a_star, b_star = meta["KS"].index(1), meta["PS"].index(2)
    res["gamma_star_P2_k1"] = float(G[a_star, b_star])
    res["star_measured_ratio"] = float(10 ** Zm[a_star, b_star])
    res["star_pred_AC_ratio"] = float(10 ** Zac[a_star, b_star])
    # trend: median gamma per column (P) and per row (k)
    res["gamma_median_vs_P"] = [float(np.median(G[:, b])) for b in range(nP)]
    res["gamma_median_vs_k"] = [float(np.median(G[a, :])) for a in range(nK)]

    # gamma_med(k) power-law fit (k is static program text -> still compile-time)
    kk = np.array(meta["KS"], dtype=float)
    gk = np.array(res["gamma_median_vs_k"])
    alpha, logc = np.polyfit(np.log(kk), np.log(gk), 1)
    cfit = float(np.exp(logc))
    fitv = alpha * np.log(kk) + logc
    r2 = 1 - np.sum((np.log(gk) - fitv) ** 2) / np.sum(
        (np.log(gk) - np.log(gk).mean()) ** 2)
    res["gamma_powerlaw"] = dict(c=cfit, alpha=float(alpha), R2=float(r2))
    marg_k = np.log10(cfit * kk ** alpha)[:, None]

    # selectors
    gmed = res["gamma_median"]
    stats = [selector_stats(Zm, Zl1, "L1 certificate"),
             selector_stats(Zm, Zac, "AC certificate"),
             selector_stats(Zm, Zac + np.log10(gmed),
                            f"AC + flat margin (gamma_med={gmed:.3f})"),
             selector_stats(Zm, Zac + marg_k,
                            f"AC + power-law margin ({cfit:.2f}*k^{alpha:.2f})")]
    # theory-pinned exponent (diameter concentration ~ sqrt(q)): one constant,
    # calibrated on the small-k rows only (held-out for k >= 9), margin <= 1
    small = kk <= 8
    c_pin = float(np.exp(np.mean(np.log(gk[small]) + 0.5 * np.log(kk[small]))))
    marg_pin = np.log10(np.minimum(1.0, c_pin / np.sqrt(kk)))[:, None]
    stats.append(selector_stats(
        Zm, Zac + marg_pin,
        f"AC + pinned margin min(1, {c_pin:.2f}/sqrt(k)), c from k<=8"))
    res["gamma_pinned_c"] = c_pin
    res["selectors"] = stats
    res["_marg_pin"] = marg_pin[:, 0].tolist()
    res["_marg_k"] = marg_k[:, 0].tolist()

    res["spot_check_max_reldiff"] = float(max(
        max(s["d_nsr"], s["d_psr"], s["d_psr_pred"])
        for s in out["spot_check"]))
    return res, Zm, Zl1, Zac, G


def render(out, res, Zm, Zl1, Zac, G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter

    meta = out["meta"]
    Pg, Kg = np.meshgrid(meta["PS"], meta["KS"])
    INK, SEC, GRID, SURFACE = bs.INK, bs.SEC, bs.GRID, bs.SURFACE
    C_PSR, C_NSR = bs.C_PSR, bs.C_NSR
    gmed = res["gamma_median"]

    Zs = gaussian_filter(Zm, sigma=0.8)
    Zacs = gaussian_filter(Zac, sigma=0.8)
    Zcal = gaussian_filter(Zac + np.log10(gmed), sigma=0.8)
    Zpl = gaussian_filter(Zac + np.array(res["_marg_pin"])[:, None], sigma=0.8)

    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=7, colors=SEC)
    for s in ax.spines.values():
        s.set_color(GRID)

    ax.contourf(Pg, Kg, Zs, levels=[-99, 0, 99], colors=[C_NSR, C_PSR],
                alpha=0.30, antialiased=True)
    ax.contour(Pg, Kg, Zs, levels=[0.0], colors="k", linewidths=1.6, zorder=4)
    ax.contour(Pg, Kg, Zacs, levels=[0.0], colors="#7a3fb8", linewidths=1.3,
               linestyles="dashed", zorder=4)
    ax.contour(Pg, Kg, Zcal, levels=[0.0], colors="#c23a80", linewidths=1.3,
               linestyles="dotted", zorder=4)
    ax.contour(Pg, Kg, Zpl, levels=[0.0], colors="#b8860b", linewidths=1.4,
               linestyles=[(0, (4, 1.5, 1, 1.5))], zorder=4)

    halo = [pe.withStroke(linewidth=2.0, foreground="#ffffff")]
    ax.text(0.10, 0.90, "NSR\nwins", transform=ax.transAxes, fontsize=8.5,
            weight="bold", color="#00654a", ha="center", va="center",
            path_effects=halo, zorder=5)
    ax.text(0.85, 0.12, "PSR\nwins", transform=ax.transAxes, fontsize=8.5,
            weight="bold", color="#00517e", ha="center", va="center",
            path_effects=halo, zorder=5)
    ax.plot([2], [1], marker="*", ms=11, color="#eb6834",
            markeredgecolor=INK, markeredgewidth=0.6, zorder=6, clip_on=False)

    handles = [
        Line2D([], [], color="k", lw=1.6, label="measured crossing"),
        Line2D([], [], color="#7a3fb8", lw=1.3, ls="--",
               label=r"$\bar\Omega_{AC}$ predicted"),
        Line2D([], [], color="#c23a80", lw=1.3, ls=":",
               label=rf"$\bar\Omega_{{AC}}$ + flat margin ($\gamma_{{med}}$={gmed:.2f})"),
        Line2D([], [], color="#b8860b", lw=1.4, ls="-.",
               label=(rf"$\bar\Omega_{{AC}}$ + $\gamma(k)$=min(1, "
                      rf"{res['gamma_pinned_c']:.2f}$/\sqrt{{k}}$)")),
        Line2D([], [], color="none",
               label=r"$\bar\Omega_{L1}$: no crossing (PSR/tie everywhere)")]
    ax.legend(handles=handles, loc="upper right", fontsize=5.6,
              framealpha=0.85)
    ax.set_xticks(range(1, 11))
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel("# differentiated coefficients  P", fontsize=7.5, color=SEC)
    ax.set_ylabel("# alphabet terms per coefficient  k", fontsize=7.5,
                  color=SEC)
    ax.set_title("Selector boundaries vs measured crossing", fontsize=8,
                 color=INK, pad=5)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig_selector_overlay.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # gamma diagnostic
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    for ax in axs:
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=6.5, colors=SEC)
        for s in ax.spines.values():
            s.set_color(GRID)
    axs[0].hist(np.log10(G.ravel()), bins=30, color="#7a3fb8", alpha=0.8)
    axs[0].axvline(np.log10(res["gamma_star_P2_k1"]), color="#eb6834", lw=1.4)
    axs[0].axvline(np.log10(gmed), color="k", lw=1.0, ls="--")
    axs[0].set_xlabel(r"$\log_{10}\gamma$", fontsize=7)
    axs[0].set_title(rf"$\gamma$ distribution (star={res['gamma_star_P2_k1']:.2f},"
                     rf" med={gmed:.2f})", fontsize=7)
    for ax, xs, lab in ((axs[1], Pg, "P"), (axs[2], Kg, "k")):
        ax.scatter(xs.ravel(), G.ravel(), s=3, alpha=0.25, color="#52514e")
        vals = sorted(set(xs.ravel()))
        med = [np.median(G.ravel()[xs.ravel() == v]) for v in vals]
        ax.plot(vals, med, color="#c23a80", lw=1.4)
        ax.axhline(1.0, color="k", lw=0.7, ls=":")
        ax.set_xlabel(lab, fontsize=7)
        ax.set_yscale("log")
        ax.set_title(rf"$\gamma$ vs {lab} (median line)", fontsize=7)
    axs[1].scatter([2], [res["gamma_star_P2_k1"]], marker="*", s=60,
                   color="#eb6834", zorder=5)
    kk = np.array(meta["KS"], dtype=float)
    pl = res["gamma_powerlaw"]
    axs[2].plot(kk, pl["c"] * kk ** pl["alpha"], color="#b8860b", lw=1.2,
                ls="--", label=rf"{pl['c']:.2f}$k^{{{pl['alpha']:.2f}}}$")
    axs[2].legend(fontsize=6, loc="lower left", framealpha=0.8)
    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig_selector_gamma.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figures written to", OUTDIR)


def main():
    if "--test" in sys.argv:
        ok = run_unit_tests()
        sys.exit(0 if ok else 1)
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(PERCELL) and "--recompute" not in sys.argv:
        out = json.load(open(PERCELL))
        print("loaded cached per-cell data")
    else:
        if not run_unit_tests():
            sys.exit("unit tests failed — not running on real data")
        out = compute()
    res, Zm, Zl1, Zac, G = analyse(out)
    json.dump(res, open(os.path.join(OUTDIR, "selector_check_summary.json"),
                        "w"), indent=1)
    render(out, res, Zm, Zl1, Zac, G)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
