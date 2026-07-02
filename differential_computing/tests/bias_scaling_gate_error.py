"""
bias_scaling_gate_error.py — how much does the 99.9% CZ kick cost PSR?

Re-runs the bias_scaling_relative sweep (TFIM n=2..7, T/T2*=0.15, relative bias,
∞ shots, deterministic τ) with the reference-fidelity kick gate error switched
on (gate_error_2q=1e-3 / gate_error_1q=1e-4, Z-type channel on the kicked pair
— the model the CZ-kick compilation now realizes literally in hardware), and
plots BOTH kick models on one graph:

  FD best-ε                  — unchanged (FD runs no kick; gate error is a
                               PSR-specific cost, shown honestly)
  PSR rescaled, ideal kick   — from bias_scaling_relative_data.json (cache)
  PSR rescaled, 99.9% kick   — this run; the rescale corrects dephasing
                               attenuation only, NOT gate error, so this
                               curve floors at the gate-error bias.

Run:  conda run -n qec_pg python differential_computing/tests/bias_scaling_gate_error.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bias_scaling_relative import run, obs_local, obs_extensive, NS, T, T2

NS_GE = [n for n in NS if n < 5]   # keep density-matrix sims small (user cap)

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
CACHE_IDEAL = os.path.join(FIGDIR, "bias_scaling_relative_data.json")
N_PLOT_MIN = 3

# CZ-kick fidelity variants.  99.95% = the cryo platform's best (headline);
# 99.9% = the literature reference (Evered-class).  Cache file per variant.
VARIANTS = [
    ("99.9%", 1e-3, os.path.join(FIGDIR, "bias_scaling_gate_error_data.json")),
    ("99.95%", 5e-4, os.path.join(FIGDIR, "bias_scaling_gate_error_9995_data.json")),
]


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    results = {}   # fid_label -> {"loc": [...], "ext": [...]}
    for fid_label, eps2, cache in VARIANTS:
        if os.path.exists(cache):
            d = json.load(open(cache)); loc, ext = d["loc"], d["ext"]
            print(f"[{fid_label}] loaded cache: local n={[r['n'] for r in loc]}, "
                  f"extensive n={[r['n'] for r in ext]}")
        else:
            loc, ext = [], []

        def save():
            json.dump({"loc": loc, "ext": ext}, open(cache, "w"), default=float)

        for n in NS_GE:
            if n not in [r["n"] for r in loc]:
                loc.append(run(n, obs_local, "local", gate_error=eps2)); save()
            if n not in [r["n"] for r in ext]:
                ext.append(run(n, obs_extensive, "extensive", gate_error=eps2))
                save()
        print(f"[{fid_label}] cached: {cache}")
        results[fid_label] = {"loc": loc, "ext": ext}

    ideal = json.load(open(CACHE_IDEAL))

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    for key, c, mk, name in [("loc", "#7b1fa2", "s", "local ⟨Z0Z1⟩"),
                             ("ext", "#e65100", "^", "extensive ⟨ΣZZ⟩")]:
        id_res = [r for r in ideal[key] if r["n"] >= N_PLOT_MIN]
        nn = [r["n"] for r in id_res]
        ax.semilogy(nn, [r["fd_rel"] for r in id_res], mk + "-", color=c,
                    lw=2.4, label=f"FD best-ε — {name} (no kick, unchanged)")
        ax.semilogy(nn, [r["res_rel"] for r in id_res], mk + "--", color=c,
                    lw=1.8, alpha=0.75, mfc="white",
                    label=f"PSR rescaled, ideal kick — {name}")
        # 99.95% (cryo, headline) bold dotted; 99.9% (reference) light dotted
        for fid_label, style in (("99.95%", dict(lw=2.0, alpha=1.0, mfc=c)),
                                 ("99.9%", dict(lw=1.2, alpha=0.45, mfc="none"))):
            ge = [r for r in results[fid_label][key] if r["n"] >= N_PLOT_MIN]
            ax.semilogy([r["n"] for r in ge], [r["res_rel"] for r in ge],
                        mk + ":", color=c,
                        label=f"PSR rescaled, {fid_label} CZ kick — {name}",
                        **style)
    ax.set_xlabel("chain length n")
    ax.set_ylabel("relative gradient bias  |estimate − ideal| / |ideal|   (∞ shots)")
    ax.set_title("Cost of a realistic kick (T/T2* = 0.15): CZ gate error adds a\n"
                 "PSR-specific bias the rescale does not correct — at the cryo "
                 "platform's\n99.95% fidelity PSR still sits far below FD's floor",
                 fontsize=10.5)
    ax.set_xticks([n for n in NS if n >= N_PLOT_MIN])
    ax.grid(True, which="both", axis="y", alpha=0.15)
    ax.legend(frameon=False, fontsize=7.2)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "bias_scaling_gate_error.png")
    fig.savefig(out)
    print(f"saved: {out}")

    # "how much worse" summary
    for fid_label, _, _ in VARIANTS:
        print(f"\n  [{fid_label} CZ kick]")
        for key in ("loc", "ext"):
            for r in results[fid_label][key]:
                match = [q for q in ideal[key] if q["n"] == r["n"]]
                if match:
                    print(f"  {key} n={r['n']}: rescaled "
                          f"{match[0]['res_rel']:.4f} → {r['res_rel']:.4f}  "
                          f"(FD {match[0]['fd_rel']:.4f})")


if __name__ == "__main__":
    main()
