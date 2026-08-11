"""
tab3_verification.py — regenerate the paper's Tab 3 (compilation verification
round-trip).

For each case we compile the differentiated program to a neutral-atom tweezer
schedule + PulseLedger, reconstruct the Hamiltonian of every segment from the
ledger meta-parameters, and compare:
  (a) per-segment  ||H_compiled − H_target||  (the honest solver-decomposition
      residual, identity removed), and
  (b) the end-to-end gradient: ledger-reconstructed vs direct QuTiP.

Cases: 1q / 2q / 3q single-evolution round-trips, and a multi-layer program
(PSR differentiates a parameterized dressing layer wrapped by two FIXED layers —
the reconstructed-gradient check plus the Layer-1 compilation round-trip).

Caches figures/tab3_verification.json and prints a Markdown table.
Run:  conda run -n qec_pg python differential_computing/tests/tab3_verification.py
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
from simuq.braket.diffQC_provider import diffQCProvider
from observable_program_generator import observable_program_generator
from qutip_sequential import QuTiPSequentialRunner
from combine_gradient import combine_gradient_results

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
T, X_VAL, TOL = 0.5, 1.0, 0.1


def _compile_provider(H_param, x_val, T, n_qubits):
    prov = diffQCProvider()
    qs = QSystem(); q = [Qubit(qs) for _ in range(n_qubits)]
    qs.add_evolution(H_param.set_parameterizedHam({"x": x_val}), T)
    prov.compile(qs, "quera", "Aquila", "rydberg2d", tol=TOL, verbose=0)
    return prov


def _max_seg_norms(norm_diffs):
    by_seg = {}
    for nd in norm_diffs:
        s = nd["seg"]
        by_seg[s] = max(by_seg.get(s, 0.0), nd["norm_diff"])
    return {int(k): float(v) for k, v in sorted(by_seg.items())}


def roundtrip(label, H_param, n_qubits, obs, x_val=X_VAL, T=T):
    np.random.seed(42)
    programs = observable_program_generator(
        H_param, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    prov = _compile_provider(H_param, x_val, T, n_qubits)
    prov.run(programs, None, T=T, backend="hardware", verbose=0)
    runner = QuTiPSequentialRunner(n_qubits=n_qubits)
    res = prov.verify(programs, obs, T=T, psi0=runner.zero_state(), verbose=0)
    rel = res["error"] / (abs(res["ground_truth"]) + 1e-12)
    seg = _max_seg_norms(res["norm_diffs"])
    print(f"  {label:12s}: grad_err={res['error']:.2e} ({rel:.2%})  "
          f"seg-norms={ {k: round(v, 4) for k, v in seg.items()} }")
    return dict(label=label, n_qubits=n_qubits,
                ground_truth=res["ground_truth"], reconstructed=res["reconstructed"],
                abs_error=res["error"], rel_error=rel, seg_norms=seg)


def multilayer():
    x = sp.Symbol("x"); x_val = 1.0
    J01 = sp.sin(2 * x); J02 = 1.2
    qs1 = QSystem(); q1 = [Qubit(qs1) for _ in range(3)]
    H1 = J01 * q1[0].Z * q1[1].Z + J01 * q1[0].X + J01 * q1[1].X
    qsx = QSystem(); qx = [Qubit(qsx) for _ in range(3)]
    HX = 5.0 * (qx[0].X + qx[1].X + qx[2].X)
    qs2 = QSystem(); q2 = [Qubit(qs2) for _ in range(3)]
    H2 = J02 * q2[0].Z * q2[2].Z + J02 * q2[0].X + J02 * q2[2].X
    T1, TX, T2 = 0.5, 0.1, 0.5

    np.random.seed(42)
    programs = observable_program_generator(
        H1, T1, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    runner = QuTiPSequentialRunner(n_qubits=3)
    psi0 = runner.zero_state(); obs = runner.zz_observable(0, 1)

    def expfn(H_list):                      # PSR branch (L1), then fixed L2, L3
        st = runner.run_sequence(H_list, psi0)
        st = qp.sesolve(HX.to_qutip_qobj(), st, [0, float(TX)]).states[-1]
        st = qp.sesolve(H2.to_qutip_qobj(), st, [0, float(T2)]).states[-1]
        return float(qp.expect(obs, st).real)

    grad_psr = combine_gradient_results(programs, expfn, T=T1)

    def f_full(xv):
        st = psi0
        st = qp.sesolve(H1.set_parameterizedHam({"x": xv}).to_qutip_qobj(), st, [0, float(T1)]).states[-1]
        st = qp.sesolve(HX.to_qutip_qobj(), st, [0, float(TX)]).states[-1]
        st = qp.sesolve(H2.to_qutip_qobj(), st, [0, float(T2)]).states[-1]
        return float(qp.expect(obs, st).real)

    eps = 1e-4
    grad_fd = (f_full(x_val + eps) - f_full(x_val - eps)) / (2 * eps)
    sem_rel = abs(grad_psr - grad_fd) / (abs(grad_fd) + 1e-12)

    # Layer-1 compilation round-trip
    prov1 = _compile_provider(H1, x_val, T1, 3)
    prov1.run(programs, None, T=T1, backend="hardware", verbose=0)
    res = prov1.verify(programs, obs, T=T1, psi0=psi0, verbose=0)
    seg = _max_seg_norms(res["norm_diffs"])
    print(f"  {'multi-layer':12s}: semantics(PSR vs FD)={sem_rel:.2%}  "
          f"L1-compile_err={res['error']:.2e} ({res['error']/(abs(res['ground_truth'])+1e-12):.2%})  "
          f"seg-norms={ {k: round(v, 4) for k, v in seg.items()} }")
    return dict(label="multi-layer", grad_psr=float(grad_psr), grad_fd=float(grad_fd),
                semantics_rel_error=sem_rel, L1_abs_error=res["error"],
                L1_rel_error=res["error"] / (abs(res["ground_truth"]) + 1e-12), seg_norms=seg)


def md_table(rows, ml):
    def segstr(seg):
        return ", ".join(f"seg{k} {v:.3f}" for k, v in seg.items())
    lines = ["| Case | Reconstructed-gradient error | Max ‖H_comp−H_tgt‖ per segment |",
             "|------|------------------------------|--------------------------------|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['abs_error']:.1e} ({r['rel_error']:.2%}) | {segstr(r['seg_norms'])} |")
    lines.append(f"| multi-layer (L1 compile) | {ml['L1_abs_error']:.1e} ({ml['L1_rel_error']:.2%}) | {segstr(ml['seg_norms'])} |")
    lines.append(f"| multi-layer (PSR semantics, through 2 fixed layers) | {ml['semantics_rel_error']:.2%} | — |")
    return "\n".join(lines)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    x = sp.Symbol("x")

    def build_1q():
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        return x * q[0].Z + q[0].X

    def build_2q():
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]; J = sp.sin(2 * x)
        return J * q[0].Z * q[1].Z + J * q[0].X + J * q[1].X

    def build_3q():
        qs = QSystem(); q = [Qubit(qs) for _ in range(3)]; J = sp.sin(2 * x)
        return J * q[0].Z * q[1].Z + J * q[0].X + J * q[1].X

    runner2, runner3 = QuTiPSequentialRunner(2), QuTiPSequentialRunner(3)
    print("Tab 3 — compilation verification round-trip (tol=%.2f, T=%.1f, x=%.1f)" % (TOL, T, X_VAL))
    rows = [
        roundtrip("1q", build_1q(), 2, qp.tensor(qp.sigmaz(), qp.qeye(2))),
        roundtrip("2q", build_2q(), 2, runner2.zz_observable(0, 1)),
        roundtrip("3q", build_3q(), 3, runner3.zz_observable(0, 1)),
    ]
    ml = multilayer()

    out = {"meta": {"tol": TOL, "T": T, "x_val": X_VAL}, "cases": rows, "multilayer": ml}
    cache = os.path.join(FIGDIR, "tab3_verification.json")
    json.dump(out, open(cache, "w"), indent=2, default=float)
    print("\n" + md_table(rows, ml))
    print(f"\ncached: {cache}")


if __name__ == "__main__":
    main()
