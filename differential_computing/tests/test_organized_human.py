"""
End-to-end two-layer dressing pipeline.

|psi0> --[ H1(x): J01 dressing ]--[ H_X: global X ]--[ H2: J02 dressing ]-- <Z0Z1>
              parameterized              fixed               fixed

3 qubits. Layer 1 parameterized (J01=sin(2x)), Layers 2+3 fixed TI.
All layers compiled to hardware ops and emitted to DSL.

Run standalone:
    conda run -n qec_pg python differential_computing/tests/test_organized_human.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Clear cached modules
to_remove = [k for k in sys.modules if k.startswith('simuq')
             or k in ('tweezer_mapper', 'pulse_ledger', 'aod_channel',
                       'verify_compilation', 'observable_program_generator',
                       'qutip_sequential', 'combine_gradient')]
for k in to_remove:
    del sys.modules[k]

import numpy as np
import sympy as sp
import qutip as qp

from simuq import QSystem, Qubit
from simuq.braket.diffQC_provider import diffQCProvider, to_pulsedsl_simple
from observable_program_generator import observable_program_generator
from qutip_sequential import QuTiPSequentialRunner
from combine_gradient import combine_gradient_results
from tweezer_mapper import TweezerMapper
from verify_compilation import norm_check, _remove_identity


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Hamiltonian Definition
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP 1: Hamiltonian Definition")
print("=" * 60)

x_sym = sp.Symbol('x')
x_val = 1.0
J01 = sp.sin(2 * x_sym)   # parameterized coupling qubits 0-1
J02_fixed = 1.2            # fixed coupling qubits 0-2

# Layer 1: J01 dressing (parameterized by x)
qs1 = QSystem(); q1 = [Qubit(qs1) for _ in range(3)]
H1_param = J01 * q1[0].Z * q1[1].Z + J01 * q1[0].X + J01 * q1[1].X

# Layer 2: Global X rotation on all qubits (fixed)
qs_x = QSystem(); q_x = [Qubit(qs_x) for _ in range(3)]
H_X = 5.0 * (q_x[0].X + q_x[1].X + q_x[2].X)

# Layer 3: J02 dressing (fixed coupling 0-2)
qs2 = QSystem(); q2 = [Qubit(qs2) for _ in range(3)]
H2_fixed = J02_fixed * q2[0].Z * q2[2].Z + J02_fixed * q2[0].X + J02_fixed * q2[2].X

# Durations
T1  = 0.5    # layer 1 (parameterized dressing)
T_X = 0.1    # global X (short pulse)
T2  = 0.5    # layer 2 (fixed dressing)

print(f"Layer 1: H1(x) = sin(2x)·(Z₀Z₁ + X₀ + X₁)   T1 = {T1}")
print(f"Layer 2: H_X   = 5.0·(X₀ + X₁ + X₂)          T_X = {T_X}")
print(f"Layer 3: H2    = 1.2·(Z₀Z₂ + X₀ + X₂)        T2 = {T2}")
print(f"Total time: {T1 + T_X + T2}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PSR Program Generation (Layer 1 only)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 2: PSR Program Generation")
print("=" * 60)

np.random.seed(42)
programs = observable_program_generator(
    H1_param, T1, n_sample=1, n_repetition=1,
    diff_var="x", value=x_val,
)

print(f"PSR: {len(programs)} Hj terms, {len(programs[0][0])} H_lists each")
for ti, (H_tot_list, ugrad, n_rep) in enumerate(programs):
    Hj = H_tot_list[0][1][0]
    terms = [(t.to_tuple(), float(c)) for t, c in Hj.ham]
    print(f"  term {ti}: Hj = {terms}  ugrad = {ugrad:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: PSR Gradient Verification (QuTiP, no compilation)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3: PSR Gradient Verification (QuTiP)")
print("=" * 60)

runner = QuTiPSequentialRunner(n_qubits=3)
psi0 = runner.zero_state()
obs = runner.zz_observable(0, 1)

def expfn_multilayer(H_list):
    """Evolve through PSR branches (layer 1), then fixed layers 2 and 3."""
    state = runner.run_sequence(H_list, psi0)
    result_x = qp.sesolve(H_X.to_qutip_qobj(), state, [0, float(T_X)])
    state = result_x.states[-1]
    result_2 = qp.sesolve(H2_fixed.to_qutip_qobj(), state, [0, float(T2)])
    state = result_2.states[-1]
    return float(qp.expect(obs, state).real)

grad_psr = combine_gradient_results(programs, expfn_multilayer, T=T1)

def f_full(xv):
    H1_eval = H1_param.set_parameterizedHam({"x": xv})
    state = psi0
    r1 = qp.sesolve(H1_eval.to_qutip_qobj(), state, [0, float(T1)]); state = r1.states[-1]
    r2 = qp.sesolve(H_X.to_qutip_qobj(), state, [0, float(T_X)]); state = r2.states[-1]
    r3 = qp.sesolve(H2_fixed.to_qutip_qobj(), state, [0, float(T2)]); state = r3.states[-1]
    return float(qp.expect(obs, state).real)

eps = 1e-4
grad_fd = (f_full(x_val + eps) - f_full(x_val - eps)) / (2 * eps)

print(f"PSR gradient:  {grad_psr:.6f}")
print(f"FD  gradient:  {grad_fd:.6f}")
print(f"Relative error: {abs(grad_psr - grad_fd) / (abs(grad_fd) + 1e-12):.4%}")
print(f"Sign match: {np.sign(grad_psr) == np.sign(grad_fd)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Compile Layer 1 (parameterized)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4: Compile Layer 1 (parameterized dressing)")
print("=" * 60)

prov1 = diffQCProvider()
qs1_c = QSystem(); q1_c = [Qubit(qs1_c) for _ in range(3)]
H1_eval = H1_param.set_parameterizedHam({"x": x_val})
qs1_c.add_evolution(H1_eval, T1)
prov1.compile(qs1_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)

prov1.run(programs, None, T=T1, backend="hardware", verbose=1)

# Show solver instructions
boxes1 = prov1.prog[2]
print("\nLayer 1 solver instructions:")
for be, dur in boxes1:
    for (li,ji), ins, _, lv in be:
        print(f"  {ins.name}: lvars={[f'{v:.4f}' for v in lv]}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Compile Layer 2 (global X, fixed)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 5: Compile Layer 2 (global X)")
print("=" * 60)

prov2 = diffQCProvider()
qs_x_c = QSystem(); q_x_c = [Qubit(qs_x_c) for _ in range(3)]
qs_x_c.add_evolution(H_X, T_X)
prov2.compile(qs_x_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)

# For fixed layers, create a trivial "programs" with one H_list
programs_x = [([[[H_X, T_X]]], 1.0, 1)]
prov2.run(programs_x, None, T=T_X, backend="hardware", verbose=1)

boxes2 = prov2.prog[2]
print("\nLayer 2 solver instructions:")
for be, dur in boxes2:
    for (li,ji), ins, _, lv in be:
        print(f"  {ins.name}: lvars={[f'{v:.4f}' for v in lv]}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Compile Layer 3 (fixed dressing)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 6: Compile Layer 3 (fixed dressing)")
print("=" * 60)

prov3 = diffQCProvider()
qs2_c = QSystem(); q2_c = [Qubit(qs2_c) for _ in range(3)]
qs2_c.add_evolution(H2_fixed, T2)
prov3.compile(qs2_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)

programs_2 = [([[[H2_fixed, T2]]], 1.0, 1)]
prov3.run(programs_2, None, T=T2, backend="hardware", verbose=1)

boxes3 = prov3.prog[2]
print("\nLayer 3 solver instructions:")
for be, dur in boxes3:
    for (li,ji), ins, _, lv in be:
        print(f"  {ins.name}: lvars={[f'{v:.4f}' for v in lv]}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Verify Compilation Norms
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 7: Verify Compilation (per layer)")
print("=" * 60)

# Layer 1: full PSR verify
print("\n--- Layer 1 (PSR verify) ---")
result1 = prov1.verify(programs, obs, T=T1, psi0=psi0, verbose=1)

# Layer 2: norm check only (fixed, no PSR)
print("\n--- Layer 2 (norm check) ---")
norms2 = norm_check(programs_x, prov2._pulse_ledgers, 3)
for nd in norms2:
    print(f"  seg {nd['seg']}: ||H_compiled - H_target|| = {nd['norm_diff']:.2e}")

# Layer 3: norm check only
print("\n--- Layer 3 (norm check) ---")
norms3 = norm_check(programs_2, prov3._pulse_ledgers, 3)
for nd in norms3:
    print(f"  seg {nd['seg']}: ||H_compiled - H_target|| = {nd['norm_diff']:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Concatenate Schedule Ops
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 8: Concatenate Schedule Ops (all layers)")
print("=" * 60)

# Layer 1: first PSR branch (term 0, branch 0)
layer1_ops = prov1._branch_ops[0][0][0]
# Layer 2: single segment
layer2_ops = prov2._branch_ops[0][0][0]
# Layer 3: single segment
layer3_ops = prov3._branch_ops[0][0][0]

all_ops = layer1_ops + layer2_ops + layer3_ops

print(f"Layer 1: {len(layer1_ops)} ops")
print(f"Layer 2: {len(layer2_ops)} ops")
print(f"Layer 3: {len(layer3_ops)} ops")
print(f"Total:   {len(all_ops)} ops")

# Channel mapping
print("\nChannel mapping (3 qubits):")
print("  ch[0..2] = Detuning site 0,1,2")
print("  ch[3..5] = Rabi site 0,1,2")
print("  ch[6]    = Dressing (global)")
print("  ch[7]    = AOD transport")

print("\nCombined schedule:")
for i, op in enumerate(all_ops):
    dur_ns = int(op["duration"] * 1000)
    layer = "L1" if i < len(layer1_ops) else ("L2" if i < len(layer1_ops) + len(layer2_ops) else "L3")
    if op["op"] == "aod":
        print(f"  [{i:2d}] {layer} AOD   amp={op['amplitude']:.1f}  dur={dur_ns} ns")
    elif op["op"] == "play":
        print(f"  [{i:2d}] {layer} PLAY  ch={op['channel']}  amp={op['amplitude']:.4f}  dur={dur_ns} ns")
    elif op["op"] == "delay":
        print(f"  [{i:2d}] {layer} DELAY dur={dur_ns} ns")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: Emit to PulseDSL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 9: Emit to PulseDSL")
print("=" * 60)

sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")
from PulseDSL_py import Channels, Schedule, PulseLib
from PulseDSL_py.pulselib import set_platform

n_channels = 8  # 3 detuning + 3 rabi + 1 dressing + 1 AOD
ch, reg = Channels(n_channels)
schedule = Schedule()
set_platform(PulseLib.Rydberg)
aod_ch = ch[n_channels - 1]

to_pulsedsl_simple(all_ops, ch, aod_ch)
print(f"\nEmitted {len(all_ops)} ops to PulseDSL")
print("Call schedule.view() for visualization")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10: AWG compile_gates (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 10: AWG compile_gates (placeholder)")
print("=" * 60)

# Channel name mapping
ch_names = {
    0: "detuning q0", 1: "detuning q1", 2: "detuning q2",
    3: "rabi q0",     4: "rabi q1",     5: "rabi q2",
    6: "dressing",    7: "AOD",
}

# Collect per-channel waveform segments
from collections import defaultdict
channel_segments = defaultdict(list)
t_cursor = 0.0

for op in all_ops:
    dur_ns = int(op["duration"] * 1000)
    if op["op"] == "play":
        channel_segments[op["channel"]].append({
            "start_ns": int(t_cursor * 1000),
            "dur_ns": dur_ns,
            "amplitude": op["amplitude"],
            "phase": op.get("phase", 0.0),
        })
    t_cursor += op["duration"]

print("Per-channel waveform segments:")
for ch_idx in sorted(channel_segments):
    segs = channel_segments[ch_idx]
    name = ch_names.get(ch_idx, f"ch{ch_idx}")
    total_ns = sum(s["dur_ns"] for s in segs)
    print(f"  {name:15s}: {len(segs)} segments, {total_ns} ns total")
    for s in segs[:3]:  # show first 3
        print(f"    t={s['start_ns']}ns  dur={s['dur_ns']}ns  amp={s['amplitude']:.4f}  phase={s['phase']:.4f}")
    if len(segs) > 3:
        print(f"    ... ({len(segs)-3} more)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11: Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 11: Summary")
print("=" * 60)

print(f"""
Pipeline complete for two-layer dressing example.

| Layer | Hamiltonian | T | #Ops | Norm error |
|-------|-------------|------|------|------------|
| L1 (param) | sin(2x)·(Z₀Z₁+X₀+X₁) | {T1} | {len(layer1_ops)} | {max(nd['norm_diff'] for nd in result1['norm_diffs']):.2e} |
| L2 (fixed) | 5.0·(X₀+X₁+X₂) | {T_X} | {len(layer2_ops)} | {max(nd['norm_diff'] for nd in norms2):.2e} |
| L3 (fixed) | 1.2·(Z₀Z₂+X₀+X₂) | {T2} | {len(layer3_ops)} | {max(nd['norm_diff'] for nd in norms3):.2e} |

PSR gradient:     {grad_psr:.6f}
FD gradient:      {grad_fd:.6f}
L1 gradient error: {result1['error']:.2e}
Total schedule:   {len(all_ops)} ops → PulseDSL → {len(channel_segments)} active channels
""")

print("DONE")
