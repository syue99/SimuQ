"""
analytic_rescale.py — scalable analytic correction of PSR's dephasing attenuation.

Under dephasing the analog PSR gradient is attenuated: g_noisy ≈ λ·g_ideal.  λ is
NOT a universal function of T/T2*, but its leading rate IS computable from the
ideal trajectory (first-order Lindblad perturbation theory), with NO noisy
simulation.  This module computes that correction in a way that stays TRACTABLE
as qubit count grows, via Lieb–Robinson LIGHT-CONE TRUNCATION: the slope is a sum
of per-qubit terms dominated by qubits within the observable's light cone, so we
evaluate it on a small local subsystem whose size depends on evolution depth, not
total system size.

Public API
----------
slope_from_trajectory(H, O, psi0, T, z_ops, n_grid)  -> float
    Exact attenuation slope d/dθ[(1/(2T))·d<O>/dΓ/g_ideal] proxy; here we expose
    the raw per-qubit-summed d<O>/dΓ and the helpers below assemble λ.

rescale_factor(slope_lambda, T, T2)  -> float
    The multiplicative correction 1/λ ≈ exp(−slope·T/T2) to apply to g_noisy.

lightcone_qubits(edges, support, radius)  -> set
    Qubits within `radius` hops of the observable support on the interaction graph.

For a chain, chain_slope(theta, T, total_n, radius) builds the TRUNCATED local
Hamiltonian (size 2·radius+1) and returns the attenuation slope — its cost is
independent of total_n.
"""

import numpy as np
import qutip as qp

_I = qp.qeye(2); _X = qp.sigmax(); _Z = qp.sigmaz()


def _emb(op, i, n):
    return qp.tensor([op if k == i else _I for k in range(n)])


# ── light-cone graph helper ───────────────────────────────────────────────────
def lightcone_qubits(edges, support, radius):
    """BFS `radius` hops out from `support` on the interaction graph `edges`."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    frontier = set(support)
    seen = set(support)
    for _ in range(radius):
        nxt = set()
        for q in frontier:
            nxt |= adj.get(q, set())
        nxt -= seen
        seen |= nxt
        frontier = nxt
    return seen


# ── core: first-order dephasing correction from the ideal trajectory ──────────
def dO_dGamma(H, O, psi0, T, n, z_sites, n_grid=140):
    """d<O>/dΓ|_0 = ∫ Σ_{i∈z_sites} [<χ_i(t)|O|χ_i(t)> − <O>(T)] dt, χ_i=U(T−t)Z_iU(t)ψ0.

    z_sites restricts the sum to the (light-cone) qubits that matter; distant
    qubits contribute ~0, so this is exact up to the truncation radius.
    """
    ts = np.linspace(0.0, T, n_grid)
    Us = [(-1j * H * t).expm() for t in ts]
    psiT = Us[-1] * psi0
    OT = float(qp.expect(O, psiT).real)
    Zs = {i: _emb(_Z, i, n) for i in z_sites}
    integ = np.zeros(len(ts))
    for k, t in enumerate(ts):
        UTt = (-1j * H * (T - t)).expm()
        psit = Us[k] * psi0
        acc = 0.0
        for i in z_sites:
            chi = UTt * (Zs[i] * psit)
            acc += float(qp.expect(O, chi).real) - OT
        integ[k] = acc
    return float(np.trapz(integ, ts))


def lambda_slope(H_fn, O, psi0, T, n, z_sites, theta, h=1e-3, n_grid=140):
    """Slope s = dλ/d(T/T2*) at zero noise = (1/(2T))·(dg/dΓ)/g_ideal.

    H_fn(theta) returns the Hamiltonian; g_ideal and dg/dΓ are θ-derivatives of
    <O>(T) and d<O>/dΓ respectively (exact, noiseless).
    """
    def OT(th):
        return float(qp.expect(O, (-1j * H_fn(th) * T).expm() * psi0).real)

    def dOdG(th):
        return dO_dGamma(H_fn(th), O, psi0, T, n, z_sites, n_grid)

    g_ideal = (OT(theta + h) - OT(theta - h)) / (2 * h)
    dg_dG = (dOdG(theta + h) - dOdG(theta - h)) / (2 * h)
    return (dg_dG / g_ideal) / (2.0 * T)


def rescale_factor(slope, T, T2):
    """Multiplicative correction 1/λ ≈ exp(−slope·T/T2) for the PSR gradient."""
    x = T / T2
    return float(np.exp(-slope * x))


# ── chain with light-cone truncation (scalable: cost ∝ radius, not total_n) ────
def chain_H_local(theta, m):
    """Truncated chain Hamiltonian on m qubits (observable at site 0)."""
    H = 0
    for i in range(m - 1):
        H = H + theta * _emb(_Z, i, m) * _emb(_Z, i + 1, m)
    for i in range(m):
        H = H + _emb(_X, i, m)
    return H


def chain_slope(theta, T, radius, n_grid=120):
    """Attenuation slope for the chain observable Z_0, computed on the local
    light-cone subsystem of size m = radius+1 (cost independent of total chain
    length).  Returns s = dλ/d(T/T2*)."""
    m = radius + 1
    O = _emb(_Z, 0, m)
    psi0 = qp.tensor([qp.basis(2, 0)] * m)
    return lambda_slope(lambda th: chain_H_local(th, m), O, psi0, T, m,
                        z_sites=range(m), theta=theta, n_grid=n_grid)
