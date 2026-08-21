# F_select data note

Sweep: NQ=7, T=1.0, P in 1..20,
k in 1..14, 6 seeds/cell (means of
log quantities). Alphabet: 7 X + 7 Z
+ 21 ZZ terms. Units R=1, dt=1 (caption states the
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
general 15.4% of cells, aligned
10.7%. The divergence is one-sided and safe
(Remark B.1: a loose certificate costs shots, never bias): following the
certificate forfeits at most 1.86x shots on the general pool
and 1.71x on the aligned pool — inside the 10x margin
everywhere, and mostly inside 2x. The looseness is the Assumption-4.4
diameter certificate: measured diam concentrates below 2 Sum|v| for
random-sign tangents (operator-norm concentration); the sqrt(2) branch
bound is nearly tight here (measured <sigma> = 1.39).
There is no meaningful predicted CROSSING to overlay, so the main figure
carries the finding in its caption instead of a degenerate contour; the
appendix draws the aligned pool's measured crossing (alignment, chi -> 1,
shrinks the NSR region — visible as the dash-dot boundary left of the
solid one).

Takeaway sentence for 6.3: the compiler can compute its choice before
running anything; where its certificate is loose the error is bounded
(<= 1.9x shots here) and always lands on the bias-free side.

NSR-win fraction (honesty rail): general 15.4%,
aligned 10.7% — NSR does win on a Pauli-only system
in the small-P, large-k corner (subextensive tangents).

Telescoping (chi = 1/m): NOT drawable in this alphabet. The compressing
family Sum_j (Z_j - Z_{j+1}) has non-involutive generators — precisely the
Assumption-4.7-fails, NSR-only regime — and commuting PAULI families are
jointly extremizable (chi = Theta(1); frustration gives constant factors,
never 1/m). This regime is covered by the theorem, not by measurement;
6.3 states it in one sentence together with the non-involutive table row.

The aligned (ZZ-only) crossing appears in the appendix variant
(F_select_appendix): coefficient alignment moves the boundary toward PSR
(chi -> 1 removes NSR's folding headroom).
