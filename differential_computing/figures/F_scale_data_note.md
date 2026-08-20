# F_scale data note (FSCALE_REVISION compliance)

Machine: macOS-26.5-arm64-arm-64bit; Python 3.10.0; wall-clock (time.perf_counter).
Repetitions (G6): 3 compiles/point (median, IQR band);
10 PSR branches/point timed individually (distinct tau);
16 NSR tables/point. Warm-up: one untimed branch map / table
per point before timing. Cache: F_scale_data.json (delete to re-time).

## Accounting (R5)

**"One PSR branch" is MARGINAL cost**: the evolution solve is cached in the
compiled boxes; the timed operation is TweezerMapper.map_hlist re-emitting
schedule ops + pulse ledger for one [evolve(tau), kick, evolve(T-tau)]
branch. The re-map is structurally forced (branch-specific tau and kick
insertion) — it is not a recompile.

**"One NSR branch"** = specializer.nsr_shift_table: the Nyquist branch
B + s·A shares the source segment structure, so its marginal compile is an
O(n) closed-form coefficient rescale (dressing amplitude + detunings) bound
to the shared schedule at execution. Exactness is unit-tested
(test_nsr_shift_table_realizes_shifted_target). Measured
191 µs at n=1000 — well above the
~1 µs timer floor, so it is drawn at its measured value (R1).

**Totals for one gradient** (representative m=48, matching
F6/F-loop/F-phase per G2; P=1 scalar parameter, M=999 tangent
components at n=1000): PSR = 2·m·M = 95,904 branch maps x
175 ms = 4.66 h; NSR = 2·N =
16 tables = 3.1 ms. The PSR total is a mapping
(not solver) cost and is amortizable — branches share segment structure and
tau enters only as two durations — but the amortization is future work and
the un-amortized number is the honest one today.

## Scope (R5.2)

Timing runs target -> machine-native schedule ops (concrete amplitudes,
phases, durations, positions) + pulse ledger. It does NOT include
pulse-shape synthesis: the PulseDSL emission layer (placeholder shapes,
16-channel logical cap, physical-channel COMB encoding) does not complete
at n=100 and is excluded as engineering work. Figure titles say "schedule
ops + ledger", not "device-ready pulses"; G0 wording should follow.

## Two-regime PSR slope (R4a)

Fitted per window: slope ~1.1 on n in [10, 200], ~1.5 on
n in [200, 1000] (windows stated per R4b; the global average is never
annotated). Measured per-pass attribution across n: the dressing-H ledger
rebuild is the largest single pass throughout (~48% at n=1000,
120 ms) and grows near-linearly with a heavy
constant; the ledger position-snapshot pass (n positions x ~4n plays,
structurally O(n^2)) is the FASTEST-GROWING pass — its share rises from
~20% (n<=100) to ~36% at n=1000 (89 ms) — and is
what drives the tail steepening; op emission is the remainder
(40 ms).
Caveat: the breakdown comes from a separate instrumented run whose total
(249 ms) exceeds the unwrapped headline
median (175 ms) by more than raw wrapper
arithmetic (12k wrapped calls plus allocator state); we therefore report
the breakdown as SHARES of the instrumented run, not as headline-additive
milliseconds.

## Headline ratio (R2)

One PSR branch at n=1000: 175 ms vs
60 s source compile = **0.3%**.

Suggested 6.4 sentence: "At n=1000, compiling one PSR derivative branch
costs 175 ms — 0.3% of the source
compilation it attaches to — and an NSR branch only an O(n) coefficient
table (191 µs): differentiation is not the
compilation bottleneck; the source path is."

## Scoping n=1000 (R3)

All n>12 numbers are the SPECIALIZED path (target-aware layer: frozen
chain/grid geometry, pruned bonds, analytic warm start); the generic
all-pairs path caps at n=12. Stated in both captions and panel (a).
