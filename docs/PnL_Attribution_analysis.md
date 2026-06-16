# PnL Attribution

Decomposing the realized PnL of an accumulator position (from the client's
long perspective) into Greek contributions, and interpreting the residual —
especially how the knock-out barrier reshapes both the PnL profile and the
residual.

---

## Purpose

When moving from the close of one trading day to the next trading day, spot and volatility move, 
and the mark-to-market PnL of accumulator shifts. A trading desk needs to answer two questions:


1. How well does a first-/second-order Greek model explain the realized PnL?
2. What is left in the residual, and what does it reveal about the
   position's hidden risks?

All analysis is from the **client's (long) perspective**: the client
buys and holds the accumulator.

---

## Methodology

For a move (dS, dSigma) over a short horizon dt, realized PnL is approximated by
a multivariate Taylor expansion of the value function:

$$
\begin{aligned}
\text{PnL} \approx\&
\underbrace{\Delta \ dS + \tfrac12 \Gamma \ dS^2}_{\text{spot, 1st + 2nd order}}
\
&+ \underbrace{\text{Vega} \ d\sigma + \tfrac12 \text{Vomma} \ d\sigma^2}_{\text{vol, 1st + 2nd order}}
\
&+ \underbrace{\text{Vanna} \ dS \ d\sigma}_{\text{cross spot-vol}}
\
&+ \underbrace{\Theta \ dt}_{\text{time decay}}
\
&+ \underbrace{
\tfrac16 \frac{\partial^3 V}{\partial \sigma^3} d\sigma^3
}_{\text{3rd-order vol}}
\
&+ \text{residual}
\end{aligned}
$$


The residual is defined as actual PnL minus all explained terms. First-order
Greeks come from the pricer (analytic for the normal accumulator,
finite-difference for the knock-out). Vanna, Vomma and third-order Vega are
computed here by finite difference on the repriced value.

The decomposition slice in the bar chart is taken at a +0.5% vol shift.
Third-order Vega is included for completeness but is negligible in this
scenario — with a vol shift of only 0.5%, the $\sigma^3$ term is on the order of
1e-7. It becomes relevant only under large vol shocks (e.g. ±5% stress
scenarios).

### Test scenario

| Parameter                       | Value                         |
|---------------------------------|-------------------------------|
| Spot S                          | 1219                          |
| Strike K                        | 1175                          |
| Barrier B                       | 1240 (only ~1.7% above spot)  |
| Volatility                      | 23%                           |
| Participation PR                | 1.0                           |
| Leverage L                      | 3.0                           |
| Rebate (KO)                     | 10.0                          |
| Horizon                         | 15:00:01 to 21:00:00 same day |
| Vol shift (decomposition slice) | +0.5%                         |

Because spot is only 1.7% below the barrier, a +3% spot move already breaches
B and triggers knock-out — which is what makes this scenario useful for
studying barrier effects.

---

## A note on Theta = 0 

Across all rows, Theta contributes = 0 — but this is partly an artifact of how
the attribution computes it. The theta term uses a single trade-clock rate
(`theta_per_second × trade_sec`), implicitly treating all of theta as diffusion
decay and dropping the carry/discount part that accrues on calendar time. The
marking window (15:00:01 to 21:00:00) falls entirely between the afternoon close
and the night open, so trading-seconds elapsed is = 0 and this trade-clock term
vanishes. The stricter form — already in the vanilla pricer as
`theta_components` / `theta_per_observation` — splits theta into trade-clock
($\partial V/\partial T_{trade}$, diffusion) and calendar-clock
($\partial V/\partial T_{cal}$, carry + discount) components, each multiplied by
its own elapsed time. Here the dropped calendar term is genuinely small (b = 0,
and r = 3% over 6 hours discounts negligibly), so Theta = 0 is close to the true
value anyway — but under non-zero carry or a window spanning an open session,
that dropped term would become a real error.

---

## Result 1: Knock-Out Accumulator

![`PnL_Attribution_Knockout`](../images/PnL_Attribution_Knockout.png)

### The PnL heatmap (left panel)

The heatmap shows realized PnL across joint spot (rows) and vol (columns)
shocks. Two features dominate:

- **Heavy, uncapped downside.** At -5% spot, PnL is around -2000 across all
  vol shocks. The leveraged loss leg (L = 3) means downside losses pile up
  fast and are largely insensitive to vol.

- **Upside capped by knock-out.** The +3% and +5% rows read **395.72 in every
  cell, identical across all vol shocks**. Once spot rises ~3% it breaches the
  barrier, the contract knocks out, and PnL locks at the rebate value. Vol no
  longer matters because the position is dead.

This plateau is the defining signature of the knock-out accumulator: **unbounded
downside, capped upside**.

### The Greek decomposition (right panel) and the residual

A crucial point about the decomposition: the Greeks are measured **at the
initial spot S0 = 1219, while the position is still alive**. The Taylor
expansion uses these live Greeks to extrapolate PnL to each spot shock — but it
has no knowledge that the barrier exists.

This produces a subtle and important pattern in the residual:

| ΔS/S |    Actual |     Delta |    Gamma |    Vega |   Vanna | Residual |
|------|----------:|----------:|---------:|--------:|--------:|---------:|
| -5%  | -2046.915 | -1102.007 | -817.789 | -23.821 | -30.588 |  -72.831 |
| -3%  | -1013.165 |  -661.204 | -294.404 | -23.821 | -18.353 |  -15.504 |
| -1%  |  -282.925 |  -220.401 |  -32.712 | -23.821 |  -6.118 |    0.006 |
| +0%  |   -23.706 |     0.000 |   -0.000 | -23.821 |   0.000 |   -0.006 |
| +1%  |   172.496 |   220.401 |  -32.712 | -23.821 |   6.118 |    2.389 |
| +3%  |   395.718 |   661.204 | -294.404 | -23.821 |  18.353 |   34.266 |
| +5%  |   395.718 |  1102.007 | -817.789 | -23.821 |  30.588 |  104.613 |

(Vomma ≈ +0.12, Theta = 0, third-order Vega ≈ 0 across all rows
— omitted from the table for clarity.)

Within ±1% spot, (Delta + Gamma + Vega + Vanna) explain the PnL almost completely (the
residual is essentially zero at -1% and only +2.4 at +1%). The residual grows
as spot moves further, reaching +104.6 at +5% and -72.8 at -5%. This is the
expected O($dS^{3}$) truncation error of a second-order expansion — and its
structure carries information about the barrier, discussed below.

### What happens to the Greeks once knocked out?

The decomposition uses **t0 Greeks (position alive)**. If spot actually breaches
the barrier at t1, the contract is dead and one will get the corresponding rebate:
delta, gamma and vega all collapse to ~zero (only rate sensitivity on the
rebate discounting remains). The Taylor expansion, anchored on the live t0
Greeks, cannot represent this collapse.

So the residual near the barrier is not noise — it is the quantitative measure
of the knock-out effect that local Greeks structurally cannot capture. This is
precisely why barrier products require either very frequent re-hedging. (See
[`analysis/bump_size.py`](../analysis/bump_size.py) for how the Greeks
themselves become numerically unstable as spot approaches B.)

---

## Result 2: Normal Accumulator (no barrier)

![`PnL_Attribution_Normal.png`](../images/PnL_Attribution_Normal.png)

The normal accumulator shares the same daily payoff away from the barrier, so
its downside behavior is similar. The crucial difference is the upside.

| ΔS/S |    Actual |    Delta |    Gamma |    Vega |   Vanna | Residual |
|------|----------:|---------:|---------:|--------:|--------:|---------:|
| -5%  | -1970.482 | -925.640 | -948.121 | -35.098 | -16.675 |  -44.921 |
| -3%  |  -953.440 | -555.384 | -341.324 | -35.098 | -10.005 |  -11.602 |
| -1%  |  -261.659 | -185.128 |  -37.925 | -35.098 |  -3.335 |   -0.147 |
| +0%  |   -35.130 |    0.000 |   -0.000 | -35.098 |   0.000 |   -0.006 |
| +1%  |   117.685 |  185.128 |  -37.925 | -35.098 |   3.335 |    2.271 |
| +3%  |   256.762 |  555.384 | -341.324 | -35.098 |  10.005 |   67.821 |
| +5%  |   291.617 |  925.640 | -948.121 | -35.098 |  16.675 |  332.547 |

Without a barrier, the upside is uncapped. At +5%, Delta (+925.64) and Gamma
(-948.12) net to -22.48, so the quadratic Taylor extrapolation predicts a small
negative PnL. But actual PnL is +291.62 — far above the extrapolation. The gap
produces a large residual of +332.55.

This residual reflects the higher-order curvature of the accumulator
payoff under a large spot move, not a barrier truncation. The normal
accumulator's payoff keeps gaining as spot rises (until the soft cap at B),
and a second-order Taylor expansion anchored at S0 cannot keep up.

---

## Normal vs Knock-Out: side-by-side

|                 | Normal  | Knock-Out |
|-----------------|---------|-----------|
| Residual at -5% | -44.92  | - 72.83   |
| Residual at -3% | -11.60  | -15.50    |
| Residual at -1% | -0.15   | 0.006     |
| Residual at +1% | +2.27   | +2.39     |
| Residual at +3% | +67.82  | +34.27    |
| Residual at +5% | +332.55 | +104.61   |

### The key difference: reversible soft cap vs permanent knockout

Both products cap the daily upside — for the call accumulator, the daily
payoff is zero once spot is at or above B. The difference is **what kind of
cap**:

- **Normal accumulator: a reversible daily soft cap.** On a day when spot is
  above B, that day's payoff is zero. But if spot falls back below B the next
  day, the payoff resumes. The contract never dies, so its total value still
  responds to spot — which is why the +5% PnL (+292) differs from the +3% PnL
  (+257); the remaining observation days continue to carry spot exposure.

- **Knock-out accumulator: a one-touch permanent cap.** The first time spot
  touches B, the entire contract terminates and pays only the rebate for all
  remaining days. This is why the +3% and +5% rows are identical (395.72): once
  knocked out, additional spot moves change nothing.

So the +3%/+5% plateau is unique to the KO not because the normal has no cap,
but because the normal's cap is reversible and per-day, while the KO's is
permanent and path-dependent.

### How well do the First and second order explain the PnL
**First-/second-order Greeks explain small moves well but degrade for large
ones.** Within ±1% spot, Delta + Gamma + Vega + Vanna capture nearly all the PnL; beyond
±3%, the residual grows as the quadratic Taylor expansion breaks down.

### Residual analysis

The residual measures the risk that local Greeks cannot see, and the two
products' residuals have **different origins**.

**The residual is asymmetric — upside larger than downside.** 

For the normal
accumulator, the +5% residual (+332.55) dwarfs the -5% residual (-44.92). This
maps directly onto the payoff structure:

- **Upside**: as spot rises toward B, the payoff transitions from linear
  participation (PR) into the soft cap near the boundary. This sharp change in
  curvature produces large higher-order terms — the quadratic Taylor expansion
  fails hardest here.
- **Downside**: the payoff is a smooth linear leverage leg (-L). Curvature
  changes gently, so higher-order terms stay small.


**The knock-out limits the upside residual.** The KO's +5% residual (+104.61) is
far smaller than the normal's (+332.55). Because the KO knocks out around +3%
and pins actual PnL at the rebate (395.72), the bounded actual value stays
relatively close to the Delta+Gamma extrapolation. The normal accumulator, with
no permanent cap, lets value drift far above the quadratic extrapolation on the
upside, producing a much larger residual.



---


## Reading the Greek signs

The signs of the Greek contributions encode the entire risk character of the
accumulator. From the client's long perspective:

### Delta > 0 

`delta_pnl = Delta × dS`. The contribution is negative when spot falls and
positive when spot rises, so Delta itself is positive 
(In most cases, when the spot is near the barrier level, the sign of delta
could change). The call accumulator
pays the client when spot rises (participation PR) and costs the client when
spot falls (leverage L) — a fundamentally bullish position, hence positive
delta.

### Gamma < 0 (the client is short convexity)

`gamma_pnl = ½ × Gamma × dS²`. Since dS² ≥ 0, the contribution's sign equals
Gamma's sign — and it is negative in **every** row. The client is short
convexity: the upside participates only linearly (PR = 1), while the downside
is linearly leveraged (L = 3). This "slow up, fast down" payoff has negative
second derivative.

Negative gamma means the client loses on the gamma term regardless of
direction — a large move either way hurts. This is the mathematical root of
the accumulator's reputation as a "sell convexity" product: the client is
betting on calm markets and is penalized by volatility.

### Vega < 0 (the client is short volatility)

Vega is negative (about -70 per 1% vol at this point; the table shows the
contribution at +0.5%, hence ~-35). The payoff is asymmetric: modest linear
upside (PR = 1, capped at B) versus heavily leveraged downside (L = 3). Higher
vol raises the probability of spot reaching the extreme regions, and because
the downside is leveraged while the upside is capped, the unfavorable tail
grows faster than the favorable one. So higher vol hurts the client. Lower vol
makes the position more valuable.



### Putting it together: the asymmetry

The interaction of these signs produces the accumulator's signature
asymmetry. Compare the -5% and +5% rows:

- **At -5% (spot crashes):** delta_pnl (-925) and gamma_pnl (-948) are both
  negative and **stack**, producing a catastrophic actual PnL of -1970.

- **At +5% (spot rallies):** delta_pnl (+925) is positive but gamma_pnl (-948)
  is still negative, so they **offset**. The delta gain is eaten by the gamma
  loss, leaving only +291.

The client loses heavily on the downside (delta and gamma compound) but gains
only modestly on the upside (delta gain partly cancelled by gamma loss). This
"lose more, gain less" profile is the direct consequence of L > PR and only the gain only happens between K and B,
and it is  why the accumulator can inflict severe losses on a buyer in a sharp sell-off.



