# Accumulator Static Replication

The examples below all use call scenarios; the same logic applies to put
scenarios.

Replicating the daily piecewise-linear payoff of normal accumulator
into a portfolio of three vanilla puts.

---

## Introduction

- An accumulator's single-day payoff is **piecewise-linear** in spot, with a
  cliff at $S = B$ that drops the payoff to zero.
- A naive 2-strike portfolio (sell K-put + sell B-put) **cannot** replicate
  the cliff — it consistently underprices the accumulator by a constant
  $PR(B - K)$ across the entire region $S < B$.
- Carr-Madan replication requires the payoff to be free of jump discontinuities.
  I therefore **regularize** the cliff into a steep ramp over $[B, B(1+\varepsilon)]$,
  introducing a third strike $K_3 = B(1+\varepsilon)$.
- After regularization, the Carr-Madan integral collapses to a finite sum of
  three vanilla puts with explicit weights:
  $w_1 = PR \cdot R$, $w_2 = w_1 + PR$, $w_3 = L - PR$, where $R = (K_2 - K_1)/(K_3 - K_2)$.
- The strike-shift $\varepsilon$ is an engineering parameter, not a model
  output. Smaller $\varepsilon$ tracks the true cliff more faithfully but
  blows up weights as $1/\varepsilon$. Default $\varepsilon = 0.002$ gives
  weights around 500x.

---

## 1. Contract Specification

For a **call normal accumulator** with strike $K$, upper boundary $B > K$,
participation rate $PR$, and loss leverage $L \geq PR$ pays on each
observation date $t_i$:

$$
f(S_{t_i}) =
\begin{cases}
-L \cdot (K - S_{t_i}) & \text{if } S_{t_i} < K \quad \text{(downside, leveraged)} \\
+PR \cdot (S_{t_i} - K) & \text{if } K \leq S_{t_i} < B \quad \text{(gain zone)} \\
0 & \text{if } S_{t_i} \geq B \quad \text{(capped out)}
\end{cases}
$$

The total contract value at inception is the discounted sum across all
observation dates:

$$
V_0 = \sum_{i=1}^{N} e^{-r \cdot T_{cal,i}} \cdot \mathbb{E}^{\mathbb{Q}}[f(S_{t_i})]
$$

For a **put accumulator** with $B < K$, the payoff is the mirror image
(sign-flipped and reflected). The same replication framework applies with
roles of puts and calls swapped.

### Typical Parameters

A real-world example from the Chinese commodity OTC market:

| Parameter | Symbol | Typical Value |
|---|---|---|
| Spot | $S_0$ | 3362 |
| Strike | $K$ | 3307.87  |
| Upper boundary | $B$ | 3409  |
| Participation rate | $PR$ | 1.0 |
| Loss leverage | $L$ | 2.0 |
| Volatility | $\sigma$ | 9% |
| Tenor | $T$ | ~ 60 calendar days |

---

## 2. Why a 2-Vanilla Portfolio Cannot Replicate

The most natural first attempt is to use two puts: sell one put with strike $K$
and another put with strike $B$. Matching slopes on both sides of $B$:

- For $S < K$: slope must be $+L$
- For $K \leq S < B$: slope must be $+PR$

Solving for two weights $a$ and $b$ (puts sold at $K$ and $B$ respectively):

$$
g(S) = -a \cdot (K - S)^+ - b \cdot (B - S)^+
$$

gives $a = L - PR$ and $b = PR$. Computing the resulting payoff at three
test points (with $K = 100$, $B = 110$, $PR = 1$, $L = 2$):

| $S$ | Accumulator $f(S)$ | 2-put portfolio $g(S)$ | Difference |
|---|---|---|---|
| $90$ | $-20$ | $-30$ | $-10$ |
| $100$ | $0$ | $-10$ | $-10$ |
| $110$ | $+10$ | $0$ | $-10$ |
| $120$ | $0$ | $0$ | $0$ |

The 2-put portfolio is **uniformly below** the accumulator by exactly
$PR(B - K) = 10$ for all $S < B$. This difference is the **cliff** at $S = B$.


---

## 3. The Strike-Shift Modification

We replace the true accumulator with an approximation $\tilde{f}$ that
softens the cliff into a steep linear ramp over $[B, B(1+\varepsilon)]$:

$$
\tilde{f}(S) =
\begin{cases}
-L(K - S) & S < K \\
+PR(S - K) & K \leq S < B \\
PR(B - K) \cdot \dfrac{B(1+\varepsilon) - S}{B\varepsilon} & B \leq S < B(1+\varepsilon) \\
0 & S \geq B(1+\varepsilon)
\end{cases}
$$

Now the payoff has:
- **Three kinks** at $K_1 = K$, $K_2 = B$, $K_3 = B(1+\varepsilon)$
- **Four piecewise-linear regions** between them
- **No discontinuities** — $\tilde{f}$ is continuous everywhere

### The Trade-Off

| Smaller $\varepsilon$   | Larger $\varepsilon$   |
|-------------------------|------------------------|
| Closer to true cliff    | More payoff distortion |
| Harder to hedge near B  | Easier to hedge near B |

The default $\varepsilon = 0.002$ is just a value used in this example 
and does not represent the optimal tradeoff.

### Crucial Point

The strike $K_3 = B(1+\varepsilon)$ is **not** derived from Carr-Madan.
It is an engineering decision made **before** Carr-Madan is applied.
Carr-Madan only operates on the smoothed payoff $\tilde{f}$ and produces
the corresponding three-vanilla portfolio.

---

## 4. Carr-Madan Applied to $\tilde{f}$

### 4.1 The Carr-Madan Identity

Carr-Madan (1998) states: for any sufficiently smooth European payoff
$f$ and any reference level $\kappa$,

$$
f(S_T) = f(\kappa) + f'(\kappa)(S_T - \kappa)
       + \int_0^{\kappa} f''(z) (z - S_T)^+ \, dz
       + \int_{\kappa}^{\infty} f''(z) (S_T - z)^+ \, dz
$$

The first integral is a continuum of puts (strikes $z < \kappa$), the
second is a continuum of calls (strikes $z > \kappa$).

### 4.2 Choice of Reference Point

For accumulator we choose $\kappa \to +\infty$ :

- $\tilde{f}(\kappa) = 0$ since $\kappa$ is in the flat region beyond $K_3$
- $\tilde{f}'(\kappa) = 0$ for the same reason
- The call-leg integral vanishes because $\tilde{f}''(z) = 0$ for $z > K_3$

The formula reduces to:

$$
\tilde{f}(S_T) = \int_0^{\infty} \tilde{f}''(z) (z - S_T)^+ \, dz
$$

### 4.3 Computing $\tilde{f}''$ in the Distribution Sense

The four-region slopes of $\tilde{f}$:

| Region | Range | Slope $\tilde{f}'(z)$ |
|---|---|---|
| I | $z < K_1$ | $+L$ |
| II | $K_1 < z < K_2$ | $+PR$ |
| III | $K_2 < z < K_3$ | $-\dfrac{PR(B-K)}{B\varepsilon}$ |
| IV | $z > K_3$ | $0$ |

For a piecewise-linear function, the distributional second derivative is
a sum of Dirac deltas at the kinks, weighted by the slope jumps
$\Delta \tilde{f}' = \tilde{f}'(z^+) - \tilde{f}'(z^-)$:

- At $z = K_1$: $\Delta \tilde{f}' = PR - L$
- At $z = K_2$: $\Delta \tilde{f}' = -\dfrac{PR(B-K)}{B\varepsilon} - PR$
- At $z = K_3$: $\Delta \tilde{f}' = +\dfrac{PR(B-K)}{B\varepsilon}$

So:


$\tilde{f}''(z) = (PR - L) \, \delta(z - K_1)$
$- \left[ \tfrac{PR(B-K)}{B\varepsilon} + PR \right] \delta(z - K_2)$
$+ \tfrac{PR(B-K)}{B\varepsilon} \, \delta(z - K_3)
$

### 4.4 Sifting the Integral

The Dirac delta has the sifting property: $\int \delta(z - K_i) \, g(z) \, dz = g(K_i)$.
Applying this to each term:


$\tilde{f}(S_T) = (PR - L)(K_1 - S_T)^+$
$- \left[ \tfrac{PR(B-K)}{B\varepsilon} + PR \right] (K_2 - S_T)^+$
$+ \tfrac{PR(B-K)}{B\varepsilon} (K_3 - S_T)^+
$

This is a **finite combination of three vanilla put payoffs**.

### 4.5 Final Weights

Define $R = \dfrac{K_2 - K_1}{K_3 - K_2} = \dfrac{B - K}{B \varepsilon}$.
Rearranging signs so all weights are positive:

$$
\boxed{
\tilde{f}(S_T) = w_1 \cdot (K_3 - S_T)^+ - w_2 \cdot (K_2 - S_T)^+ - w_3 \cdot (K_1 - S_T)^+
}
$$

with

$$
w_1 = PR \cdot R, \qquad w_2 = w_1 + PR, \qquad w_3 = L - PR
$$

This is exactly what is implemented in
[`models/normal_accumulator.py`](../models/normal_accumulator.py):

```python
self.K1 = K
self.K2 = B
self.K3 = B * (1.0 + strike_shift)
self.ratio = (self.K2 - self.K1) / (self.K3 - self.K2)
self.w1 = PR * self.ratio
self.w2 = PR * self.ratio + PR
self.w3 = L - PR
```

---

## 5. Endpoint Verification

We verify the formula at two endpoints to confirm the algebra.

**Case A: $S_T = K_2 = B$** (spot at the boundary)

By definition: $\tilde{f}(B) = PR(B - K)$.

By replication:

$$
w_1 \cdot (K_3 - B)^+ - w_2 \cdot 0 - w_3 \cdot 0
= PR \cdot R \cdot B\varepsilon
= PR \cdot \tfrac{B - K}{B\varepsilon} \cdot B\varepsilon
= PR(B - K) 
$$

**Case B: $S_T = 0$** (extreme downside)

By definition: $\tilde{f}(0) = -LK$.

By replication:

$$
w_1 K_3 - w_2 K_2 - w_3 K_1
$$

Substituting $w_1 = PR(B-K)/(B\varepsilon)$, $w_2 = w_1 + PR$, $w_3 = L - PR$,
$K_3 = B(1+\varepsilon)$, $K_2 = B$, $K_1 = K$:


$= \frac{PR(B-K)}{B\varepsilon} \cdot B(1+\varepsilon)$
$- \left[\frac{PR(B-K)}{B\varepsilon} + PR\right] \cdot B$
$- (L - PR) K
$

$
= PR(B-K) \cdot \frac{1+\varepsilon - 1}{\varepsilon} - PR \cdot B - (L-PR) K
= PR(B-K) - PR \cdot B - (L-PR) K = -LK 
$

---

## 6. Full Contract Pricing

The contract value at inception is the sum of replication-portfolio values
across all $N$ observation dates:

$$
V_0 = w_1 \sum_{i=1}^{N} P(K_3, t_i) - w_2 \sum_{i=1}^{N} P(K_2, t_i) - w_3 \sum_{i=1}^{N} P(K_1, t_i)
$$

where $P(K, t_i)$ is the time-0 value of a vanilla put with strike $K$
expiring at $t_i$. Each $P(K, t_i)$ is computed with the dual-time BSM
formula: diffusion driven by $T_{trade, i}$, discounting by $T_{cal, i}$


Greeks follow the same linear structure — since the replication is static
(weights don't depend on $S$ or $\sigma$), each Greek of the accumulator
is the same weighted sum of vanilla Greeks.

---

## 7. Worked Numerical Example

Using the typical parameters above:

| Quantity                     | Value               |
|------------------------------|---------------------|
| $S_0$                        | 1219                |
| $K_1 = K$                    | 1175                |
| $K_2 = B$                    | 1263                |
| $K_3 = B(1+\varepsilon)$     | 1265.526            |
| $R$                          | 1                   |
| $L$                          | 3                   |
| strike shift ($\varepsilon$) | 0.002               |
| $\sigma$                     | 0.23                |
| $r$                          | 0.03                |
| $b$                          | 0                   |
| option type                  | call                |
| start_dt                     | 2026.05.19 14:09:01 |
| end_dt                       | 2026.06.23 15:00:00 |


The full pricer output (run `python -m models.normal_accumulator`):

$$
\begin{array}{lcl}
\text{=====Accumulator Summary=====}\\
\text{Price} :\ 15.997635 \\
\text{Delta} :\ +19.933029 \\
\text{Gamma} :\ -0.488764 \\
\text{Vega} :\ -75.350854 \\
\text{Theta} :\ +79.383089
\end{array}
$$

## 8. Limitations

- **Modification error**: The smoothed payoff $\tilde{f}$ differs from
  the true $f$ over the ramp region $[B, B(1+\varepsilon)]$. 

- **Flat-vol assumption**: The model assumes volatility is a constant. However, Under a non-flat smile, the three strikes $K_1, K_2, K_3$ would
  have inconsistent implied vols, and the static-replication weights
  would no longer be exact hedge ratios.

## Extension to Knock-Out

The KO accumulator has the same daily payoff as the normal version, but
adds a knock-out feature: once spot touches $B$, the contract terminates 
with a fixed rebate. The Carr-Madan replication no longer applies because 
the payoff becomes path-dependent. Instead, we price each observation day's 
contribution via the Haug closed-form barrier formula with BGK discrete 
correction (see [`models/ko_accumulator.py`](../models/ko_accumulator.py).
