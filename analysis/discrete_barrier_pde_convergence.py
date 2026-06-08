"""
Convergence figure for the three-way barrier verification.

The two methods converge along DIFFERENT axes, so the figure has two panels and
each panel uses the OTHER method as the reference "truth":

  LEFT  -- PDE grid convergence (deterministic). Price vs grid spacing dS.
           * PDE-plain converges at first order O(dS) -- roughly linear in dS,
             so its dS->0 intercept (linear fit) is the true discrete price.
           * PDE-Richardson is already flat/accurate on coarse grids.
           * Grey band = MC +/- 1.96*SE (high path count) -- the reference truth.
           * Dashed line = Haug+BGK (continuous approximation): it sits OFF the
             MC band by exactly the BGK error.

  RIGHT -- MC path convergence (statistical). Price vs number of paths (log x).
           * MC estimates with +/-1.96*SE error bars; the interval shrinks ~1/sqrt(N).
           * Horizontal line = PDE-Richardson -- the deterministic reference truth.
           * The MC interval brackets the PDE line and tightens around it.

Default instrument: CUO (up-and-out call) with a cash rebate -- a knock-out
with payoff at the barrier, so it exercises both the absorption and the
rebate-at-hit machinery.
"""
import numpy as np
import matplotlib.pyplot as plt

from core.discrete_barrier_pde import DiscreteBarrierPDE
from core.discrete_barrier_mc import DiscreteBarrierMC
from core.discrete_barrier import DiscreteBarrierPricer

def make_figure(params, bt="cuo",
                pde_n_space_list=(200, 400, 800, 1600, 3200), pde_n_sub=60,
                mc_paths_list=(1000, 4000, 16000, 64000, 256000, 1024000),
                mc_truth_paths=2000000, pde_truth_n_space=1600):
    bt = bt.lower()

    # reference of MC and PDE
    mc_truth = DiscreteBarrierMC(**params, n_paths=mc_truth_paths, seed=42)
    m_star, se_star = mc_truth.price_with_se(bt)                      # MC band for LEFT panel
    pde_truth = DiscreteBarrierPDE(**params, n_space=pde_truth_n_space,
                                   n_sub=pde_n_sub).discrete_price(bt)  # PDE line for RIGHT panel

    bgk = DiscreteBarrierPricer(**params).discrete_price(bt)         # Haug+BGK dashed line

    # PDE grid convergence
    ds, plain, rich = [], [], []
    for ns in pde_n_space_list:
        d = DiscreteBarrierPDE(**params, n_space=ns, n_sub=pde_n_sub)
        s_max = d.s_max_mult * max(params["S"], params["X"], params["H"])
        ds.append(s_max / ns)
        plain.append(d.discrete_price(bt, richardson=False))
        rich.append(d.discrete_price(bt))
    ds = np.array(ds); plain = np.array(plain); rich = np.array(rich)
    # linear fit of plain vs dS -> intercept is the dS->0 limit
    c1, c0 = np.polyfit(ds, plain, 1)
    ds_line = np.linspace(0, ds.max() * 1.05, 50)

    # MC path convergence
    Ns, mc_vals, mc_ses = [], [], []
    for i, N in enumerate(mc_paths_list):
        mc = DiscreteBarrierMC(**params, n_paths=N, seed=42 + i)
        v, s = mc.price_with_se(bt)
        Ns.append(N); mc_vals.append(v); mc_ses.append(s)
    Ns = np.array(Ns); mc_vals = np.array(mc_vals); mc_ses = np.array(mc_ses)


    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # LEFT
    axL.axhspan(m_star - 1.96 * se_star, m_star + 1.96 * se_star, color="0.85",
                label=f"MC 95% CI  ({mc_truth_paths/1e6:.0f}M paths)")
    axL.axhline(m_star, color="0.55", lw=1.0, ls=":")
    axL.plot(ds_line, c0 + c1 * ds_line, color="steelblue", lw=1.0, ls=":",
             label=f"O(dS) fit  ->  intercept {c0:.4f}")
    axL.plot(ds, plain, "o-", color="steelblue", lw=1.3, ms=6, label="PDE plain  (first order)")
    axL.plot(ds, rich, "s-", color="seagreen", lw=1.3, ms=6, label="PDE Richardson")
    axL.axhline(bgk, color="crimson", lw=1.4, ls="--", label="Haug + BGK (continuous approx.)")
    axL.scatter([0], [c0], color="steelblue", marker="x", s=70, zorder=5)
    axL.set_xlabel("grid spacing  dS"); axL.set_ylabel("price")
    axL.set_title("PDE grid convergence (deterministic)")
    axL.set_xlim(left=-0.03)
    axL.legend(fontsize=8, loc="best"); axL.grid(alpha=0.25)

    # RIGHT
    axR.axhline(pde_truth, color="seagreen", lw=1.6, label=f"PDE Richardson = {pde_truth:.4f}")
    axR.errorbar(Ns, mc_vals, yerr=1.96 * mc_ses, fmt="o", color="darkorange",
                 ecolor="darkorange", elinewidth=1.3, capsize=4, ms=5,
                 label="MC  (point +/- 95% CI)")
    axR.set_xscale("log"); axR.set_xlabel("number of MC paths"); axR.set_ylabel("price")
    axR.set_title("MC path convergence (statistical)")
    axR.legend(fontsize=8, loc="best"); axR.grid(alpha=0.25, which="both")

    fig.suptitle(f"{bt.upper()}  convergence    "
                 f"S={params['S']} X={params['X']} H={params['H']} "
                 f"sigma={params['sigma']} b={params['b']} K={params['K']}  "
                 f"(discrete, {len(DiscreteBarrierPDE(**params).obs_dts)} obs)",
                 fontsize=11)
    fig.tight_layout()

    plt.show()
    print(f"  MC truth      = {m_star:.5f} +/- {se_star:.5f}")
    print(f"  PDE Richardson= {pde_truth:.5f}")
    print(f"  O(dS) intercept={c0:.5f}   Haug+BGK={bgk:.5f}  (BGK err vs PDE = {bgk - pde_truth:+.5f})")



if __name__ == "__main__":
    params = dict(start_dt="2026.05.01 14:00:00", end_dt="2026.05.28 15:00:00",
                  S=100, X=100.0, H=110.0, r=0.03, b=0.0, sigma=0.20, K=3.0)
    make_figure(params, bt="cuo")
