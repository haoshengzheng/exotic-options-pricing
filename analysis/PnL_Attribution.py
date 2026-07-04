"""
PnL attribution for Accumulators via Greek decomposition.

Explains the changed PnL of an accumulator over a short horizon (e.g. inception to the next overnight mark)
by decomposing it into contributions from each Greek, using a multi-variate Taylor expansion in
spot and volatility:

    PnL ~= Delta*dS + 0.5*Gamma*dS^2          (spot, 1st + 2nd order)
         + Vega*dSigma + 0.5*Vomma*dSigma^2   (vol, 1st + 2nd order)
         + Vanna*dS*dSigma                    (cross spot-vol)
         + Theta*dt                           (time decay)
         + (1/6)*(d3V/dSigma3)*dSigma^3       (3rd-order vol)
         + residual                           (everything unexplained)

The residual isolates whatever the Taylor expansion fails to capture:
higher-order terms, cross-terms beyond Vanna, and — most importantly for
the knock-out accumulator — the effect of the barrier, which a local
Taylor expansion around the initial spot cannot see.

Outputs
-------
1. PnL heatmap: realized PnL across a grid of joint (dS, dSigma) moves.
2. Greek decomposition bar chart: stacked Greek contributions along the
   spot axis at a fixed small vol change, overlaid with actual PnL.

Greeks are computed by the underlying pricers (analytic for the normal
accumulator, finite-difference for the knock-out). Second- and third-order
greeks (Vanna, Vomma, third-order Vega) are computed here by finite
difference on the repriced value V1.
"""
import numpy as np
import matplotlib.pyplot as plt
from core.time_utils import parse_dt,count_trading_seconds_precise, SECONDS_PER_FULL_TRADE_DAY,trading_days_per_year
from models.normal_accumulator import AccumulatorReplication
from models.ko_accumulator import KnockOutAccumulatorPricer


def build_pricer(option_type, base, S, sigma, start_dt):
    if option_type == "normal":
        return AccumulatorReplication(
            S=S, K=base['K'], B=base['B'],
            r=base['r'], b=base['b'], sigma=sigma,
            L=base['L'], PR=base['PR'],
            strike_shift=base['strike_shift'],
            option_type=base['option_type'],
            start_dt=start_dt,
            end_dt=base['end_dt'],
        )

    elif option_type == "knockout":
        return KnockOutAccumulatorPricer(
            S=S, K=base['K'], B=base['B'],
            r=base['r'], b=base['b'], sigma=sigma,
            PR=base['PR'], L=base['L'],
            rebate=base['rebate'],
            option_type=base['option_type'],
            start_dt=start_dt,
            end_dt=base['end_dt'],
        )
    else:
        raise ValueError(f"option_type {option_type} is not supported, register it in build_pricer.")


def theta_per_second(pricer):
    """Convert the pricer's per-trading-day theta into a per-second rate for fine time bumps."""
    return pricer.greeks()['theta'] / SECONDS_PER_FULL_TRADE_DAY


def compute_pnl(option_type, base_params:dict, next_pnl_dt:str, dS_fracs:tuple=(-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05),
                dsigma: tuple=(-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03),) -> dict:
    """
    Compute the PnL attribution for one accumulator at next mark.

    Reprices the position at t1 (the next mark) under a grid of joint spot
    and vol shocks, builds the PnL matrix, then decomposes the PnL along the
    spot axis (at the central vol shift) into Greek contributions plus a residual.

    PARAMETERS:

    option_type : {'normal', 'knockout'}
    base_params : Contract parameters (S, K, B, r, b, sigma, PR, L, rebate, dates...).
    next_pnl_dt : Timestamp of the next mark (t1).
    dS_fracs : Relative spot shocks for the grid.
    dsigma : Absolute vol shocks for the grid.
    """
    ann = trading_days_per_year
    S0 = base_params['S']
    sigma0 = base_params['sigma']
    t0 = parse_dt(base_params['start_dt'])
    t1 = parse_dt(next_pnl_dt)
    t0_str = base_params['start_dt']
    t1_str = t1.strftime("%Y-%m-%d %H:%M:%S")
    trade_sec = count_trading_seconds_precise(t0, t1)
    cal_sec = (t1 - t0).total_seconds()
    dt_trade = trade_sec / (ann * SECONDS_PER_FULL_TRADE_DAY)
    dt_cal = cal_sec / (365 * 86400)


    p0 = build_pricer(option_type, base_params, S0, sigma0, t0_str)
    V0 = p0.price()
    greeks0 = p0.greeks()
    theta0 = theta_per_second(p0)

    def V_t0(S, sigma):
        return build_pricer(option_type, base_params, S, sigma, t0_str).price()

    def V_t1(S, sigma):
        return build_pricer(option_type, base_params, S, sigma, t1_str).price()

    pnl_matrix = np.zeros((len(dS_fracs), len(dsigma)))
    for i, fs in enumerate(dS_fracs):
        for j, ds in enumerate(dsigma):
            pnl_matrix[i, j] = V_t1(S0 * (1 + fs), sigma0 + ds) - V0


    # Vanna and Vomma Calculation
    eps_S = S0 * 1e-4
    eps_v = 1e-4
    v_up = V_t0(S0 + eps_S, sigma0 + eps_v)
    v_dn = V_t0(S0 - eps_S, sigma0 - eps_v)
    v_mix1 = V_t0(S0 + eps_S, sigma0 - eps_v)
    v_mix2 = V_t0(S0 - eps_S, sigma0 + eps_v)
    vanna0 = (v_up + v_dn - v_mix1 - v_mix2) / (4 * eps_S * eps_v)

    v_sigma_up = V_t0(S0, sigma0 + eps_v)
    v_sigma_dn = V_t0(S0, sigma0 - eps_v)
    v_sigma_mid = V_t0(S0, sigma0)
    vomma0 = (v_sigma_up - 2 * v_sigma_mid + v_sigma_dn) / (eps_v ** 2)

    v_sigma_2up = V_t0(S0, sigma0 + 2 * eps_v)
    v_sigma_2dn = V_t0(S0, sigma0 - 2 * eps_v)
    third_order_vega = (v_sigma_2up - 2 * v_sigma_up + 2 * v_sigma_dn - v_sigma_2dn) / (2 * eps_v ** 3)


    j_mid = len(dsigma) // 2
    greek_decomp = []

    for i, fs in enumerate(dS_fracs):
        dS = S0 * fs
        sigma_change = dsigma[j_mid]

        actual = V_t1(S0 + dS, sigma0 + sigma_change) - V0
        delta_pnl = greeks0['delta'] * dS
        gamma_pnl = 0.5 * greeks0['gamma'] * dS**2
        vega_pnl = greeks0['vega'] * sigma_change * 100
        vanna_pnl = vanna0 * dS * sigma_change
        vomma_pnl = 0.5 * vomma0 * sigma_change**2
        theta_pnl = theta0 * trade_sec
        third_order_vega_pnl = (third_order_vega * sigma_change ** 3) / 6
        residual = actual - (delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + theta_pnl + vomma_pnl + third_order_vega_pnl)

        greek_decomp.append({
            'dS_frac': fs,
            'actual': actual,
            'delta': delta_pnl,
            'gamma': gamma_pnl,
            'vega': vega_pnl,
            'vanna': vanna_pnl,
            'vomma': vomma_pnl,
            'theta': theta_pnl,
            'third_order_vega': third_order_vega_pnl,
            'residual': residual,
        })


    return dict(
        pnl_matrix=pnl_matrix,
        greek_decomp=greek_decomp,
        dS_labels=[f"{f:+.0%}" for f in dS_fracs],
        dsig_labels=[f"{d*100:+.1f}%" for d in dsigma],
        t0_str=t0_str,
        t1_str=t1_str,
        V0=V0,
        base_greeks=greeks0,
        dt_trade_yr=dt_trade,
        dt_cal_yr=dt_cal,
    )

def plot_pnl(res, base_params, option_type):
    """ Plot the PnL heatmap and the stacked Greek-decomposition bar chart, and print the decomposition table to stdout."""
    fig = plt.figure(figsize=(16, 6))

    fig.suptitle(
        f"PnL [{option_type.upper()}] | S={base_params['S']} "
        f"K={base_params['K']} B={base_params['B']} σ={base_params['sigma']:.0%}\n"
        f"{res['t0_str']} → {res['t1_str']} | V0={res['V0']:.2f}"
    )

    # heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    mat = res['pnl_matrix']
    vmax = np.abs(mat).max()

    im = ax1.imshow(mat, cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax1.text(
                j, i,
                f"{val:.2f}",
                ha='center',
                va='center',
                fontsize=8,
                color='black' if abs(val) < 0.6 * vmax else 'white'
            )
    ax1.set_xticks(range(len(res['dsig_labels'])))
    ax1.set_xticklabels(res['dsig_labels'])

    ax1.set_yticks(range(len(res['dS_labels'])))
    ax1.set_yticklabels(res['dS_labels'])
    ax1.set_xlabel('Volatility Change')
    ax1.set_ylabel('Spot Return dS / S')
    ax1.set_title("PnL Heatmap")

    # bar
    ax2 = fig.add_subplot(1, 2, 2)

    dec = res['greek_decomp']
    x = np.arange(len(dec))

    keys = ['delta', 'gamma', 'vega', 'vanna', 'vomma', 'theta','third_order_vega', 'residual']
    cols = ['steelblue', 'orange', 'purple', 'brown', 'red', 'green','black', 'grey']

    pos_bot = np.zeros(len(dec))
    neg_bot = np.zeros(len(dec))

    for k, c in zip(keys, cols):
        vals = np.array([d[k] for d in dec])

        pos = np.where(vals >= 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)

        ax2.bar(x, pos, bottom=pos_bot, label=k, color=c)
        ax2.bar(x, neg, bottom=neg_bot, label="_nolegend_", color=c)

        pos_bot += pos
        neg_bot += neg

    actual = np.array([d['actual'] for d in dec])
    ax2.plot(x, actual, 'ko--', label='Actual')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(res['dS_labels'])
    ax2.set_xlabel("Spot Move dS / S")
    ax2.set_ylabel("PnL")
    sigma_slice = res['dsig_labels'][len(res['dsig_labels']) // 2]
    ax2.set_title(f"PnL Decomposition (volatility change = {sigma_slice})")
    ax2.legend()

    plt.tight_layout()
    plt.show()



    print(
        f"  {'ΔS/S':>6}  {'Actual':>9}  "
        f"{'Delta':>9}  {'Gamma':>9}  "
        f"{'Vega':>9}  {'Vanna':>9}  {'Vomma':>9}  "
        f"{'Theta':>9}  {'third_vega':>9} {'Residual':>9}"
    )

    print(f"{'─' * 90}")


    for d in res['greek_decomp']:
        print(
            f"  {d['dS_frac']:>+6.0%}  "
            f"{d['actual']:>9.3f}  "
            f"{d['delta']:>9.3f}  "
            f"{d['gamma']:>9.3f}  "
            f"{d['vega']:>9.3f}  "
            f"{d['vanna']:>9.3f}  "
            f"{d['vomma']:>9.3f}  "
            f"{d['theta']:>9.3f}  "
            f"{d['third_order_vega']:>9.3f}  "
            f"{d['residual']:>9.3f}"
        )

    print(f"{'─' * 90}")


if __name__ == '__main__':
    params = dict(
        S=1219, K=1175, B=1240,
        r=0.03, b=0.0, sigma=0.23,
        L=3.0, PR=1.0,  rebate=10.0,
        strike_shift=0.002,
        option_type='call',
        start_dt="2026.05.19 15:00:01",
        end_dt="2026.06.23 15:00:00",
        seed=42,
    )

    res = compute_pnl(option_type= "normal",
        base_params=params, next_pnl_dt="2026.05.19 21:00:00",
        dS_fracs=(-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05),
        dsigma=(-0.05, -0.03, -0.01, 0.005, 0.01, 0.03, 0.05),
     )

    plot_pnl(res, params, "normal")