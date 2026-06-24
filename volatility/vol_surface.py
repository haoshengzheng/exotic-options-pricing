"""
Implied volatility surface for US equity call options — TSLA data.

IV is solved from mid = (bid + ask) / 2 using the Newton-Raphson / bisection
solver in volatility.iv_solver, which internally uses core.time_utils for
precise US-market trading-seconds calculation.


Notes on American option pricing：

The data set contains only CALL options on TSLA (no regular dividend).
By Merton (1973), early exercise of a call on a non-dividend-paying stock is
never optimal → American call = European call → BSM is exact for pricing and
IV inversion.  The iv_solver uses BSM (VanillaBSM with the b-parameter), so
it is fully correct here.

For American PUT options the argument breaks down (early exercise can be
optimal for deep ITM puts), and BSM would underestimate the put price,
causing the back-solved IV to be systematically too high.  A dedicated
American-put pricer (e.g. Barone-Adesi & Whaley or a binomial tree) would
be required — but that is outside the scope of this file.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize, brentq
from scipy.stats import norm

from core.time_utils_US import parse_dt,count_trading_seconds_precise,SECONDS_PER_FULL_TRADE_DAY, trading_days_per_year
from core.vanilla import VanillaBSM


warnings.filterwarnings('ignore')

def _bsm_delta(S: float, K: float, T_trade: float, T_cal: float, r: float, b: float, sigma: float) -> float:
    """BSM call delta under the dual-time convention (T_trade drives the diffusion d1, T_cal drives the carry/discount factor)."""
    if T_trade <= 0:
        return 1.0 if S >= K else 0.0
    d1 = (np.log(S / K) + b * T_cal + 0.5 * sigma ** 2 * T_trade) / (sigma * np.sqrt(T_trade))
    return np.exp((b - r) * T_cal) * norm.cdf(d1)


def _strike_from_delta(target_delta: float, S: float, T_trade: float, T_cal: float,
                       r: float, b: float, sigma: float) -> float:
    """Invert delta to a strike via Brent — used to place approximate delta tick marks (10Δ..90Δ) on the smile plots."""
    try:
        res = brentq(lambda K: _bsm_delta(S, K, T_trade, T_cal, r, b, sigma) - target_delta,
                      S * 0.01, S * 10, xtol=0.01,)
        strike_k = res[0] if isinstance(res, (tuple, list)) else res
        return float(strike_k)
    except ValueError:
        return np.nan




def _svi_w(x: np.ndarray, a, b, rho, m, sig) -> np.ndarray:
    """Raw SVI total-variance slice: w(x) = a + b[rho(x−m) + sqrt((x−m)^2 +sigma^2)],  x = ln(K/F),  w = IV²·T."""
    return a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sig ** 2))


def _fit_svi(x: np.ndarray, iv: np.ndarray, T: float):
    """Fit SVI to one expiry slice. Returns (a,b,ρ,m,σ) or None."""
    w_obs = iv ** 2 * T

    def loss(p):
        a, b_, rho, m_, sig = p
        if b_ < 0 or sig < 1e-5 or abs(rho) >= 1 or a + b_ * sig * np.sqrt(1 - rho ** 2) < 0:
            return 1e9
        return float(np.sum((_svi_w(x, a, b_, rho, m_, sig) - w_obs) ** 2))

    w0      = float(np.median(w_obs))
    starts  = [
        [max(w0 * 0.5, 1e-4), 0.10, -0.30, float(np.median(x)), 0.10],
        [max(w0 * 0.3, 1e-4), 0.15, -0.50,  0.0,                0.20],
        [max(w0 * 0.7, 1e-4), 0.05, -0.10, float(np.mean(x)),   0.05],
    ]
    bounds  = [(1e-6, 2), (1e-4, 3), (-0.999, 0.999), (-1.5, 1.5), (1e-5, 2)]
    best_v, best_x = np.inf, None
    for x0 in starts:
        res = minimize(loss, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 600, 'ftol': 1e-15})
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return best_x



class VolSurface:
    """
    Implied-volatility surface for TSLA call options under the dual-time framework.

    Pipeline: load option quotes → solve IV from mid prices (robust
    Newton/bisection) → clean (volume, spread, delta-range, log-moneyness
    filters) → interpolate a thin-plate-spline RBF surface on (log-moneyness,
    sqrtT). SVI is fitted per expiry for the smile plots; the surface itself is the RBF.

    Produces four figures: term structure, per-expiry smiles (with SVI),
    3-D surface + skew/convexity/dispersion diagnostics, and sticky-rule
    smile dynamics.
    """
    def __init__(self, filepath: str, r: float = 0.04, b:float = 0.04, min_volume: int = 0,min_price: float = 0.50,
                 max_spread_pct: float = 0.25,  delta_range: tuple = (0.05, 0.95),  lm_range: tuple = (-0.55, 0.80),  ):

        self.filepath       = filepath
        self.r, self.b      = r, b
        self.min_volume, self.min_price, self.max_spread_pct = min_volume, min_price, max_spread_pct
        self.delta_range, self.lm_range    = delta_range, lm_range

        self._load_and_solve()
        self._build_rbf()


    @staticmethod
    def _solve_iv_fast(price_mkt: float, S: float, K: float,T_trade: float, T_cal: float,r: float, b: float,
                       sigma_low: float = 1e-4, max_iter: int = 100, sigma_high: float = 10.0,epsilon: float = 1e-6) -> float:
        """Invert BSM price to IV. Brenner-Subrahmanyam ATM seed, then
        Newton-Raphson on vega, falling back to bisection when vega is too
        small (deep OTM). Returns NaN if the price violates no-arbitrage bounds."""
        phi = 1
        v_high = VanillaBSM(S, K, T_trade, T_cal, r, b, sigma=sigma_high)
        if price_mkt >= v_high.price(phi):
            return sigma_high

        v_low = VanillaBSM(S, K, T_trade, T_cal, r, b, sigma=sigma_low)
        if price_mkt <= v_low.price(phi):
            return sigma_low

        forward   = S * np.exp((b - r) * T_cal)
        discount  = np.exp(-r * T_cal)
        intrinsic = max(0.0, forward - K * discount)
        if price_mkt < intrinsic - epsilon:
            return np.nan
        if T_trade <= 0:
            return np.nan
        if price_mkt > S * np.exp((b - r) * T_cal):
            return np.nan

        # Consider the initial seed of sigma (using Brenner and Subrahmanyam method which is effective at ATM)
        sigma = np.sqrt(2 * np.pi / T_trade) * (price_mkt / (S * np.exp((b - r) * T_cal)))
        sigma = float(np.clip(sigma, sigma_low, sigma_high))

        low, high = sigma_low, sigma_high

        for _ in range(max_iter):
            v    = VanillaBSM(S, K, T_trade, T_cal, r, b, sigma)
            diff = v.price(phi) - price_mkt
            if abs(diff) < epsilon:
                return sigma
            vega = v.vega() * 100
            if abs(vega) > 1e-12:
                sigma_new = sigma - diff / vega
                sigma = np.clip(sigma_new, sigma_low, sigma_high)
            else:
                break       #Newton may fail for deep OTM options due to low Vega, so fall back to bisection.

        for _ in range(200):
            mid_s  = 0.5 * (low + high)
            diff_m = VanillaBSM(S, K, T_trade, T_cal, r, b, mid_s).price(phi) - price_mkt
            if abs(diff_m) < epsilon:
                return mid_s
            if diff_m < 0:
                low = mid_s
            else:
                high = mid_s

        return 0.5 * (low + high)

    def _load_and_solve(self):
        raw = pd.read_excel(self.filepath)
        raw.columns = [c.strip().strip('[]') for c in raw.columns]

        raw['quote_dt']  = pd.to_datetime(raw['QUOTE_READTIME'].str.strip())
        raw['expire_dt'] = pd.to_datetime(raw['EXPIRE_DATE'].str.strip())

        self.quote_dt = raw['quote_dt'].iloc[0]
        self.S        = float(raw['UNDERLYING_LAST'].iloc[0])

        start_dt_obj = parse_dt(self.quote_dt.strftime('%Y-%m-%d %H:%M:%S'))
        raw['expire_close'] = raw['expire_dt'] + pd.Timedelta(hours=16)
        raw['TTE_cal'] = (raw['expire_close'] - self.quote_dt).dt.total_seconds() / (365 * 86400)

        raw['mid']   = 0.5 * (raw['C_BID'] + raw['C_ASK'])
        raw['log_m'] = np.log(raw['STRIKE'] / self.S)
        raw['spread_pct'] = (raw['C_ASK'] - raw['C_BID']) / raw['mid'].clip(lower=1e-6)


        t_trade_map = {}
        for exp_close in raw['expire_close'].unique():
            end_dt_obj = parse_dt(exp_close.strftime('%Y-%m-%d %H:%M:%S'))
            seconds = count_trading_seconds_precise(start_dt_obj, end_dt_obj)
            t_trade_map[exp_close] = seconds / (trading_days_per_year * SECONDS_PER_FULL_TRADE_DAY)
        raw['TTE_trade'] = raw['expire_close'].map(t_trade_map)

        df = raw[raw['TTE_cal'] > 2 / 365].copy()
        df = df[df['mid'] >= self.min_price]
        df = df[df['C_VOLUME'] >= self.min_volume]
        df = df[df['spread_pct'] <= self.max_spread_pct]
        df = df[df['log_m'].between(*self.lm_range)]
        df = df[~((df['log_m'] > 0.3) & (df['TTE_trade'] < 0.05))]
        df = df.reset_index(drop=True)
        n_exp = df['expire_dt'].nunique()
        print(f'Solving IV for {len(df)} options across {n_exp} expiries...')


        t_cache: dict[pd.Timestamp, tuple[float, float]] = {}
        for exp in df['expire_dt'].unique():
            end_str     = exp.strftime('%Y-%m-%d') + ' 16:00:00'
            end_dt_obj  = parse_dt(end_str)
            T_cal       = max(0.0, (end_dt_obj - start_dt_obj).total_seconds() / (365 * 86400))
            trade_sec   = count_trading_seconds_precise(start_dt_obj, end_dt_obj)
            T_trade     = max(0.0, trade_sec / (trading_days_per_year * SECONDS_PER_FULL_TRADE_DAY))
            t_cache[exp] = (T_trade, T_cal)
        all_ivs = np.full(len(df), np.nan)
        for i, row in df.iterrows():
            T_trade, T_cal = t_cache[row['expire_dt']]
            all_ivs[i] = self._solve_iv_fast(price_mkt = float(row['mid']), S = self.S, K = float(row['STRIKE']), T_trade = T_trade,
                                             T_cal = T_cal, r = self.r, b = self.b,)

        df['iv'] = all_ivs


        sigma_high = 10.0
        df = df[df['iv'].notna()]
        df = df[df['iv'] > 0.01]
        df = df[df['iv'] < sigma_high ]

        delta_low, delta_high = self.delta_range
        df['delta'] = [
            _bsm_delta(self.S, float(row['STRIKE']),
                       t_cache[row['expire_dt']][0],t_cache[row['expire_dt']][1],
                       self.r, self.b, float(row['iv']))
            for _, row in df.iterrows()
        ]
        n_before = len(df)
        df = df[(df['delta'] >= delta_low) & (df['delta'] <= delta_high)]
        print(f'  Delta filter [{delta_low:.0%}, {delta_high:.0%}]: '
              f'dropped {n_before - len(df)} deep OTM/ITM rows')

        print(f'Kept {len(df)} options after IV filter  '
              f'({len(raw) - len(df)} dropped)')

        self.df       = df.reset_index(drop=True)
        self.expiries = sorted(df['expire_dt'].unique())
        self.TTEs     = np.array(sorted(df['TTE_trade'].unique()))


    def _build_rbf(self):
        """Build the surface as a thin-plate-spline RBF on (log_m, sqrtT). sqrtT is
        used because skew is proportional to 1/sqrtT and variance is proportional to T, which linearises the
        surface and stabilises interpolation/extrapolation."""
        X = np.column_stack([self.df['log_m'].values,
                             np.sqrt(self.df['TTE_trade'].values)])
        y = self.df['iv'].values
        self._rbf = RBFInterpolator(X, y, kernel='thin_plate_spline', smoothing=5e-4)

    def get_vol(self, K: float, TTE_trade: float) -> float:
        """IV at a single (strike, trade-time-to-expiry) from the RBF surface."""
        lm = np.log(K / self.S)
        return float(self._rbf(np.array([[lm, np.sqrt(TTE_trade)]])))

    def get_vol_vec(self, lm_arr: np.ndarray, TTE_trade: float) -> np.ndarray:
        """Vectorised RBF IV lookup across a log-moneyness array at fixed T."""
        X = np.column_stack([lm_arr, np.full_like(lm_arr, np.sqrt(TTE_trade))])
        return self._rbf(X)


    def get_slice(self, idx: int):
        exp = self.expiries[idx]
        sl  = self.df[self.df['expire_dt'] == exp].sort_values('STRIKE')
        T_cal   = float(sl['TTE_cal'].iloc[0])
        T_trade = float(sl['TTE_trade'].iloc[0])
        K   = sl['STRIKE'].values.astype(float)
        lm  = sl['log_m'].values.astype(float)
        iv  = sl['iv'].values.astype(float)
        mid = sl['mid'].values.astype(float)
        deltas = np.array([_bsm_delta(self.S, K, T_trade, T_cal, self.r, self.b, sigma)
                           for K, sigma in zip(K, iv)])
        return K, lm, iv, deltas, T_trade, exp.strftime('%Y-%m-%d'), mid, T_cal


    def term_structure(self):
        """ATM IV (nearest-to-money strike) vs trade-time-to-expiry."""
        t_trade, ivs, labels = [], [], []
        for i in range(len(self.expiries)):
            K, lm, iv, _, T_trade, lbl, _ , _ = self.get_slice(i)
            atm = np.argmin(np.abs(K - self.S))
            t_trade.append(T_trade); ivs.append(iv[atm]); labels.append(lbl)
        return np.array(t_trade), np.array(ivs), labels


    def surface_grid(self, n_lm: int = 70, n_T: int = 60):
        """Evaluate the RBF surface on a (log-moneyness × T) mesh for plotting."""
        lm_arr = np.linspace(*self.lm_range, n_lm)
        T_arr  = np.linspace(self.TTEs.min(), self.TTEs.max(), n_T)
        LM, TT = np.meshgrid(lm_arr, T_arr)
        X  = np.column_stack([LM.ravel(), np.sqrt(TT.ravel())])
        IV = np.clip(self._rbf(X).reshape(TT.shape), 0.01, None)
        return LM, TT, IV

    def sticky_smiles(self, expiry_idx: int, dS_pcts: list[float] | None = None) -> dict:
        """
        How the smile moves when spot goes S0 -> S1 = S0*(1+dS), under two desk
        conventions. x is log-moneyness relative to the NEW spot S1.
        """
        if dS_pcts is None:
            dS_pcts = [-0.10, -0.05, 0.05, 0.10]

        _, _, _, _, T, _, _, _ = self.get_slice(expiry_idx)

        x_dense = np.linspace(-0.50, 0.50, 300)
        iv_orig = self.get_vol_vec(x_dense, T)

        out = {'x': x_dense, 'iv_orig': iv_orig, 'S0': self.S, 'T': T}

        for dS in dS_pcts:
            ds = np.log(1 + dS)  # = ln(S1/S0)
            key = f'{dS:+.2f}'

            iv_ss = self.get_vol_vec(x_dense + ds, T)  # SS: old smile at x+ds
            iv_sm = self.get_vol_vec(x_dense, T)  # SM: unchanged in moneyness

            out[key] = {
                'S1': self.S * (1 + dS),
                'iv_ss': iv_ss,
                'iv_sm': iv_sm,
            }
        return out



    def plot_all(self, save_prefix: str | None = None):
        f1 = self._plot_term_structure()
        f2 = self._plot_smiles()
        f3 = self._plot_surface()
        f4 = self._plot_sticky()
        if save_prefix:
            for fig, tag in zip([f1, f2, f3, f4],
                                 ['term_structure', 'smiles', 'surface', 'sticky']):
                fig.savefig(f'{save_prefix}_{tag}.png', dpi=150, bbox_inches='tight')
        return f1, f2, f3, f4



    def _plot_term_structure(self):
        tts, ivs, labels = self.term_structure()
        td = tts * 252
        tw = ivs ** 2 * tts

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f'TSLA Implied Volatility — Term Structure\n'
            f'S = {self.S:.2f}  |  Quote: {self.quote_dt.strftime("%Y-%m-%d %H:%M")}'
            f'  |  IV solved from (bid+ask)/2 via Newton-Raphson / bisection',
            fontsize=12, fontweight='bold')

        # ATM IV
        ax = axes[0]
        ax.plot(td, ivs * 100, 'o-', color='steelblue', lw=2, ms=7, zorder=3)
        for x, y, lbl in zip(td, ivs * 100, labels):
            ax.annotate(lbl, (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=6.5, color='#555')
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('ATM Implied Vol (%)')
        ax.set_title('ATM IV  (nearest-to-money strike)')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.grid(True, alpha=0.3)

        # Total variance σ²·T
        ax2 = axes[1]
        ax2.plot(td, tw * 100, 's-', color='tomato', lw=2, ms=7, zorder=3)
        for x, y, lbl in zip(td, tw * 100, labels):
            ax2.annotate(lbl, (x, y), textcoords='offset points',
                         xytext=(4, 4), fontsize=6.5, color='#555')
        from numpy.polynomial import polynomial as P
        c      = P.polyfit(tts, tw, 1)
        t_fit  = np.linspace(tts.min(), tts.max(), 100)
        ax2.plot(t_fit * 252, P.polyval(t_fit, c) * 100, '--', color='gray',
                 lw=1.2, label='Linear fit')
        ax2.set_xlabel('Days to Expiry')
        ax2.set_ylabel('σ²·T  ×100')
        ax2.set_title('Total Variance  (no-arb: must be monotone & ~linear)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Forward implied vol per period
        ax3 = axes[2]
        if len(tts) >= 2:
            fwd_ivs, mids_d = [], []
            for i in range(len(tts) - 1):
                w1, w2 = ivs[i] ** 2 * tts[i], ivs[i + 1] ** 2 * tts[i + 1]
                dt     = tts[i + 1] - tts[i]
                fv     = (w2 - w1) / dt if dt > 1e-6 else np.nan
                if fv > 0:
                    fwd_ivs.append(np.sqrt(fv) * 100)
                    mids_d.append(0.5 * (td[i] + td[i + 1]))
            ax3.bar(mids_d, fwd_ivs, width=np.diff(td) * 0.7,
                    color='mediumpurple', alpha=0.8, edgecolor='white',
                    label='Forward IV per period')
        ax3.plot(td, ivs * 100, 'o-', color='steelblue', lw=1.5, ms=5,
                 label='ATM spot IV', zorder=3)
        ax3.set_xlabel('Days to Expiry')
        ax3.set_ylabel('Implied Vol (%)')
        ax3.set_title('Forward IV vs Spot ATM IV')
        ax3.legend(fontsize=9)
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # Figure 2: Smiles per expiry

    def _plot_smiles(self):
        n     = len(self.expiries)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        colors = cm.plasma(np.linspace(0.1, 0.9, n))

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
        fig.suptitle(
            f'TSLA Implied Volatility Smile per Expiry  |  S = {self.S:.2f}\n'
            'IV from mid prices  ·  colour = delta  ·  line = SVI fit',
            fontsize=12, fontweight='bold')

        axes_flat = axes.ravel() if nrows > 1 else np.array(axes)

        for i in range(n):
            ax = axes_flat[i]
            K, lm, iv, deltas, T_trade, lbl, mid, T_cal = self.get_slice(i)

            # scatter coloured by delta
            sc = ax.scatter(lm * 100, iv * 100,
                            c=deltas, cmap='coolwarm_r', vmin=0, vmax=1,
                            s=45, zorder=4, edgecolors='k', lw=0.3)

            # SVI fit overlay
            log_fwd_m = lm - (self.b - self.r) * T_cal
            if len(lm) >= 4:
                params = _fit_svi(log_fwd_m, iv, T_trade)
                if params is not None:
                    x_fit   = np.linspace(log_fwd_m.min(), log_fwd_m.max(), 200)
                    lm_fit  = x_fit + (self.b - self.r) * T_cal
                    w_fit   = _svi_w(x_fit, *params)
                    iv_fit  = np.sqrt(np.maximum(w_fit / T_trade, 0))
                    ax.plot(lm_fit * 100, iv_fit * 100, 'k-', lw=1.5,
                            alpha=0.85, label='SVI', zorder=5)

            ax.axvline(0, color='royalblue', lw=0.9, ls='--', alpha=0.7)

            # delta tick marks on top axis
            twin = ax.twiny()
            twin.set_xlim(ax.get_xlim())
            med_iv = float(np.median(iv))
            tick_lm = []
            for d in [0.10, 0.25, 0.50, 0.75, 0.90]:
                ki = _strike_from_delta(d, self.S, T_trade,T_cal, self.r, self.b, med_iv)
                if not np.isnan(ki):
                    tick_lm.append(np.log(ki / self.S) * 100)
            twin.set_xticks(tick_lm)
            twin.set_xticklabels(['10Δ', '25Δ', '50Δ', '75Δ', '90Δ'], fontsize=6.5)
            twin.set_xlabel('Approx. Δ', fontsize=7)

            ax.set_xlabel('Log-Moneyness ln(K/S) ×100', fontsize=8)
            ax.set_ylabel('IV (%)', fontsize=8)
            ax.set_title(f'{lbl}  T={T_trade*252:.0f}d', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.25)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            if len(lm) >= 4:
                ax.legend(fontsize=7)
            plt.colorbar(sc, ax=ax, label='Δ', fraction=0.04, pad=0.04)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        return fig

    # Figure 3: 3-D surface + heatmap + skew + convexity

    def _plot_surface(self):
        LM, TT, IV = self.surface_grid()

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f'TSLA Implied Volatility Surface  |  S = {self.S:.2f}  '
            f'Quote: {self.quote_dt.strftime("%Y-%m-%d")}',
            fontsize=13, fontweight='bold')

        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

        ax3d = fig.add_subplot(gs[0, :2], projection='3d')
        surf = ax3d.plot_surface(LM * 100, TT * 252, IV * 100,
                                  cmap='RdYlGn_r', alpha=0.88,
                                  linewidth=0, antialiased=True)
        ax3d.scatter(self.df['log_m'] * 100, self.df['TTE_trade'] * 252,
                     self.df['iv'] * 100, c='black', s=6, zorder=5, depthshade=True)
        ax3d.set_xlabel('Log-Moneyness (%)', labelpad=8)
        ax3d.set_ylabel('Days to Expiry', labelpad=8)
        ax3d.set_zlabel('IV (%)', labelpad=8)
        ax3d.set_title('3-D Implied Volatility Surface', fontsize=11)
        ax3d.view_init(elev=25, azim=-55)
        cb = fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)
        cb.set_label('IV (%)')

        # Heatmap
        ax2d = fig.add_subplot(gs[0, 2])
        cont = ax2d.contourf(LM[0] * 100, TT[:, 0] * 252, IV * 100,
                             levels=30, cmap='RdYlGn_r')
        ax2d.contour(LM[0] * 100, TT[:, 0] * 252, IV * 100,
                     levels=10, colors='white', linewidths=0.4, alpha=0.5)
        ax2d.scatter(self.df['log_m'] * 100, self.df['TTE_trade'] * 252,
                     c='white', s=8, alpha=0.6, edgecolors='k', lw=0.3)
        ax2d.axvline(0, color='black', lw=1, ls='--')
        ax2d.set_xlabel('Log-Moneyness (%)')
        ax2d.set_ylabel('Days to Expiry')
        ax2d.set_title('IV Heatmap', fontsize=11)
        fig.colorbar(cont, ax=ax2d).set_label('IV (%)')

        # 25delta Skew & Butterfly per expiry
        ax_sk = fig.add_subplot(gs[1, 0])
        ax_cv = fig.add_subplot(gs[1, 1])
        skews, convex, days = [], [], []
        for i in range(len(self.expiries)):
            K, lm, iv, deltas, T_trade, lbl, _, _ = self.get_slice(i)
            if len(deltas) < 5:
                continue
            i25 = np.argmin(np.abs(deltas - 0.25))
            i50 = np.argmin(np.abs(deltas - 0.50))
            i75 = np.argmin(np.abs(deltas - 0.75))
            skews.append((iv[i25] - iv[i75]) * 100)
            convex.append((0.5 * (iv[i25] + iv[i75]) - iv[i50]) * 100)
            days.append(T_trade * 252)

        widths = np.array(days) * 0.12
        ax_sk.bar(days, skews,  width=widths, color='mediumpurple', alpha=0.8, edgecolor='white')
        ax_sk.axhline(0, color='gray', lw=0.8, ls='--')
        ax_sk.set_xlabel('Days to Expiry')
        ax_sk.set_ylabel('IV_25Δ − IV_75Δ (%)')
        ax_sk.set_title('Skew Term Structure\n(25Δ call − 75Δ call)')
        ax_sk.grid(True, alpha=0.3, axis='y')

        ax_cv.bar(days, convex, width=widths, color='darkorange', alpha=0.8, edgecolor='white')
        ax_cv.axhline(0, color='gray', lw=0.8, ls='--')
        ax_cv.set_xlabel('Days to Expiry')
        ax_cv.set_ylabel('½(IV_25Δ + IV_75Δ) − IV_ATM (%)')
        ax_cv.set_title('Smile Convexity (25Δ Butterfly proxy)')
        ax_cv.grid(True, alpha=0.3, axis='y')

        # Smile dispersion (vol-of-vol proxy)
        ax_d = fig.add_subplot(gs[1, 2])
        disp, days_d = [], []
        for i in range(len(self.expiries)):
            K, lm, iv, deltas, T_trade, lbl, _ ,_= self.get_slice(i)
            mask = (deltas >= 0.20) & (deltas <= 0.80)
            if mask.sum() >= 3:
                disp.append(float(np.std(iv[mask]) * 100))
                days_d.append(T_trade * 252)
        ax_d.plot(days_d, disp, 'D-', color='teal', lw=2, ms=6)
        ax_d.set_xlabel('Days to Expiry')
        ax_d.set_ylabel('σ(IV | 20Δ–80Δ)  %')
        ax_d.set_title('Smile Dispersion\n(vol-of-vol proxy)')
        ax_d.grid(True, alpha=0.3)

        return fig

    # Figure 4: Sticky Rules

    def _plot_sticky(self, expiry_idx: int = 3,
                     dS_pcts: tuple = (-0.10, -0.05, 0.05, 0.10)):
        _, _, _, _, T, lbl, _, _ = self.get_slice(expiry_idx)
        res = self.sticky_smiles(expiry_idx, list(dS_pcts))

        neg_c = ['#d62728', '#ff7f0e']
        pos_c = ['#2ca02c', '#1f77b4']
        clrs = neg_c + pos_c
        sdS = sorted(dS_pcts)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(
            f'TSLA Sticky Rules  -  Expiry: {lbl}  (T = {T * 252:.0f}d)\n'
            r'X-axis: log-moneyness relative to new spot $S_1$',
            fontsize=13, fontweight='bold')

        def _draw_panel(ax, rule_key, title, note):
            ax.plot(res['x'] * 100, res['iv_orig'] * 100,
                    'k-', lw=2.5, label=f'Current  S0={self.S:.1f}', zorder=6)
            ax.axvline(0, color='gray', lw=1, ls='--', alpha=0.5)
            for c, dS in zip(clrs, sdS):
                d = res[f'{dS:+.2f}']
                ls = '--' if dS < 0 else '-.'
                ax.plot(res['x'] * 100, d[f'iv_{rule_key}'] * 100,
                        ls, color=c, lw=1.8, alpha=0.85,
                        label=f"S1={d['S1']:.1f} ({dS * 100:+.0f}%)")
            ax.set_xlabel('Log-Moneyness ln(K/S1) x100')
            ax.set_ylabel('IV (%)')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.28)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=7.5,
                    va='bottom', color='#333',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))

        _draw_panel(
            axes[0], 'ss',
            r'A. Sticky Strike  -  $\sigma(K,T)$ constant',
            'Each strike keeps its vol; smile shifts by ln(S1/S0) in moneyness.\n'
            'ATM vol drops when spot rises -> negative vol-spot correlation.\n'
            'Hedging: Delta_eff = Delta_BSM (no skew correction).')

        _draw_panel(
            axes[1], 'sm',
            r'B. Sticky Moneyness  -  $\sigma(K/S,T)$ constant',
            'Smile fixed in spot-moneyness; no lateral shift.\n'
            'ATM vol unchanged as spot moves.\n'
            'Hedging: Delta_eff = Delta_BSM - vega*skew/S.')

        # Panel C: overlay both rules for +10% shock
        ax = axes[2]
        dS_ = 0.10
        d_ = res[f'{dS_:+.2f}']
        S1_ = d_['S1']
        ax.plot(res['x'] * 100, res['iv_orig'] * 100,
                'k-', lw=2.5, label=f'Current  S0={self.S:.1f}', zorder=6)
        ax.plot(res['x'] * 100, d_['iv_ss'] * 100,
                '-', color='#e31a1c', lw=2.2, label='Sticky Strike')
        ax.plot(res['x'] * 100, d_['iv_sm'] * 100,
                '-', color='#1f78b4', lw=2.2, label='Sticky Moneyness')
        ax.axvline(0, color='gray', lw=1, ls='--', alpha=0.5)

        iv_ss0 = float(self.get_vol_vec(np.array([np.log(S1_ / self.S)]), T)[0])
        iv_sm0 = float(self.get_vol_vec(np.array([0.0]), T)[0])
        ax.annotate(
            f'ATM under SS: {iv_ss0 * 100:.1f}%\nATM under SM: {iv_sm0 * 100:.1f}%',
            xy=(0, iv_sm0 * 100), xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=8.5,
            bbox=dict(boxstyle='round', fc='white', alpha=0.9))

        ax.set_xlabel('Log-Moneyness ln(K/S1) x100')
        ax.set_ylabel('IV (%)')
        ax.set_title(f'C. Both Rules  -  +{dS_ * 100:.0f}% spot shock  (S1={S1_:.1f})',
                     fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.28)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.text(
            0.02, 0.03,
            'Hedging delta correction:\n'
            '  SS:  Delta_eff = Delta_BSM\n'
            '  SM:  Delta_eff = Delta_BSM - vega*skew/S',
            transform=ax.transAxes, fontsize=8, va='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.85))

        plt.tight_layout()
        return fig


def main():
    filepath = 'C:/Users/windows10/Documents/GitHub/Option-pricing-analysis/volatility/tsla_option.xlsx'
    print('Constructing TSLA vol surface...\n')
    vs = VolSurface(filepath, r=0.04, b=0.04, min_volume=50, min_price=0.50,
                    max_spread_pct=0.25, delta_range=(0.05, 0.95))

    tts, ivs, labels = vs.term_structure()
    print('\nATM IV term structure (from solver):')
    for t, iv, lbl in zip(tts, ivs, labels):
        print(f'  {lbl}  T={t*252:5.0f}d  ATM IV = {iv*100:.2f}%')

    print('\nGenerating plots...')
    vs.plot_all()
    print('Done.')
    return vs


if __name__ == '__main__':
    main()
    plt.show()
