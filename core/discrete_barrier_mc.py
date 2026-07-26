"""
Monte-Carlo pricer for discretely-monitored single-barrier options.

For each observation interval, carry accrues in CALENDAR time and diffusion in
TRADING time.

Hence E[S_T] = S0 * exp(b * T_cal), while discounting is exp(-r * T_cal).
Antithetic variates are used for variance reduction.

Rebate convention: rebate-at-hit. A knocked-out path receives K at the first
breached observation and discounts it from THAT observation's calendar time
(not from maturity). Knock-in rebate (paid at expiry if never touched) is
discounted from maturity, matching Haug's term_E.
"""
import numpy as np
from core.time_utils import (parse_dt, count_trading_seconds_precise,
                             generate_trading_day_obs, SECONDS_PER_FULL_TRADE_DAY,
                             trading_days_per_year)

CAL_SECONDS_PER_YEAR = 365 * 24 * 3600  

def simulate_paths_discrete(S0, b, sigma, dt_trade_steps, dt_cal_steps, n_paths, rng):
    """Simulate GBM in trading time at the observation checkpoints.
    Returns S of shape [n_paths, n_steps+1]; column 0 is S0, columns 1. are
    the prices at each observation date. Uses antithetic variates."""
    assert n_paths % 2 == 0, "n_paths must be even (antithetic variates)"
    half = n_paths // 2
    dt_trade = np.asarray(dt_trade_steps, dtype=float)
    dt_cal = np.asarray(dt_cal_steps, dtype=float)
    n_steps = len(dt_trade)
    eps_half = rng.standard_normal((half, n_steps))
    eps = np.vstack([eps_half, -eps_half])
    drift = b * dt_cal - 0.5 * sigma ** 2 * dt_trade
    diffusion = sigma * eps * np.sqrt(dt_trade)
    cumulative_log_return = np.cumsum(drift + diffusion, axis=1)
    log_paths = np.column_stack([np.zeros(paths), cumulative_log_return])
    return S0 * np.exp(log_paths)


class DiscreteBarrierMC:
    """
    PARAMETERS:

    start_dt, end_dt : inception / maturity timestamps (strings).
    S, X, H          : spot, strike, contract barrier.
    r, b, sigma      : risk-free rate, cost of carry, annualized vol .
    K                : cash rebate (rebate-at-hit for knock-out; at-expiry-if-no-touch for knock-in).
    n_paths          : number of MC paths (even; antithetic).
    seed             : RNG seed.
    """

    def __init__(self, start_dt, end_dt, S, X, H, r, b, sigma, K=0.0,
                 trading_days_per_year=trading_days_per_year, n_paths=200000, seed=42):
        self.start = parse_dt(start_dt); self.end = parse_dt(end_dt)
        self.S, self.X, self.H = float(S), float(X), float(H)
        self.r, self.b, self.sigma, self.K = r, b, sigma, float(K)
        self.ann = trading_days_per_year
        self.n_paths = int(n_paths); self.seed = seed

        self.T_cal   = (self.end - self.start).total_seconds() / CAL_SECONDS_PER_YEAR
        self.T_trade = count_trading_seconds_precise(self.start, self.end) / (self.ann * SECONDS_PER_FULL_TRADE_DAY)

        obs_strs = generate_trading_day_obs(start_dt, end_dt)
        self.obs_dts = [parse_dt(s) for s in obs_strs]
        self.n_obs = len(self.obs_dts)

        checkpoints = [self.start] + self.obs_dts
        self.dt_trade_steps = [count_trading_seconds_precise(a, c) / (self.ann * SECONDS_PER_FULL_TRADE_DAY)
                         for a, c in zip(checkpoints[:-1], checkpoints[1:])]

        self.dt_cal_steps = [(c - a).total_seconds() / CAL_SECONDS_PER_YEAR
                             for a, c in zip(checkpoints[:-1], checkpoints[1:])]

        self.t_cal_obs = np.array([(o - self.start).total_seconds() / CAL_SECONDS_PER_YEAR
                                   for o in self.obs_dts])


    def _simulate(self):
        rng = np.random.default_rng(self.seed)
        return simulate_paths_discrete(self.S, self.b, self.sigma, self.dt_trade_steps, self.dt_cal_steps,
                                       self.n_paths, rng)

    def price_with_se(self, barrier_type):
        """Return (price, standard_error). SE is computed on antithetic pair
        means, which is the correct (uninflated) estimator for antithetics."""
        bt = barrier_type.lower()
        is_call = bt.startswith('c'); is_upper = 'u' in bt
        is_out = 'o' in bt; is_in = 'i' in bt
        disc_mat = np.exp(-self.r * self.T_cal)

        hit0 = (is_upper and self.S >= self.H) or (not is_upper and self.S <= self.H)
        if is_out and hit0:
            return self.K, 0.0

        S_all = self._simulate()
        obs_prices = S_all[:, 1:]          # [n_paths, n_obs]
        ST = S_all[:, -1]
        vanilla = np.maximum(ST - self.X, 0.0) if is_call else np.maximum(self.X - ST, 0.0)

        breach = obs_prices >= self.H if is_upper else obs_prices <= self.H
        knocked = np.any(breach, axis=1)

        if is_in and hit0:
            pv = vanilla * disc_mat        # already alive -> vanilla
        elif is_out:
            first_idx = np.argmax(breach, axis=1)             # first breached obs
            disc_knock = np.exp(-self.r * self.t_cal_obs[first_idx])
            pv = np.where(knocked, self.K * disc_knock, vanilla * disc_mat)
        else:  # knock-in
            # touched -> vanilla (disc from maturity); never touched -> K no-touch rebate (disc from maturity)
            pv = np.where(knocked, vanilla * disc_mat, self.K * disc_mat)

        price = float(np.mean(pv))
        half = self.n_paths // 2
        pair_means = 0.5 * (pv[:half] + pv[half:])
        se = float(np.std(pair_means, ddof=1) / np.sqrt(half))
        return price, se

    def price(self, barrier_type, show_detail=False):
        p, se = self.price_with_se(barrier_type)
        if show_detail:
            self._print_detail(barrier_type, p, se)
        return p

    def _print_detail(self, bt, price, se):
        print("=" * 56)
        print(f"  Discrete-barrier MC  [{bt.upper()}]")
        print("=" * 56)
        for k, v in self.time_info.items():
            print(f"  {k:<16}: {v}")
        print(f"  {'paths':<16}: {self.n_paths}")
        print("-" * 56)
        print(f"  ->  MC price: {price:.6f}   (SE {se:.6f}, 95% +/-{1.96*se:.6f})")
        print("=" * 56)

    @property
    def time_info(self):
        return {"start": str(self.start), "end": str(self.end),
                "T_cal(yr)": round(self.T_cal, 6), "T_trade(yr)": round(self.T_trade, 6),
                "n_obs": self.n_obs}

