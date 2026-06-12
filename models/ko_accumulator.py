"""
Knock-Out Accumulator pricer.

Unlike the normal accumulator (priced via Carr-Madan static replication of a piecewise-linear payoff),
the knock-out variant adds a path-dependent feature:
once close-spot touches the boundary B on any observation datetime, the contract terminates and pays a fixed daily rebate for the remaining days.

Decomposition of the daily payoff:

For a call KO accumulator, if the option has not knocked out (the close-spot has not touched B)
Its daily payoff is the same piecewise-linear shape as the normal accumulator:

    +PR * (S_t - K)   if k <= S_t < B    (Gain leg)
    -L  * (K - S_t)   if S_t <  K     (leveraged loss leg)

But each leg now lives inside a knock-out structure:

    Gain leg : the buyer long a call knocked out at the up-barrier B  -> CUO (call up-out), carrying the daily rebate when knocked out
    Loss leg : the buyer short a put knocked out at the up-barrier B  -> PUO (put up-out), with no rebate

So the per-day value is:

    daily = PR * CUO(strike=K, barrier=B, rebate)  -  L * PUO(strike=K, barrier=B, 0)

The total contract value sums these daily contributions across all observation dates

For a put KO accumulator the barrier is below the strike, and the legs become PDO (long, with rebate) and CDO (short, no rebate).

Greeks are computed by finite difference, since the closed form of greeks of the barrier are very complicated.

A separate Monte Carlo pricer (KnockOutAccumulatorMC) simulates the full path-dependent payoff directly and is used for cross-validation.
"""
from datetime import  time
import numpy as np
from core.time_utils import parse_dt, count_trading_seconds_precise, generate_trading_day_obs, SECONDS_PER_FULL_TRADE_DAY, trading_days_per_year
from core.discrete_barrier import HaugBarrierDualTime

class KnockOutAccumulatorPricer:
    """
    Each daily observation is decomposed into long PR weights of CUO (up-and-out call), and short L weights of PUO (up-and-out put)

    PARAMETERS:

    S: Spot price at inception.
    K: Accumulator strike.
    B: Knock-out barrier. For a call accumulator B > K (upper barrier); for a put accumulator B < K (lower barrier).
    r, b, sigma: Risk-free rate, cost of carry, annualized volatility.
    L: Leverage multiplier on the loss leg. Must be >= PR.
    PR: Participation rate between strike and barrier.
    rebate: Cash paid per remaining day once knocked out
    option_type: {'call', 'put'}
    start_dt, end_dt: Inception and maturity timestamps.
    beta: BGK continuity-correction constant., default 0.5826
    """
    BETA_DEFAULT = 0.5826
    def __init__(self, S: float, K: float, B: float, r: float, b: float, sigma: float, PR: float, L: float, rebate: float, option_type: str,
                 start_dt: str, end_dt: str, beta: float = BETA_DEFAULT,):

        self.S, self.K, self.B, self.r, self.b, self.sigma = S, K, B, r, b, sigma
        self.PR, self.L, self.rebate = PR, L, rebate
        self.option_type = option_type.lower()
        self.start, self.end = parse_dt(start_dt), parse_dt(end_dt)
        self.start_str, self.end_str = start_dt, end_dt
        self.ann   = trading_days_per_year
        self.beta  = beta
        self.dt    = 1.0 / trading_days_per_year

        self.observation_dates = generate_trading_day_obs(start_dt, end_dt, obs_time=time(15, 0))

        self._obs_info: list[dict] = []
        for obs_dt_str in self.observation_dates:
            obs_dt = parse_dt(obs_dt_str)
            cal_sec = (obs_dt - self.start).total_seconds()
            T_cal_i = cal_sec / (365 * 86400)
            trade_sec = count_trading_seconds_precise(self.start, obs_dt)
            T_trade_i = trade_sec / (self.ann * SECONDS_PER_FULL_TRADE_DAY)
            self._obs_info.append({
                'dt_str': obs_dt_str,
                'T_cal': T_cal_i,
                'T_trade': T_trade_i,
            })

    def _one_barrier_price(self, info: dict, barrier_type: str, rebate_val: float, S:float, sigma:float, t_shift_trade: float = 0.0, t_shift_cal: float = 0.0) -> float:
        """
        Price a single knock-out barrier option for one observation date.

        Applies the BGK barrier adjustment, checks whether spot has already reached the adjusted barrier (in which case only the rebate remains),
        and otherwise delegates to the Haug closed-form pricer.

        The `t_shift` argument shifts maturity backward in trading time; it is used by greeks() to compute theta via a forward time bump.
        """
        T_trade_raw = info['T_trade']
        T_cal_raw = info['T_cal']
        T_trade = T_trade_raw - t_shift_trade
        T_cal = T_cal_raw - t_shift_cal
        if T_trade <= 0.0 or T_cal <= 0.0:
            return 0.0
        T_cal = T_cal_raw * (T_trade / T_trade_raw)    # Scale calendar time by the same fraction the trading time shrank.
        is_upper = ('u' in barrier_type)
        H_adj = self.B * np.exp((+1 if is_upper else -1) * self.beta * sigma * np.sqrt(self.dt))
        hit = (is_upper and S >= H_adj) or (not is_upper and S <= H_adj)
        if hit:
            return rebate_val

        barrier_price = HaugBarrierDualTime(S=S, X=self.K, H=H_adj,T_cal=T_cal, T_trade=T_trade,r=self.r, b=self.b, sigma=sigma,K=rebate_val, )

        return barrier_price.price(barrier_type)

    def _total_price(self, S:float, sigma:float, t_shift_trade: float = 0.0, t_shift_cal: float = 0.0, obs_info: list[dict] | None = None,) -> float:

        if obs_info is None:
            obs_info = self._obs_info
        bt_long = 'cuo' if self.option_type == 'call' else 'pdo'
        bt_short = 'puo' if self.option_type == 'call' else 'cdo'
        total = 0.0
        for info in obs_info:
            p_long = self.PR * self._one_barrier_price(info, bt_long, self.rebate, S, sigma, t_shift_trade, t_shift_cal)
            p_short = self.L * self._one_barrier_price(info, bt_short, 0.0, S, sigma, t_shift_trade, t_shift_cal)
            total += p_long - p_short
        return total

    def price(self) -> float:
        return self._total_price(self.S, self.sigma)


    def greeks(self, dS_frac: float = 1e-3, d_sigma: float = 1e-4, theta_seconds:float = 60) -> dict:
        """
        Finite-difference Greeks.

        Delta, Gamma : central difference in spot, bump = S * dS_frac
        Vega         : central difference in vol, reported per 1% vol move
        Theta        : dual-time forward difference.

        Bump sizes default to 1e-3 and 1e-4; see analysis/bump_size.py for a study of bump-size stability, especially near the barrier where the
        knock-out discontinuity makes finite differences delicate.
        """
        p0  = self.price()
        h_S = self.S * dS_frac
        p_up   = self._total_price(self.S + h_S, self.sigma)
        p_down = self._total_price(self.S - h_S, self.sigma)
        delta  = (p_up - p_down) / (2 * h_S)
        gamma  = (p_up - 2 * p0 + p_down) / (h_S ** 2)

        p_vu   = self._total_price(self.S, self.sigma + d_sigma)
        p_vd   = self._total_price(self.S, self.sigma - d_sigma)
        vega   = (p_vu - p_vd) / (2 * d_sigma) / 100

        h_trade = theta_seconds / (self.ann * SECONDS_PER_FULL_TRADE_DAY)
        h_cal   = h_trade * (self.ann / 365.0)
        p_ht  = self._total_price(self.S, self.sigma, t_shift_trade=h_trade, t_shift_cal=h_cal)
        theta = (p_ht - p0) / h_trade / self.ann

        return {
            'delta': round(delta, 8),
            'gamma': round(gamma, 8),
            'vega':  round(vega,  8),
            'theta': round(theta, 8),
        }

    def obs_breakdown(self) -> list[dict]:
        bt_long = 'cuo' if self.option_type == 'call' else 'pdo'
        bt_short = 'puo' if self.option_type == 'call' else 'cdo'
        rows = []
        for info in self._obs_info:
            p_long = self.PR * self._one_barrier_price(info, bt_long, self.rebate, self.S, self.sigma)
            p_short = self.L * self._one_barrier_price(info, bt_short, 0.0, self.S, self.sigma)
            rows.append({
                'date': str(info['dt_str'])[:10],
                'T_trade': round(info['T_trade'], 5),
                'long': round(p_long, 6),
                'short': round(p_short, 6),
                'contrib': round(p_long - p_short, 6),
            })
        return rows

    def summary(self, max_rows: int = 5, show_greeks: bool = True):
        px = self.price()
        rows = self.obs_breakdown()
        n = len(rows)

        print("\n===== Knock out Accumulator Summary =====")
        print(f"  Price : {px:.6f}")
        if show_greeks:
            gk = self.greeks()
            print(f"  Delta : {gk['delta']:+.8f}")
            print(f"  Gamma : {gk['gamma']:+.8f}")
            print(f"  Vega  : {gk['vega']:+.8f}  (per 1% vol)")
            print(f"  Theta : {gk['theta']:+.8f}  (per trading day)")
        print("\n--- Daily Contributions ---")
        if max_rows > 0 and n > 0:
            ll = 'CUO×PR' if self.option_type == 'call' else 'PDO×PR'
            sl = 'PUO×L' if self.option_type == 'call' else 'CDO×L'
            print(f"  {'Date':^12}  {'T_trade':>8}  {ll:>10}  {sl:>10}  {'Net':>10}")
            print(f"{'-' * 62}")
            for row in rows[:max_rows]:
                print(f"  {row['date']:^12} | {row['T_trade']:>8.5f}|"
                      f"  {row['long']:>10.4f} | {row['short']:>10.4f} | {row['contrib']:>10.4f}")
            if n > max_rows:
                print(f"  ... （ {n} days total）")


class KnockOutAccumulatorMC:
    """
    Monte Carlo pricer for the knock-out accumulator, used to cross-validate KnockOutAccumulatorPricer.

    Simulates GBM paths under the trading-time clock with antithetic variates.
    For each path, finds the first observation date (if any) where close-spot touches the barrier B.
    Before knock-out, the path remains as the normal piecewise-linear daily payoff. On knock-out, the path will terminate
    and instead receives the daily rebate for all remaining days.

    Note on barrier monitoring: the MC checks knock-out only at observation datetime (discrete monitoring), consistent with the contract terms.
    The analytic pricer applies the BGK adjustment to bridge discrete and continuous;
    convergence between the two is the validation check (analysis/mc_convergence.py).

    PARAMETERS:

    n_paths: Number of simulated paths
    seed: RNG seed for reproducibility.
    All other parameters as in AccumulatorReplication.
    """
    def __init__(self, S: float, K: float, B: float, r: float, b: float, sigma: float, PR: float, L: float, rebate: float,option_type: str,
                 start_dt: str, end_dt: str,n_paths: int = 20000,seed: int = 42, ):
        self.option_type = option_type.lower()
        self.start = parse_dt(start_dt)
        self.observation_dates = generate_trading_day_obs(start_dt, end_dt, obs_time=time(15, 0))
        self.obs_dts = [parse_dt(d) for d in self.observation_dates]
        self.S, self.K, self.B = S, K, B
        self.sigma, self.r, self.b = sigma, r, b
        self.L, self.PR, self.rebate = L, PR, rebate
        self.ann = trading_days_per_year
        self.n_paths = n_paths
        self.seed = seed
        self.n_obs = len(self.obs_dts)

        checkpoints = [self.start] + self.obs_dts
        self.dt_trade = []
        self.dt_cal = []
        self.T_cal_obs = []

        for i in range(self.n_obs):
            trade_sec = count_trading_seconds_precise(checkpoints[i], checkpoints[i + 1])
            self.dt_trade.append(trade_sec / (self.ann * SECONDS_PER_FULL_TRADE_DAY))
            step_cal_sec = (checkpoints[i + 1] - checkpoints[i]).total_seconds()
            self.dt_cal.append(step_cal_sec / (365 * 86400))
            cal_sec = (self.obs_dts[i] - self.start).total_seconds()
            self.T_cal_obs.append(cal_sec / (365 * 86400))

        self.dt_trade = np.array(self.dt_trade)
        self.dt_cal = np.array(self.dt_cal)
        self.T_cal_obs = np.array(self.T_cal_obs)
        self.df_obs = np.exp(-self.r * self.T_cal_obs)

    def simulate_gbm_path(self, rng: np.random.Generator) -> np.ndarray:
        half_paths = self.n_paths // 2
        epsilon_half = rng.standard_normal((half_paths, self.n_obs))
        epsilon = np.vstack([epsilon_half, - epsilon_half])
        drift = self.b * self.dt_cal - 0.5 * self.sigma ** 2 * self.dt_trade
        diffusion = self.sigma * epsilon * np.sqrt(self.dt_trade)
        cumulative_log_return = np.cumsum(drift + diffusion, axis=1)
        S_obs = self.S * np.exp(cumulative_log_return)
        return S_obs

    def pay_off(self, S_obs: np.ndarray) -> np.ndarray:
        n_paths, n_obs = S_obs.shape
        knockout = S_obs >= self.B if self.option_type == 'call' else S_obs <= self.B
        knockout_cum = np.cumsum(knockout, axis=1) > 0  # Cumulative knock-out flag: False before first touch, True after.
        knockout_ever = knockout.any(axis=1)            # Whether the path has been knocked out
        knockout_day = np.where(knockout_ever, knockout.argmax(axis=1), n_obs)  # First knock-out observation index per path (n_obs if never).

        if self.option_type == 'call':
            normal_payoff = np.where(S_obs >= self.K, self.PR * (S_obs - self.K),
                              -self.L * (self.K - S_obs))
        else:
            normal_payoff = np.where(S_obs <= self.K, self.PR * (self.K - S_obs),
                              -self.L * (S_obs - self.K))

        normal_payoff_days = ~knockout_cum
        pv_total_normal_payoff = (normal_payoff * normal_payoff_days * self.df_obs).sum(axis=1)

        safe_idx = knockout_day.clip(0, n_obs - 1)
        remaining_days = n_obs - knockout_day
        rebate_df = self.df_obs[safe_idx]
        pv_rebate = np.where(knockout_ever, self.rebate * remaining_days * rebate_df, 0.0)

        return pv_total_normal_payoff + pv_rebate

    def price(self) -> float:
        rng   = np.random.default_rng(self.seed)
        S_obs = self.simulate_gbm_path(rng)
        return float(self.pay_off(S_obs).mean())

    def price_with_ci(self) -> tuple[float, float, float]:
        rng   = np.random.default_rng(self.seed)
        S_obs = self.simulate_gbm_path(rng)
        pv    = self.pay_off(S_obs)
        mean  = float(pv.mean())
        se    = float(pv.std(ddof=1) / np.sqrt(self.n_paths))
        return mean, mean - 1.96 * se, mean + 1.96 * se

    def summary(self):
        mean, lo, hi = self.price_with_ci()
        print(f"\n===== Knockout Accumulator MC Summary =====")
        print(f"Type    : {self.option_type.upper()}")
        print(f"Paths   : {self.n_paths:,}  (antithetic)")
        print(f"Price   : {mean:.6f}")
        print(f"95% CI  : [{lo:.6f},  {hi:.6f}]")
        print("=====================================")

if __name__ == '__main__':
    params = dict(
        S=3362, K=3315, B=3409,
        r=0.03, b=0.0, sigma=0.09,
        L=2.0, PR=1.0, rebate = 10.0,
        strike_shift=0.002,
        option_type='call',
        start_dt="2026.04.20 21:17:01",
        end_dt="2026.06.18 15:00:00",
        seed=42,
    )

    pricer = KnockOutAccumulatorPricer(S=params['S'], K=params['K'], B=params['B'],
        r=params['r'], b=params['b'], sigma=params['sigma'],
        PR=params['PR'], L=params['L'], rebate=params['rebate'],
        option_type=params['option_type'],
        start_dt=params['start_dt'], end_dt=params['end_dt'],)


    pricer.summary(max_rows=10)