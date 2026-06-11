"""
Normal accumulator option pricer via static replication.

The Normal accumulator is a path-of-daily-payoffs structure:
on each trading day from start_dt to end_dt, the buyer receives a piecewise-linear payoff in spot S_t:

    Call accumulator daily payoff (per unit):
        +PR * (S_t - K)        if K <= S_t < B          (gain leg)
         0                     if S_t >= B              (boundary cap)
        -L  * (K - S_t)        if S_t < K               (leveraged loss leg)

    Put accumulator is the mirror image (sign-flipped, with B below K).

PR is the participation rate (typically 1.0), L is the leverage on losses (typically 2.0 or higher),
and B is the upper boundary that caps each day's upside.

Pricing approach:
Each daily payoff is a piecewise-linear function of S_t with three kinks.
By  Carr-Madan (1998), any European payoff f(S_T) admits a static replication

    f(S_T) = f(K) + f'(K)(S_T - K)
            + integral_0^K f''(z) (z - S_T)+ dz       (put leg)
            + integral_K^inf f''(z) (S_T - z)+ dz     (call leg)

For the call accumulator, replicating with puts:
    K1 = K           (downside leverage kink)
    K2 = B           (boundary kink)
    K3 = B*(1+eps)   (boundary kink, shifted by `strike_shift` eps)

Weights are derived by matching the three slope-jumps at K1, K2, K3:
    w1 = PR * (K3-K2)/(K2-K1)    # slope between K2 and K3
    w2 = w1 + PR                 # slope between K2 and K3
    w3 = L - PR                  # slope between K1 and K2

See docs/accumulator_replication.md for the full derivation and a worked numerical example.

The MC pricer (AccumulatorMC) below is an independent path-simulation implementation used to cross-validate the replication pricer.
"""
from datetime import  time
import numpy as np
from core.time_utils import parse_dt, count_trading_seconds_precise, generate_trading_day_obs, SECONDS_PER_FULL_TRADE_DAY, trading_days_per_year
from core.vanilla import VanillaBSM


class AccumulatorReplication:
    """
    Each daily piecewise-linear payoff is decomposed into a portfolio of three vanilla puts (for call accumulator) or three vanilla calls
    (for put accumulator) via Carr-Madan. The total contract value is the sum of these three-vanilla portfolios across all observation dates.

    PARAMETERS:

    S: Spot price at inception.
    K: Strike of the accumulator payoff.
    B: Upper boundary (call) / lower boundary (put). Each day's payoff is zero when close-spot crosses this level (a soft cap,
       not a knock-out for the normal accumulator — see ko_accumulator.py for KO variant).
    r, b, sigma: Risk-free rate, cost of carry, annualized volatility.
    L: Leverage multiplier on the loss leg. Must be >= PR.
    PR: Participation rate between strike and barrier.
    strike_shift: The parameter that controls the smoothness of the payoff at cliff (S = B)
    option_type : {'call', 'put'}
    start_dt, end_dt: Inception and maturity timestamps (parsed by parse_dt).
    """
    def __init__(self, S: float, K: float, B: float, r: float, b: float, sigma: float, L: float, PR: float, strike_shift: float,
                 option_type: str, start_dt: str, end_dt: str):
        if strike_shift <= 0:
            raise ValueError('strike_shift must be > 0')
        if L < PR:
            raise ValueError('L must be larger than PR')
        self.S, self.K, self.B, self.r, self.b, self.sigma, self.L, self.PR, self.strike_shift = S, K, B, r, b, sigma, L, PR, strike_shift
        self.option_type = option_type.lower()
        self.start, self.end = parse_dt(start_dt), parse_dt(end_dt)
        self.observation_dates = generate_trading_day_obs(start_dt, end_dt, obs_time=time(15,0))
        self.obs_dts = [parse_dt(d) for d in self.observation_dates]
        self.ann = trading_days_per_year

        if self.option_type == 'call':
            self.K1 = K
            self.K2 = B
            self.K3 = B * (1.0 + strike_shift)
            self.cp_phi = -1
            self.ratio = (self.K2 - self.K1) / (self.K3 - self.K2)
            self.w1 = PR * self.ratio
            self.w2 = PR * self.ratio + PR
            self.w3 = L - PR
        else:
            self.K1 = B * (1.0 - strike_shift)
            self.K2 = B
            self.K3 = K
            self.cp_phi = 1
            self.ratio = (self.K3 - self.K2) / (self.K2 - self.K1)
            self.w1 = PR * self.ratio
            self.w2 = PR * self.ratio + PR
            self.w3 = L - PR

        self._obs_info: list[dict] = []
        for obs_dt in self.obs_dts:
            if obs_dt < self.start:
                raise ValueError(f"Observation date {obs_dt} is before start {self.start}")
            cal_sec = (obs_dt - self.start).total_seconds()
            T_cal_i = cal_sec / (365 * 86400)
            trade_sec = count_trading_seconds_precise(self.start, obs_dt)
            T_trade_i = trade_sec / (self.ann * SECONDS_PER_FULL_TRADE_DAY)
            df_i = np.exp(-self.r * T_cal_i)
            self._obs_info.append({'dt': obs_dt, 'T_cal': T_cal_i, 'T_trade': T_trade_i, 'df': df_i, 'trade_sec': trade_sec})

    def _sum_call(self,K):
        """Sum of vanilla call prices at strike K across all observation dates."""
        total = 0
        for info in self._obs_info:
            v = VanillaBSM(S=self.S, K=K, T_trade=info['T_trade'], T_cal=info['T_cal'], r=self.r, b=self.b, sigma=self.sigma)
            total += v.price(1)
        return total

    def _sum_put(self,K):
        """Sum of vanilla put prices at strike K across all observation dates."""
        total = 0
        for info in self._obs_info:
            v = VanillaBSM(S=self.S, K=K, T_trade=info['T_trade'], T_cal=info['T_cal'], r=self.r, b=self.b,sigma=self.sigma)
            total += v.price(- 1)
        return total

    def price(self) -> float:
        """Total accumulator value via static replication."""
        if self.option_type == "call":
            sum_K3 = self._sum_put(self.K3)
            sum_K2 = self._sum_put(self.K2)
            sum_K1 = self._sum_put(self.K1)
            total = self.w1 * sum_K3 - self.w2 * sum_K2 - self.w3 * sum_K1
        else:
            sum_K1 = self._sum_call(self.B * (1.0 - self.strike_shift))
            sum_K2 = self._sum_call(self.B)
            sum_K3 = self._sum_call(self.K)
            total = self.w1 * sum_K1 - self.w2 * sum_K2 - self.w3 * sum_K3
        return total

    def greeks(self) -> dict:
        delta = gamma = vega = theta = 0.0
        for info in self._obs_info:
            v1 = VanillaBSM(self.S, self.K1, info['T_trade'], info['T_cal'], self.r, self.b, self.sigma)
            v2 = VanillaBSM(self.S, self.K2, info['T_trade'], info['T_cal'], self.r, self.b, self.sigma)
            v3 = VanillaBSM(self.S, self.K3, info['T_trade'], info['T_cal'], self.r, self.b, self.sigma)

            if self.option_type == "call":
                delta += (self.w1 * v3.delta(-1)
                          - self.w2 * v2.delta(-1)
                          - self.w3 * v1.delta(-1))
                gamma += (self.w1 * v3.gamma()
                         - self.w2 * v2.gamma()
                         - self.w3 * v1.gamma())
                vega += (self.w1 * v3.vega()
                         - self.w2 * v2.vega()
                         - self.w3 * v1.vega())
                theta += (self.w1 * v3.theta(-1)
                          - self.w2 * v2.theta(-1)
                          - self.w3 * v1.theta(-1))
            else:
                delta += (self.w1 * v1.delta(+1)
                          - self.w2 * v2.delta(+1)
                          - self.w3 * v3.delta(+1))
                gamma += (self.w1 * v1.gamma()
                          - self.w2 * v2.gamma()
                          - self.w3 * v3.gamma())
                vega += (self.w1 * v1.vega()
                         - self.w2 * v2.vega()
                         - self.w3 * v3.vega())
                theta += (self.w1 * v1.theta(+1)
                          - self.w2 * v2.theta(+1)
                          - self.w3 * v3.theta(+1))

        return {'delta': round(delta, 6), 'gamma': round(gamma, 6), 'vega': round(vega, 6), 'theta': round(theta, 6) }

    def obs_breakdown(self) -> list[dict]:
        """Per-observation contribution to the total price"""
        rows = []
        for info in self._obs_info:
            v1 = VanillaBSM(self.S, self.K1, info['T_trade'], info['T_cal'], self.r, self.b, self.sigma)
            v2 = VanillaBSM(self.S, self.K2, info['T_trade'], info['T_cal'], self.r, self.b, self.sigma)
            v3 = VanillaBSM(self.S, self.K3, info['T_trade'], info['T_cal'], self.r, self.b, self.sigma)

            if self.option_type == "call":
                p1, p2, p3 = v1.price(-1), v2.price(-1), v3.price(-1)
                contrib = self.w1 * p3 - self.w2 * p2 - self.w3 * p1
            else:
                p1, p2, p3 = v1.price(+1), v2.price(+1), v3.price(+1)
                contrib = self.w1 * p1 - self.w2 * p2 - self.w3 * p3

            rows.append({
                'date': str(info['dt'])[:10],
                'T': round(info['T_trade'], 4),
                'value': round(contrib, 6),
                'trade_sec':round(info['trade_sec'], 4)
            })

        return rows

    def summary(self, max_rows: int = 20):
        px = self.price()
        gk = self.greeks()
        rows = self.obs_breakdown()

        print("\n===== Accumulator Summary =====")
        print(f"Price : {px:.6f}")
        print(f"Delta : {gk['delta']:+.6f}")
        print(f"Gamma : {gk['gamma']:+.6f}")
        print(f"Vega  : {gk['vega']:+.6f}")
        print(f"Theta : {gk['theta']:+.6f}")

        print("\n--- Daily Contributions ---")
        for i, r in enumerate(rows[:max_rows]):
            print(f"{r['date']} | T={r['T']:.4f} | {r['value']:+.6f} | {r['trade_sec']:.6f} " )

        if len(rows) > max_rows:
            print(f"... ({len(rows)} days total)")

        print("----------------------------")

class AccumulatorMC:
    """
    Monte Carlo pricer for the accumulator, used to cross-validate AccumulatorReplication.

    Simulates GBM paths under the trading-time with antithetic variates, evaluates the piecewise-linear payoff on each path at
    each observation date, and discounts each contribution by the calendar-time.

    Convergence verification with replication method see analysis/mc_convergence.py.

    PARAMETERS:

    n_paths: Number of simulated paths
    seed: RNG seed for reproducibility.
    All other parameters as in AccumulatorReplication.
    """
    def __init__(self, S: float, K: float, B: float, r: float, b: float, sigma: float, L: float, PR: float,
                 option_type: str, start_dt: str, end_dt: str, n_paths: int = 20000, seed:int = 42):

        self.option_type = option_type.lower()
        self.start = parse_dt(start_dt)
        self.observation_dates = generate_trading_day_obs(start_dt, end_dt, obs_time=time(15, 0))
        self.obs_dts = [parse_dt(d) for d in self.observation_dates]
        self.S, self.K, self.B = S, K, B
        self.sigma, self.r, self.b = sigma, r, b
        self.L, self.PR = L, PR
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
        epsilon      = np.vstack([epsilon_half, - epsilon_half])
        drift = self.b * self.dt_cal - 0.5 * self.sigma ** 2 * self.dt_trade
        diffusion    = self.sigma * epsilon * np.sqrt(self.dt_trade)
        cumulative_log_return = np.cumsum(drift + diffusion, axis=1)
        S_obs        = self.S * np.exp(cumulative_log_return)
        return S_obs

    def pay_off(self, S_obs: np.ndarray) -> np.ndarray:
        if self.option_type == 'call':
            payoff = np.where (S_obs >= self.B, 0.0,
                               np.where(S_obs >= self.K, self.PR * (S_obs - self.K),
                                        - self.L * (self.K - S_obs))
                               )
        else:
            payoff = np.where (S_obs <= self.B, 0.0,
                               np.where(S_obs <= self.K, self.PR * (self.K - S_obs),
                                        - self.L * (S_obs - self.K))
                               )
        return payoff

    def price(self):
        rng = np.random.default_rng(self.seed)
        S_obs = self.simulate_gbm_path(rng)
        payoff = self.pay_off(S_obs)
        pv_paths = payoff @ self.df_obs
        mc_price = np.mean(pv_paths)
        return mc_price

    def obs_breakdown(self) -> list[dict]:
        rng = np.random.default_rng(self.seed)
        S_obs = self.simulate_gbm_path(rng)
        payoff_matrix = self.pay_off(S_obs)
        contrib_per_day = np.mean(payoff_matrix, axis=0) * self.df_obs
        rows = []
        for i, obs_dt in enumerate(self.obs_dts):
            rows.append({
                'date': str(obs_dt)[:10],
                'T': round(self.T_cal_obs[i], 4),
                'value': round(float(contrib_per_day[i]), 6),
            })
        return rows

    def summary(self, max_rows: int = 20):
        px     = self.price()
        rows   = self.obs_breakdown()
        print("\n===== Accumulator MC Summary =====")
        print(f"Type     : {self.option_type.upper()}")
        print(f"Paths    : {self.n_paths:,}  (antithetic)")
        print(f"Price    : {px:.6f}")
        print("\n--- Daily Contributions (MC) ---")
        for i, r in enumerate(rows[:max_rows]):
            print(f"  {r['date']} | T={r['T']:.4f} | {r['value']:+.6f}")
        if len(rows) > max_rows:
            print(f"  ... ({len(rows)} days total)")
        print("----------------------------")

if __name__ == '__main__':
    params = dict(
        S=1219, K=1175, B=1263,
        r=0.03, b=0.0, sigma=0.23,
        L=3.0, PR=1.0,
        strike_shift=0.002,
        option_type='call',
        start_dt="2026.05.19 14:09:01",
        end_dt="2026.06.23 15:00:00",
        seed=42,
    )

    mc = AccumulatorMC(S=1219, K=1175, B=1263, r=0.03, b=0.0, sigma=0.23,
                       L=3, PR=1, option_type='call',
                       start_dt="2026.05.19 14:09:01", end_dt="2026.06.23 15:00:00", n_paths=500000)
    S_obs = mc.simulate_gbm_path(np.random.default_rng(0))
    print("E[S_T] =", S_obs[:, -1].mean())
    print("forward =", 1219 * np.exp(0.08 * mc.T_cal_obs[-1]))
    mc.summary(max_rows=10)

    pricer = AccumulatorReplication(S=params['S'], K=params['K'], B=params['B'],
        r=params['r'], b=params['b'], sigma=params['sigma'],
        PR=params['PR'], L=params['L'], strike_shift=params['strike_shift'],
        option_type=params['option_type'],
        start_dt=params['start_dt'], end_dt=params['end_dt'],)


    pricer.summary(max_rows=10)