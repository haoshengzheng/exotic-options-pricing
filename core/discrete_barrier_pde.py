"""
Crank-Nicolson PDE pricer for discretely-monitored single-barrier options
under the dual-time (trading-time / calendar-time) framework.

It solves the dual-time BSM PDE on a finite-difference grid:

    dV/dtau = 1/2 * sigma^2 * S^2 * V_SS + b * S * V_S - r_eff * V

with tau = TRADING time-to-maturity (drives diffusion and carry b). The discount
rate is converted to the trading clock segment-by-segment via

    r_eff = r * (delta_t_cal / delta_tau_trade),

so accumulated discount over any interval equals exp(-r * delta_t_cal), exactly
mirroring the MC and the carry-fixed Haug formula.

Numerical choices
-----------------
* Crank-Nicolson in time; Thomas algorithm for the tridiagonal solve.
* Rannacher start-up: first `rannacher_steps` sub-steps after maturity and after
  every barrier touch are fully implicit, to damp CN oscillations at the kinks.
* Discrete monitoring absorbs the barrier ONLY at the daily-close observation
  nodes (from core.time_utils, shared with the MC). Continuous monitoring
  (absorb every sub-step) is provided as a rough sanity check vs Haug-continuous.
* Rebate-at-hit: a knocked-out node is set to K at the observation instant; the
  backward sweep discounts it from that date.
* Knock-in via decomposition: KI = [vanilla - KO(no rebate)] + no-touch-binary(K),
  the rebate paid at expiry if never touched (Haug term_E semantics).
* Grid uniform in S with barrier H snapped onto a node; S0 read by 3-point
  quadratic interpolation (removes the convexity bias of linear interpolation).
  Upper boundary uses the linearity condition V_SS = 0.
* Barrier absorption on a node converges at first order in dS. discrete_price()
  therefore defaults to Richardson extrapolation over two grids, which restores
  high accuracy cheaply (essential for thin barriers such as a down-out put,
  whose surviving value sits in a narrow band against the barrier).
"""
import numpy as np
from core.time_utils import (parse_dt, count_trading_seconds_precise,
                             generate_trading_day_obs, SECONDS_PER_FULL_TRADE_DAY,
                             trading_days_per_year as _TD_PER_YEAR)

CAL_SECONDS_PER_YEAR = 365 * 24 * 3600  # match the analytical / MC T_cal basis


def _thomas(sub, diag, sup, rhs):
    """Solve a tridiagonal system. sub[0] and sup[-1] are ignored."""
    n = len(diag)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = sup[0] / diag[0]; dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / m
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class DiscreteBarrierPDE:
    """
    PARAMETERS
    ----------
    start_dt, end_dt : inception / maturity timestamps (strings).
    S, X, H          : spot, strike, contract barrier.
    r, b, sigma      : risk-free rate, cost of carry, annualized vol (trading basis).
    K                : cash rebate (rebate-at-hit for KO; at-expiry-if-no-touch for KI).
    n_space          : spatial intervals on [0, S_max] (base resolution).
    n_sub            : Crank-Nicolson sub-steps per monitoring interval (base).
    s_max_mult       : S_max = s_max_mult * max(S, X, H).
    rannacher_steps  : fully-implicit start-up steps after each absorption.
    """

    def __init__(self, start_dt, end_dt, S, X, H, r, b, sigma, K=0.0,
                 trading_days_per_year=_TD_PER_YEAR, n_space=800, n_sub=40,
                 s_max_mult=3.0, rannacher_steps=2):
        self.start = parse_dt(start_dt); self.end = parse_dt(end_dt)
        self.S, self.X, self.H = float(S), float(X), float(H)
        self.r, self.b, self.sigma, self.K = r, b, sigma, float(K)
        self.ann = trading_days_per_year
        self.n_space = int(n_space); self.n_sub = int(n_sub)
        self.s_max_mult = s_max_mult; self.rannacher = int(rannacher_steps)

        self.T_cal   = (self.end - self.start).total_seconds() / CAL_SECONDS_PER_YEAR
        self.T_trade = count_trading_seconds_precise(self.start, self.end) / (self.ann * SECONDS_PER_FULL_TRADE_DAY)

        obs_strs = generate_trading_day_obs(start_dt, end_dt)
        self.obs_dts = [parse_dt(s) for s in obs_strs]

        # checkpoints in calendar order: inception, obs_1, ..., obs_M(=maturity)
        checkpoints = [self.start] + self.obs_dts
        self.segments = []  # (delta_tau_trade, delta_t_cal) for span (c_i, c_{i+1})
        for a, c in zip(checkpoints[:-1], checkpoints[1:]):
            dtau = count_trading_seconds_precise(a, c) / (self.ann * SECONDS_PER_FULL_TRADE_DAY)
            dcal = (c - a).total_seconds() / CAL_SECONDS_PER_YEAR
            self.segments.append((dtau, dcal))

    # --------------------------------------------------------- grid & operator
    def _grid_and_ops(self, n_space):
        S_max = self.s_max_mult * max(self.S, self.X, self.H)
        M = max(1, int(round(self.H / (S_max / n_space))))   # barrier index
        ds = self.H / M
        N = int(round(S_max / ds))
        grid = np.arange(N + 1) * ds                         # H sits on node M
        A = 0.5 * self.sigma ** 2 * grid ** 2 / ds ** 2
        B = self.b * grid / (2 * ds)
        L = A - B; U = A + B; Dsp = -2 * A                   # spatial coeffs (no -r_eff yet)
        L[0] = 0.0; U[0] = 0.0; Dsp[0] = 0.0                 # S=0: pure-discount, decoupled row
        L[N] = -self.b * grid[N] / ds; U[N] = 0.0; Dsp[N] = self.b * grid[N] / ds  # V_SS=0 BC
        return grid, ds, N, M, L, U, Dsp

    @staticmethod
    def _step(V, dtau, theta, r_eff, L, U, Dsp, N):
        diag_sp = Dsp - r_eff
        sub  = -theta * dtau * L
        diag = 1.0 - theta * dtau * diag_sp
        sup  = -theta * dtau * U
        LopV = np.empty(N + 1)
        LopV[1:N] = L[1:N] * V[0:N-1] + diag_sp[1:N] * V[1:N] + U[1:N] * V[2:N+1]
        LopV[0]   = diag_sp[0] * V[0]
        LopV[N]   = L[N] * V[N-1] + diag_sp[N] * V[N]
        rhs = V + (1.0 - theta) * dtau * LopV
        return _thomas(sub, diag, sup, rhs)

    @staticmethod
    def _absorb(V, knock_side, absorb_val, iH):
        if knock_side == 'down':
            V[:iH + 1] = absorb_val
        elif knock_side == 'up':
            V[iH:] = absorb_val

    def _read_S0(self, V, grid, ds, N):
        """3-point quadratic interpolation at S0 (kills the linear-interp convexity bias)."""
        j = min(max(int(self.S / ds), 1), N - 1)
        x0, x1, x2 = grid[j - 1], grid[j], grid[j + 1]
        y0, y1, y2 = V[j - 1], V[j], V[j + 1]
        s = self.S
        l0 = (s - x1) * (s - x2) / ((x0 - x1) * (x0 - x2))
        l1 = (s - x0) * (s - x2) / ((x1 - x0) * (x1 - x2))
        l2 = (s - x0) * (s - x1) / ((x2 - x0) * (x2 - x1))
        return float(y0 * l0 + y1 * l1 + y2 * l2)

    # --------------------------------------------------------- backward solver
    def _solve(self, payoff_kind, knock_side, absorb_val, monitoring, n_space, n_sub):
        grid, ds, N, iH, L, U, Dsp = self._grid_and_ops(n_space)
        if payoff_kind == 'call':
            V = np.maximum(grid - self.X, 0.0)
        elif payoff_kind == 'put':
            V = np.maximum(self.X - grid, 0.0)
        else:  # 'const_K' (no-touch binary terminal)
            V = np.full(N + 1, self.K, dtype=float)

        active = knock_side is not None
        if active:
            self._absorb(V, knock_side, absorb_val, iH)   # maturity is an observation
        for k in range(len(self.segments) - 1, -1, -1):   # march maturity -> inception
            dtau_tot, dcal_tot = self.segments[k]
            if dtau_tot <= 0:
                continue
            r_eff = self.r * (dcal_tot / dtau_tot)
            dtau = dtau_tot / n_sub
            for j in range(n_sub):
                theta = 1.0 if j < self.rannacher else 0.5
                V = self._step(V, dtau, theta, r_eff, L, U, Dsp, N)
                if active and monitoring == 'continuous':
                    self._absorb(V, knock_side, absorb_val, iH)
            if active and monitoring == 'discrete' and k >= 1:   # obs, not inception
                self._absorb(V, knock_side, absorb_val, iH)
        return self._read_S0(V, grid, ds, N)

    # ------------------------------------------------------------- price core
    def _price_core(self, barrier_type, monitoring, n_space, n_sub):
        bt = barrier_type.lower()
        kind = 'call' if bt.startswith('c') else 'put'
        is_upper = 'u' in bt; is_out = 'o' in bt
        knock_side = 'up' if is_upper else 'down'

        hit = (is_upper and self.S >= self.H) or (not is_upper and self.S <= self.H)
        if is_out and hit:
            return self.K
        if (not is_out) and hit:
            return self._solve(kind, None, None, monitoring, n_space, n_sub)  # KI already alive

        if is_out:
            return self._solve(kind, knock_side, self.K, monitoring, n_space, n_sub)
        # knock-in by decomposition
        v_vanilla     = self._solve(kind, None, None, monitoring, n_space, n_sub)
        v_ko_norebate = self._solve(kind, knock_side, 0.0, monitoring, n_space, n_sub)
        v_ki = v_vanilla - v_ko_norebate
        if self.K > 0:
            v_ki += self._solve('const_K', knock_side, 0.0, monitoring, n_space, n_sub)
        return v_ki

    # --------------------------------------------------------------- public API
    def price(self, barrier_type, monitoring='discrete', richardson=False):
        if not richardson:
            return self._price_core(barrier_type, monitoring, self.n_space, self.n_sub)
        v1 = self._price_core(barrier_type, monitoring, self.n_space, self.n_sub)
        v2 = self._price_core(barrier_type, monitoring, 2 * self.n_space, 2 * self.n_sub)
        return 2.0 * v2 - v1   # first-order (dS) Richardson extrapolation

    def discrete_price(self, barrier_type, richardson=True):
        """Discretely-monitored price; Richardson-extrapolated by default."""
        return self.price(barrier_type, monitoring='discrete', richardson=richardson)

    def continuous_price(self, barrier_type, richardson=False):
        """Continuously-monitored sanity-check price vs Haug-continuous."""
        return self.price(barrier_type, monitoring='continuous', richardson=richardson)

    def vanilla_price(self, barrier_type, monitoring='discrete', richardson=False):
        kind = 'call' if barrier_type.lower().startswith('c') else 'put'
        if not richardson:
            return self._solve(kind, None, None, monitoring, self.n_space, self.n_sub)
        v1 = self._solve(kind, None, None, monitoring, self.n_space, self.n_sub)
        v2 = self._solve(kind, None, None, monitoring, 2 * self.n_space, 2 * self.n_sub)
        return 2.0 * v2 - v1

    @property
    def time_info(self):
        return {"start": str(self.start), "end": str(self.end),
                "T_cal(yr)": round(self.T_cal, 6), "T_trade(yr)": round(self.T_trade, 6),
                "n_obs": len(self.obs_dts), "n_space": self.n_space, "n_sub": self.n_sub}