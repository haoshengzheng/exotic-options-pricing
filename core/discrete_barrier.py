"""
Discrete-monitoring barrier option pricer.

Implements all 8 standard single-barrier types (CDI/CDO/CUI/CUO/PDI/PDO/PUI/PUO) via Haug (2007)'s closed-form decomposition into six building blocks A-F,
combined with the Broadie-Glasserman-Kou (1997) continuity correction for discrete monitoring.

Two time axes are used throughout, mirroring Chinese OTC desk practice:
  - T_trade : trading time, drives diffusion
  - T_cal   : calendar time, drives discounting
"""
import numpy as np
from scipy.stats import norm
from core.time_utils import count_trading_seconds_precise, count_trading_days, parse_dt, SECONDS_PER_FULL_TRADE_DAY, trading_days_per_year



class HaugBarrierDualTime:
    """
    Closed-form pricer for continuously-monitored single-barrier options under the Haug (2007) framework, extended with a dual-time clock.

    The Haug formula decomposes every barrier payoff into six building blocks A, B, C, D, E, F:
    A, B : vanilla-like terms involving Phi(x1), Phi(x2)
    C, D : reflected terms via (H/S)^(2*mu) and (H/S)^(2*(mu+1)), obtained from reflection principle, applied to the log-spot Brownian motion
    E, F : rebate terms, paid when (E) the option expires without touching the barrier, or (F) when the barrier is hit
    Each of the 8 barrier types is a specific linear combination of these six blocks, with the combination depending on whether the strike X is
    inside or outside the barrier H.

    PARAMETERS:
    S: Spot price.
    X: Strike (Haug uses X; equivalent to K in vanilla notation).
    H: Barrier level
    T_cal: Time to maturity in calendar-time years (used for discounting and carry).
    T_trade: Time to maturity in trading-time years (used for diffusion).
    r: Continuously compounded risk-free rate.
    b: Cost of carry.
    sigma: Annualized volatility (trading-time basis).
    K: Cash rebate paid when the contract knocks out (or fails to knock in). Set to 0 if the contract pays no rebate.

    Sign:
    phi = +1 for call, -1 for put
    eta = +1 for down-barrier, -1 for up-barrier
    """
    def __init__(self, S, X, H, T_cal, T_trade, r, b, sigma, K=0.0):
        self.S, self.X, self.H = S, X, H
        self.T_cal, self.T_trade = T_cal, T_trade
        self.r, self.b, self.sigma, self.K = r, b, sigma, K

        # Standard-deviation of log-spot over [0, T_trade]
        self.sigmaT = sigma * np.sqrt(T_trade)

        scale = T_cal / T_trade
        self.b_eff = b * scale
        self.r_eff = r * scale

        self.mu = (self.b_eff - sigma ** 2 / 2) / sigma ** 2
        self.lamda = np.sqrt(self.mu ** 2 + 2 * self.r_eff / sigma ** 2)


        self.x1 = np.log(S / X)      / self.sigmaT + (1 + self.mu) * self.sigmaT
        self.x2 = np.log(S / H)      / self.sigmaT + (1 + self.mu) * self.sigmaT
        self.y1 = np.log(H**2/(S*X)) / self.sigmaT + (1 + self.mu) * self.sigmaT
        self.y2 = np.log(H / S)      / self.sigmaT + (1 + self.mu) * self.sigmaT
        self.z  = np.log(H / S)      / self.sigmaT + self.lamda * self.sigmaT


    def term_A(self, phi):
        """Vanilla-like term anchored at strike X."""
        df_S = np.exp((self.b - self.r) * self.T_cal)  # spot (carry-discount) factor, calendar clock
        df_X = np.exp(-self.r * self.T_cal)            # strike discount factor
        return (phi * self.S * df_S * norm.cdf(phi * self.x1)
                - phi * self.X * df_X * norm.cdf(phi * self.x1 - phi * self.sigmaT))

    def term_B(self, phi):
        """Vanilla-like term anchored at barrier H."""
        df_S = np.exp((self.b - self.r) * self.T_cal)
        df_X = np.exp(-self.r * self.T_cal)
        return (phi * self.S * df_S * norm.cdf(phi * self.x2)
                - phi * self.X * df_X * norm.cdf(phi * self.x2 - phi * self.sigmaT))

    def term_C(self, phi, eta):
        df_S = np.exp((self.b - self.r) * self.T_cal)
        df_X = np.exp(-self.r * self.T_cal)
        pow1 = (self.H / self.S) ** (2 * (self.mu + 1))
        pow2 = (self.H / self.S) ** (2 * self.mu)
        return (phi * self.S * df_S * pow1 * norm.cdf(eta * self.y1)
                - phi * self.X * df_X * pow2 * norm.cdf(eta * self.y1 - eta * self.sigmaT))

    def term_D(self, phi, eta):
        df_S = np.exp((self.b - self.r) * self.T_cal)
        df_X = np.exp(-self.r * self.T_cal)
        pow1 = (self.H / self.S) ** (2 * (self.mu + 1))
        pow2 = (self.H / self.S) ** (2 * self.mu)
        return (phi * self.S * df_S * pow1 * norm.cdf(eta * self.y2)
                - phi * self.X * df_X * pow2 * norm.cdf(eta * self.y2 - eta * self.sigmaT))

    def term_E(self, eta):
        """Rebate paid at expiry conditional on the barrier NOT being hit (used by knock-in contracts that fail to knock in)."""
        if self.K <= 0:
            return 0.0
        df = np.exp(-self.r * self.T_cal)
        pow2 = (self.H / self.S) ** (2 * self.mu)
        return self.K * df * (
                norm.cdf(eta * self.x2 - eta * self.sigmaT)
                - pow2 * norm.cdf(eta * self.y2 - eta * self.sigmaT)
        )

    def term_F(self, eta):
        """Rebate paid immediately on first barrier touch (used by knock-out contracts as a fixed cash payment at hit)."""
        if self.K <= 0:
            return 0.0
        pow_p = (self.H / self.S) ** (self.mu + self.lamda)
        pow_m = (self.H / self.S) ** (self.mu - self.lamda)
        return self.K * (
                pow_p * norm.cdf(eta * self.z)
                + pow_m * norm.cdf(eta * self.z - 2 * eta * self.lamda * self.sigmaT)
        )

    def price(self, barrier_type: str) -> float:
        """
        Price one of the 8 single-barrier option types.
        barrier_type : str. One of {'cdi', 'cdo', 'cui', 'cuo', 'pdi', 'pdo', 'pui', 'puo'}.
            First letter: c=call, p=put.
            Second letter: d=down barrier, u=up barrier.
            Third letter: i=knock-in, o=knock-out.
        """
        phi = 1 if barrier_type.startswith('c') else -1
        eta = 1 if 'd' in barrier_type else -1

        bt = barrier_type.lower()

        if bt == 'cdo':
            if self.X >= self.H:
                return self.term_A(phi) - self.term_C(phi, eta) + self.term_F(eta)
            else:
                return self.term_B(phi) - self.term_D(phi, eta) + self.term_F(eta)

        elif bt == 'cdi':
            if self.X >= self.H:
                return self.term_C(phi, eta) + self.term_E(eta)
            else:
                return self.term_A(phi) - self.term_B(phi) + self.term_D(phi, eta) + self.term_E(eta)

        elif bt == 'cuo':
            if self.X >= self.H:
                return self.term_F(eta)
            else:
                return (self.term_A(phi) - self.term_B(phi)
                        + self.term_C(phi, eta) - self.term_D(phi, eta) + self.term_F(eta))

        elif bt == 'cui':
            if self.X >= self.H:
                return self.term_A(phi) + self.term_E(eta)
            else:
                return self.term_B(phi) - self.term_C(phi, eta) + self.term_D(phi, eta) + self.term_E(eta)

        elif bt == 'pdo':
            if self.X >= self.H:
                return (self.term_A(phi) - self.term_B(phi)
                        + self.term_C(phi, eta) - self.term_D(phi, eta) + self.term_F(eta))
            else:
                return self.term_F(eta)

        elif bt == 'pdi':
            if self.X >= self.H:
                return self.term_B(phi) - self.term_C(phi, eta) + self.term_D(phi, eta) + self.term_E(eta)
            else:
                return self.term_A(phi) + self.term_E(eta)

        elif bt == 'puo':
            if self.X >= self.H:
                return self.term_B(phi) - self.term_D(phi, eta) + self.term_F(eta)
            else:
                return self.term_A(phi) - self.term_C(phi, eta) + self.term_F(eta)

        elif bt == 'pui':
            if self.X >= self.H:
                return self.term_A(phi) - self.term_B(phi) + self.term_D(phi, eta) + self.term_E(eta)
            else:
                return self.term_C(phi, eta) + self.term_E(eta)

        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type!r}. "
                             "Must be one of: cdi/cdo/cui/cuo/pdi/pdo/pui/puo")


class DiscreteBarrierPricer:
    """
    Discrete-monitoring barrier option pricer with BGK continuity correction.

    Real-world barrier contracts are monitored at discrete observation times (typically daily close), not continuously. Discrete monitoring makes the
    barrier harder to hit, so a continuous-barrier formula systematically over-prices knock-outs and under-prices knock-ins.

    Broadie, Glasserman & Kou (1997) show that the adjusted barrier level should be:
            H_adj = H * exp( +/- beta * sigma * sqrt(dt) )
    where beta = -zeta(1/2) / sqrt(2*pi) ~= 0.5826, dt is the time between observations, and the sign is + for upper barriers (shift up) and - for
    lower barriers (shift down). The adjusted barrier is then fed into the standard continuous-barrier (Haug) formula.

    PARAMETERS:
    start_dt, end_dt : Inception and maturity timestamps (parsed by core.time_utils.parse_dt).
    S, X, H : Spot, strike, original (contract) barrier.
    r, b, sigma :  Risk-free rate, cost of carry, annualized volatility.
    K : Cash rebate.
    beta : float, default 0.5826
    """

    BETA_DEFAULT = 0.5826

    def __init__(self, start_dt: str, end_dt: str, S: float, X: float, H: float, r: float, b: float, sigma: float, K: float = 0.0, \
                 trading_days_per_year: int = trading_days_per_year , beta: float = BETA_DEFAULT, ):

        self.start = parse_dt(start_dt)
        self.end   = parse_dt(end_dt)
        self.S, self.X, self.H_orig = S, X, H
        self.r, self.b, self.sigma, self.K = r, b, sigma, K
        self.beta = beta
        self.ann = trading_days_per_year
        self.n_trading = count_trading_days(self.start, self.end)

        total_calendar_seconds = (self.end - self.start).total_seconds()
        self.T_cal   = total_calendar_seconds / (365 * 24 * 3600)

        self.total_trade_seconds = count_trading_seconds_precise(self.start, self.end)
        self.T_trade = self.total_trade_seconds / (self.ann * SECONDS_PER_FULL_TRADE_DAY)
        self.dt = 1.0 / self.ann


    @property
    def time_info(self) -> dict:
        """Human-readable summary of the two time axes."""
        cal_days = (self.end.date() - self.start.date()).days
        return {
            "start":     str(self.start),
            "end":     str(self.end),
            "calendar_days":     cal_days,
            "T_cal(yr)":  round(self.T_cal, 6),
            "trading_days":     self.n_trading,
            "T_trade(yr)": round(self.T_trade, 6),
        }

    def _vanilla_price(self, barrier_type: str) -> float:
        """
        Plain BSM price under the dual-time clock. Used as the payoff when a knock-in barrier
        has already been touched (the contract has effectively become a vanilla option).
        """
        is_call = 'c' in barrier_type.lower()
        S, X = self.S, self.X
        r, b, sigma = self.r, self.b, self.sigma
        T_cal = self.T_cal
        T_trade = self.T_trade

        if T_trade <= 0:
            return max(0.0, S - X) if is_call else max(0.0, X - S)
        d1 = (np.log(S / X) + b * T_cal + 0.5 * sigma ** 2 * T_trade) / (sigma * np.sqrt(T_trade))
        d2 = d1 - sigma * np.sqrt(T_trade)
        if is_call:
            price = S * np.exp((b - r) * T_cal) * norm.cdf(d1) \
                    - X * np.exp(-r * T_cal) * norm.cdf(d2)
        else:
            price = X * np.exp(-r * T_cal) * norm.cdf(-d2) \
                    - S * np.exp((b - r) * T_cal) * norm.cdf(-d1)
        return price

    def _bgk_adjust(self, is_upper: bool) -> float:
        correction = self.beta * self.sigma * np.sqrt(self.dt)
        sign = +1 if is_upper else -1
        return self.H_orig * np.exp(sign * correction)

    def discrete_price(self, barrier_type: str, show_detail: bool = False,) -> float:
        """
        Price a discrete-monitored barrier option via BGK-adjusted Haug formula.
        Steps:
        1. Compute BGK-adjusted barrier H_adj.
        2. Check if spot already breaches H_adj:
           - Knock-out: pay rebate K immediately.
           - Knock-in : option already alive, return equivalent vanilla.
        3. Otherwise, price via HaugBarrierDualTime with H_adj.
        """
        bt = barrier_type.lower()
        is_upper = 'u' in bt
        is_out = 'o' in bt
        is_in = 'i' in bt
        H_adj = self._bgk_adjust(is_upper)
        pricer = HaugBarrierDualTime( S=self.S, X=self.X, H=H_adj, T_cal=self.T_cal, T_trade=self.T_trade, r=self.r, b=self.b,
                                      sigma=self.sigma, K=self.K )


        # Edge case: if spot already on the wrong side of the adjusted barrier, the contract's state is determined and the Haug formula doesn't apply.
        hit_barrier = (is_upper and self.S >= H_adj) or (not is_upper and self.S <= H_adj)
        if is_out:
            if hit_barrier:
                return self.K
        elif is_in:
            if hit_barrier:
                return self._vanilla_price(barrier_type)

        price_val = pricer.price(bt)
        if show_detail:
            self._print_detail(barrier_type, H_adj, price_val)

        return price_val

    def _print_detail(self, bt, H_adj, price_val):
        """Print the pricing breakdown for debugging."""
        correction_pct = (H_adj / self.H_orig - 1) * 100
        direction = "up" if 'u' in bt.lower() else "down"
        print("=" * 55)
        print(f"  Discrete Barrier Option Pricing Detail  [{bt.upper()}]")
        print("=" * 55)
        for k, v in self.time_info.items():
            print(f"  {k:<15}: {v}")
        print(f"  {'Spot S':<15}: {self.S}")
        print(f"  {'Strike X':<15}: {self.X}")
        print(f"  {'Original H':<15}: {self.H_orig}")
        print(f"  BGK shift ({direction}): H -> {H_adj:.4f}  ({correction_pct:+.3f}%)")
        print(f"  beta={self.beta}, sigma={self.sigma}, dt=1/{self.ann}={self.dt:.6f}")
        print(f"  {'Volatility':<15}: {self.sigma}")
        print(f"  {'Risk-free r':<15}: {self.r}")
        print(f"  {'Cost-of-carry b':<15}: {self.b}")
        print("-" * 55)
        print(f"  ->  Option price: {price_val:.6f}")
        print("=" * 55)


    def continuous_price(self, barrier_type: str) -> float:
        """
        Price the same contract assuming continuous monitoring (no BGK adjustment). Useful for quantifying the discreteness premium.
        """
        pricer = HaugBarrierDualTime( S=self.S, X=self.X, H=self.H_orig, T_cal=self.T_cal, T_trade=self.T_trade,
                                      r=self.r, b=self.b, sigma=self.sigma, K=self.K )
        bt = barrier_type.lower()
        is_upper = 'u' in bt
        is_out = 'o' in bt

        if is_upper and self.S >=self.H_orig:
            if is_out:
                return self.K
        if (not is_upper) and (self.S <= self.H_orig):
            if is_out:
                return self.K

        return pricer.price(barrier_type.lower())

    def compare(self, barrier_type: str):
        """Print side-by-side continuous vs discrete-monitored prices."""
        p_cont = self.continuous_price(barrier_type)
        p_disc = self.discrete_price(barrier_type)
        diff = p_disc - p_cont
        print(f"\n  [{barrier_type.upper()}] Continuous vs Discrete Barrier")
        print(f"  Continuous price : {p_cont:.6f}")
        print(f"  Discrete price   : {p_disc:.6f}")
        print(f"  Diff (disc-cont) : {diff:+.6f}  ({diff / p_cont * 100:+.3f}%)\n")

if __name__ == "__main__":
    pricer = DiscreteBarrierPricer(
        start_dt = "2026.05.01 14:00:00",
        end_dt   = "2026.05.28 15:00:00",
        S        = 120,
        X        = 105.0,
        H        = 110.0,
        r        = 0.03,
        b        = 0.0,
        sigma    = 0.20,
        K        = 2.0,
    )

    pricer.discrete_price("pdo", show_detail=False)
    pricer.discrete_price("pdi", show_detail=False)

    pricer.compare("cuo")
    pricer.compare("pdi")