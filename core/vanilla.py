import numpy as np
from scipy.stats import norm
from core.time_utils import trading_days_per_year, count_trading_seconds_precise, SECONDS_PER_FULL_TRADE_DAY


class VanillaBSM:
    """
    BSM pricer for European vanilla options under a dual-time framework.
    T_trade: trading time (seconds in active exchange sessions, annualized by trading-day count). Drives diffusion.
    T_cal: calendar time (continuous 365-day basis). Drives discounting and cost-of-carry.

    PARAMETERS:
    S: Spot price of the underlying.
    K: Strike.
    T_trade: Time to maturity in trading-time years.
    T_cal: Time to maturity in calendar-time years.
    r: Continuously compounded risk-free rate.
    b: Cost of carry (b = r for non-dividend stock, b = r - q for dividend-paying stock, b = 0 for futures-style underlying).
    sigma : Annualized volatility, measured in trading time.
    trading_days_per_year: Number of trading days per year in Chinese market, here used 242.

    Sign:
    All methods take phi: +1 for call, -1 for put.
    """
    def __init__(self, S: float, K: float, T_trade: float, T_cal: float, r: float, b: float, sigma: float):
        self.S, self.K = S, K
        self.T_trade, self.T_cal = T_trade, T_cal
        self.r, self.b, self.sigma = r, b, sigma

    def price(self, phi: int) -> float:
        # At expiry (trading time exhausted), return discounted intrinsic value.
        if self.T_trade <= 0:
            return np.exp(-self.r * self.T_cal) * max(0, phi * (self.S - self.K))
        d1 = (np.log(self.S / self.K) + self.b * self.T_cal + 0.5 * self.sigma ** 2 * self.T_trade) / \
             (self.sigma * np.sqrt(self.T_trade))
        d2 = d1 - self.sigma * np.sqrt(self.T_trade)
        return phi * (self.S * np.exp((self.b - self.r) * self.T_cal) * norm.cdf(phi * d1) -
                      self.K * np.exp(-self.r * self.T_cal) * norm.cdf(phi * d2))

    def _d1_d2(self):
        sqt = np.sqrt(self.T_trade)
        d1 = (np.log(self.S / self.K) + self.b * self.T_cal + 0.5 * self.sigma ** 2 * self.T_trade) / \
             (self.sigma * sqt)
        d2 = d1 - self.sigma * sqt
        return d1, d2, sqt

    def delta(self, phi: int) -> float:
        """Spot delta: dV/dS."""
        d1, _, _ = self._d1_d2()
        return phi * np.exp((self.b - self.r) * self.T_cal) * norm.cdf(phi * d1)

    def gamma(self) -> float:
        d1, _, sqt = self._d1_d2()
        return (norm.pdf(d1) * np.exp((self.b - self.r) * self.T_cal)) / (self.S * self.sigma * sqt)

    def vega(self) -> float:
        """
        Vega per 1% vol move (i.e., dV/dsigma divided by 100).
        Convention: report per absolute 0.01 change in sigma, so a vega of 0.42 means a 1% (0.01) vol increase adds 0.42 to option value.
        """
        d1, _, sqt = self._d1_d2()
        return self.S * np.exp((self.b - self.r) * self.T_cal) * norm.pdf(d1) * sqt / 100


    def theta_components(self, phi: int):
        d1, d2, sqt = self._d1_d2()
        df = np.exp((self.b - self.r) * self.T_cal)
        df_r = np.exp(-self.r * self.T_cal)

        dV_dTt = self.S * df * norm.pdf(d1) * self.sigma / (2 * sqt)
        dV_dTc = phi * (self.b - self.r) * self.S * df * norm.cdf(phi * d1) \
                 + phi * self.r * self.K * df_r * norm.cdf(phi * d2)

        theta_trade = -dV_dTt
        theta_cal = -dV_dTc
        return theta_trade, theta_cal

    def theta(self, phi: int) -> float:
        """
        Convenience theta for a *standard* roll: one trading day of trade-clock decay plus one calendar day of cal-clock decay.
        Each component uses its own divisor (242 vs 365) rather than a single shared one.
        """
        theta_trade, theta_cal = self.theta_components(phi)
        return theta_trade / trading_days_per_year + theta_cal/ 365

    def theta_per_observation(self, phi: int, now_dt, next_dt) -> float:
        dt_trade_years = count_trading_seconds_precise(now_dt, next_dt) / \
                         (SECONDS_PER_FULL_TRADE_DAY * trading_days_per_year)
        dt_cal_years = (next_dt - now_dt).total_seconds() / (365* 24 * 3600)
        theta_trade, theta_cal = self.theta_components(phi)
        return theta_trade * dt_trade_years + theta_cal * dt_cal_years

    def greeks(self, phi: int) -> dict:
        return {'delta': self.delta(phi), 'gamma': self.gamma(),
                'vega': self.vega(), 'theta': self.theta(phi)}