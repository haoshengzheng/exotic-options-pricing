import numpy as np
from scipy.stats import norm
from core.time_utils import trading_days_per_year


class VanillaBSM:
    """
    BSM pricer for European vanilla options under a dual-time framework.
    T_trade: trading time (seconds in active exchange sessions, annualized by trading-day count). Drives drift and diffusion.
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
        # At expiry, return discounted intrinsic value.
        if self.T_trade <= 0:
            return np.exp(-self.r * self.T_cal) * max(0, phi * (self.S - self.K))
        d1 = (np.log(self.S / self.K) + (self.b + 0.5 * self.sigma ** 2) * self.T_trade) / \
             (self.sigma * np.sqrt(self.T_trade))
        d2 = d1 - self.sigma * np.sqrt(self.T_trade)
        return phi * (self.S * np.exp(self.b * self.T_trade - self.r * self.T_cal) * norm.cdf(phi * d1) -
                      self.K * np.exp(-self.r * self.T_cal) * norm.cdf(phi * d2))

    def _d1_d2(self):
        """Floor T_trade to avoid division by zero when computing Greeks. 1e-6 is in stable zone of bump size."""
        t_tr = max(1e-6, self.T_trade)
        sqt = np.sqrt(t_tr)
        d1 = (np.log(self.S / self.K) + (self.b + 0.5 * self.sigma ** 2) * t_tr) / (self.sigma * sqt)
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

    def theta(self, phi: int) -> float:
        """
        Theta per trading day (not per calendar day).
        Three sources of time decay are combined: 1.Diffusion term  2.Carry cost  3.Discount on strike
        The result (annualized) is divided by `trading_days_per_year` to report per trading-day decay, matching how desks quote theta.
        """
        d1, d2, sqt = self._d1_d2()
        df_r = np.exp(-self.r * self.T_cal)
        df   = np.exp((self.b - self.r) * self.T_cal)
        theta = -(self.S * df * norm.pdf(d1) * self.sigma) / (2 * sqt) \
                - (self.b - self.r) * self.S * df * norm.cdf(phi * d1) * phi \
                - self.r * self.K * df_r * norm.cdf(phi * d2) * phi
        return theta / trading_days_per_year


    def greeks(self, phi: int) -> dict:
        return {'delta': self.delta(phi), 'gamma': self.gamma(),
                'vega': self.vega(), 'theta': self.theta(phi)}