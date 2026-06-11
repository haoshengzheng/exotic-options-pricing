"""
parity_test.py

No-arbitrage parity checks on the vanilla and discrete-barrier pricers.
Run directly:  /parity_test.py

1. Put-call parity (European vanilla):
       C - P = S*exp((b-r)*T_cal) - K*exp(-r*T_cal)
   A payoff-level identity (long call, short put = forward S_T - K). It does not
   depend on sigma or trading time — only on discounting/carry on the calendar
   clock — so it must hold for any sigma and any T_trade.

2. In-out barrier parity (rebate = 0):
       knock_in + knock_out = vanilla
   Every path either touches the barrier (in activates, out dies) or not (out
   survives, in never activates) — mutually exclusive and exhaustive — so the
   two payoffs sum to the unbarriered vanilla. A failure pinpoints an error in
   one of the Haug barrier blocks.
"""
import numpy as np

from core.vanilla import VanillaBSM
from core.discrete_barrier import HaugBarrierDualTime


TOL = 1e-8


def vanilla_price(S, K, T_trade, T_cal, r, b, sigma, phi):
    """phi = +1 call, -1 put.  (K here is the strike.)"""
    return VanillaBSM(S, K, T_trade, T_cal, r, b, sigma).price(phi)


def barrier_price(S, strike, H, T_trade, T_cal, r, b, sigma, barrier_type, rebate=0.0):
    """
    NOTE on the Haug class naming:
      X = strike   (Haug's notation)
      K = rebate   (cash paid on knock-out / failure to knock-in)
    So strike goes to X, and rebate goes to K.
    """
    pricer = HaugBarrierDualTime(
        S=S, X=strike, H=H, T_trade=T_trade, T_cal=T_cal,
        r=r, b=b, sigma=sigma, K=rebate,
    )
    return pricer.price(barrier_type)



# 1. Put-Call Parity   (T_trade != T_cal on purpose)
#    S,  K,  T_trade, T_cal, r,    b,   sigma
PARITY_CASES = [
    (100, 100, 0.85, 1.00, 0.03,  0.00, 0.20),
    (100,  90, 0.40, 0.50, 0.03,  0.01, 0.25),
    (100, 110, 1.60, 2.00, 0.05,  0.00, 0.15),
    (100, 100, 0.20, 0.25, 0.02,  0.02, 0.40),
    (120, 100, 0.90, 1.00, 0.03, -0.01, 0.30),
    (80,  100, 1.20, 1.50, 0.04,  0.00, 0.10),
]


def check_put_call_parity():
    print("Put-call parity (T_trade != T_cal; rhs uses T_cal only):")
    all_ok = True
    for S, K, T_trade, T_cal, r, b, sigma in PARITY_CASES:
        C = vanilla_price(S, K, T_trade, T_cal, r, b, sigma, +1)
        P = vanilla_price(S, K, T_trade, T_cal, r, b, sigma, -1)
        lhs = C - P
        rhs = S * np.exp((b - r) * T_cal) - K * np.exp(-r * T_cal)
        diff = abs(lhs - rhs)
        ok = diff < TOL
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  "
              f"S={S} K={K} T_trade={T_trade} T_cal={T_cal} sigma={sigma}:  "
              f"C-P={lhs:.8f}  fwd-Kpv={rhs:.8f}  diff={diff:.2e}")
    return all_ok


def check_parity_independent_of_trading_time():
    print("\nPut-call parity is invariant to T_trade (sweep T_trade, hold T_cal):")
    S, K, r, b, sigma = 100, 105, 0.03, 0.01, 0.25
    T_cal = 1.0
    rhs = S * np.exp((b - r) * T_cal) - K * np.exp(-r * T_cal)
    all_ok = True
    for T_trade in [0.3, 0.5, 0.8, 1.0, 1.3]:
        C = vanilla_price(S, K, T_trade, T_cal, r, b, sigma, +1)
        P = vanilla_price(S, K, T_trade, T_cal, r, b, sigma, -1)
        diff = abs((C - P) - rhs)
        ok = diff < TOL
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  T_trade={T_trade} T_cal={T_cal}:  "
              f"C-P={C-P:.8f}  expected={rhs:.8f}  diff={diff:.2e}")
    return all_ok



# 2. In-Out Barrier Parity  (rebate = 0, T_trade != T_cal on purpose)
#    S, strike, H, T_trade, T_cal, r,   b,  sigma, cp, direction
IN_OUT_CASES = [
    (100,  95,  85, 0.80, 1.00, 0.03, 0.00, 0.25, 'c', 'down'),
    (100,  80,  85, 0.80, 1.00, 0.03, 0.00, 0.25, 'c', 'down'),
    (100, 105, 120, 0.80, 1.00, 0.03, 0.00, 0.25, 'c', 'up'),
    (100, 125, 120, 0.80, 1.00, 0.03, 0.00, 0.25, 'c', 'up'),
    (100, 105,  85, 0.80, 1.00, 0.03, 0.00, 0.25, 'p', 'down'),
    (100,  80,  85, 0.80, 1.00, 0.03, 0.00, 0.25, 'p', 'down'),
    (100,  95, 120, 0.80, 1.00, 0.03, 0.00, 0.25, 'p', 'up'),
    (100, 125, 120, 0.80, 1.00, 0.03, 0.00, 0.25, 'p', 'up'),
    # non-zero carry with T_trade != T_cal: check of the reflection scaling
    (100, 95, 85, 0.80, 1.00, 0.03, 0.05, 0.25, 'c', 'down'),
    (100, 80, 85, 0.80, 1.00, 0.03, 0.05, 0.25, 'c', 'down'),
    (100, 105, 120, 0.80, 1.00, 0.03, 0.05, 0.25, 'c', 'up'),
    (100, 125, 120, 0.80, 1.00, 0.03, 0.05, 0.25, 'c', 'up'),
    (100, 105, 85, 0.80, 1.00, 0.03, 0.05, 0.25, 'p', 'down'),
    (100, 80, 85, 0.80, 1.00, 0.03, 0.05, 0.25, 'p', 'down'),
    (100, 95, 120, 0.80, 1.00, 0.03, 0.05, 0.25, 'p', 'up'),
    (100, 125, 120, 0.80, 1.00, 0.03, 0.05, 0.25, 'p', 'up'),
]


def check_in_out_parity():
    print("\nIn-out barrier parity (knock_in + knock_out = vanilla, rebate=0, T_trade != T_cal):")
    all_ok = True
    for S, strike, H, T_trade, T_cal, r, b, sigma, cp, direction in IN_OUT_CASES:
        phi = +1 if cp == 'c' else -1
        in_type  = f"{cp}{direction[0]}i"   # 'cdi','cui','pdi','pui'
        out_type = f"{cp}{direction[0]}o"   # 'cdo','cuo','pdo','puo'

        knock_in  = barrier_price(S, strike, H, T_trade, T_cal, r, b, sigma, in_type,  rebate=0.0)
        knock_out = barrier_price(S, strike, H, T_trade, T_cal, r, b, sigma, out_type, rebate=0.0)
        vanilla   = vanilla_price(S, strike, T_trade, T_cal, r, b, sigma, phi)

        lhs = knock_in + knock_out
        diff = abs(lhs - vanilla)
        ok = diff < TOL
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {in_type}+{out_type}  "
              f"S={S} X={strike} H={H} T_trade={T_trade} T_cal={T_cal} b={b}:  "
              f"in+out={lhs:.8f}  vanilla={vanilla:.8f}  diff={diff:.2e}")
    return all_ok


if __name__ == '__main__':
    r1 = check_put_call_parity()
    r2 = check_parity_independent_of_trading_time()
    r3 = check_in_out_parity()
    print(f"\n{'='*60}")
    print(f"Put-call parity:           {'ALL PASS' if r1 else 'SOME FAILED'}")
    print(f"Parity vs trading time:    {'ALL PASS' if r2 else 'SOME FAILED'}")
    print(f"In-out barrier parity:     {'ALL PASS' if r3 else 'SOME FAILED'}")
    print(f"{'='*60}")