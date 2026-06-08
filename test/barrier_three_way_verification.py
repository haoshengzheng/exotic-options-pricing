from core.discrete_barrier import DiscreteBarrierPricer
from core.discrete_barrier_pde import DiscreteBarrierPDE
from core.discrete_barrier_mc import DiscreteBarrierMC


def three_way_table(params, types, n_paths=500000, n_space=800, n_sub=40, seed=42):
    an  = DiscreteBarrierPricer(**params)
    pde = DiscreteBarrierPDE(**params, n_space=n_space, n_sub=n_sub)
    mc  = DiscreteBarrierMC(**params, n_paths=n_paths, seed=seed)

    info = pde.time_info
    print(f"  S={params['S']}  X={params['X']}  H={params['H']}  r={params['r']}  "
          f"b={params['b']}  sigma={params['sigma']}  K={params['K']}")
    print(f"  T_cal={info['T_cal(yr)']}  T_trade={info['T_trade(yr)']}  n_obs={info['n_obs']}"
          f"  |  PDE {n_space}x{n_sub}   MC {n_paths:,} paths")
    print("-" * 94)
    print(f"{'Type':<5}| {'Analytic':>11} {'PDE':>11} {'MC (disc) +/- SE':>22} "
          f"| {'PDE-Anlyt':>11} {'MC-PDE':>11}")
    print(f"{'':<5}| {'(Haug+BGK)':>11} {'(disc)':>11} {'(discrete truth)':>22} "
          f"| {'(BGK err)':>11} {'(MC noise)':>11}")
    print("-" * 94)
    for bt in types:
        a     = an.discrete_price(bt)
        p     = pde.discrete_price(bt)
        m, se = mc.price_with_se(bt)
        print(f"{bt.upper():<5}| {a:>11.5f} {p:>11.5f} {m:>13.5f} +/-{se:>7.5f} "
              f"| {p - a:>+11.5f} {m - p:>+12f}")
    print("-" * 94)


if __name__ == "__main__":
    common = dict(start_dt="2026.05.01 14:00:00", end_dt="2026.05.28 15:00:00",
                  r=0.03, b=0.0, sigma=0.20, K=0.0)

    print("\n############ DOWN-barrier (H=95, below spot) ############")
    three_way_table(dict(**common, S=100, X=100.0, H=95.0), ["cdo", "cdi", "pdo", "pdi"])

    print("\n############ UP-barrier (H=110, above spot) ############")
    three_way_table(dict(**common, S=100, X=100.0, H=110.0), ["cuo", "cui", "puo", "pui"])

    print("\n############ Rebate + carry (K=2, b=0.05, down) ############")
    three_way_table(dict(start_dt="2026.05.01 14:00:00", end_dt="2026.05.28 15:00:00",
                         S=100, X=100.0, H=95.0, r=0.03, b=0.05, sigma=0.20, K=2.0),
                    ["cdo", "cdi", "pdo", "pdi"])
