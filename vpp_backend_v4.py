"""
VPP lifetime sizing backend (v4: PV + wind + SMR + grid, hydrogen chain optional)
================================
Implements the "Lifetime Sizing Model for a Grid-Connected VPP" document:

  * representative days (month x day-type medoids) + stress days + a stress window
    built from the 8760-hour record, weighted so that the weights sum to 365 days
  * 25-year (configurable) evolution: load growth, PV/wind degradation,
    electrolyzer / fuel-cell efficiency loss, tariff and fuel escalation
  * replacements (with the "remaining horizon >= nu * life" rule), residual value,
    decommissioning, and the critical years used for the reliability check
  * rule-based hourly dispatch (FC -> SMR ramp -> grid -> unmet; surplus -> electrolyzer),
    vectorized across days/years, with a cyclic hydrogen tank for typical days
  * NPV, lifetime system LCOE vs grid LCOE, per-technology LCOE / LCOS, payback, IRR
  * EENS in every year as a constraint, inside a constrained-domination NSGA-II
  * full 8760-hour chronological verification of the Pareto designs

DSM, EV and Shapley allocation are intentionally NOT part of this version.
All money is in constant (real) rupees; the discount rate is a real rate.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TECHS = ["PV", "WT", "SMR", "EL", "TK", "FC"]
TECH_LABELS = {
    "PV": "Solar PV (kWp)", "WT": "Wind (kW)", "SMR": "SMR (kW)",
    "EL": "Electrolyzer (kW)", "TK": "H2 tank (kWh-eq)", "FC": "Fuel cell (kW)",
}

DEFAULT_TECH = {
    "PV":  dict(capex=45000.0,  om=0.010, life=25, decomm=0.04),
    "WT":  dict(capex=75000.0,  om=0.020, life=20, decomm=0.07),
    "SMR": dict(capex=650000.0, om=0.025, life=60, decomm=0.12),
    "EL":  dict(capex=55000.0,  om=0.020, life=12, decomm=0.05),
    "TK":  dict(capex=6000.0,   om=0.010, life=22, decomm=0.05),
    "FC":  dict(capex=95000.0,  om=0.030, life=15, decomm=0.08),
}


def default_tariff() -> np.ndarray:
    """TOU grid tariff, Rs/kWh by hour of day (same as v2)."""
    t = np.full(24, 5.0)
    t[7:10] = 7.5
    t[18:22] = 10.0
    return t


@dataclass
class Params:
    # horizon and economics
    N: int = 25
    r: float = 0.08
    nu: float = 0.5
    voll: float = 120.0
    eens_target: float = 50_000.0
    grid_cap_kw: float = 1750.0 * 0.95
    # technology performance
    pr: float = 0.784
    eta_el: float = 0.68
    eta_fc: float = 0.55
    smr_base: float = 0.55
    smr_ramp: float = 0.25
    smr_avail: float = 1.0
    smr_fuel: float = 0.9
    # lifetime evolution (placeholders until real data)
    g_load: float = 0.03
    d_pv: float = 0.005
    d_wt: float = 0.003
    d_el: float = 0.008
    d_fc: float = 0.008
    e_tariff: float = 0.01
    e_fuel: float = 0.005
    eps_capex: dict = field(default_factory=lambda: dict(PV=0.0, WT=0.0, SMR=0.0, EL=-0.04, TK=0.0, FC=-0.03))
    # wind
    v_in: float = 2.5
    v_rated: float = 11.0
    v_out: float = 25.0
    z_ref: float = 50.0
    z_hub: float = 50.0
    shear: float = 0.14
    # inter-annual scenarios
    p_low: float = 0.10
    # v4: load-growth cases (rate, probability); None -> single case g_load
    g_cases: list | None = None
    pv_p10: float = 0.96      # assumed low-resource year factors when <3 full years of data
    wt_p10: float = 0.90
    active: tuple = tuple(TECHS)   # technologies that can be built (drives replacement / critical years)
    tech: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_TECH.items()})
    tariff: np.ndarray = field(default_factory=default_tariff)

    def replace(self, **kw) -> "Params":
        return dataclasses.replace(self, **kw)


# ----------------------------------------------------------------------
# 1. CSV ingestion and alignment
# ----------------------------------------------------------------------
def _parse_ts(s: pd.Series) -> pd.Series:
    """ISO first (2025-01-01 00:00), then day-first (01/01/2025 00:00)."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="ISO8601")
    if out.isna().any():
        out = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    if out.isna().any():
        bad = s[out.isna()].iloc[0]
        raise ValueError(f"Could not read timestamp '{bad}'. Use YYYY-MM-DD HH:MM or DD/MM/YYYY HH:MM.")
    return out


def load_series(file, col: str) -> pd.Series:
    """Read a CSV with columns timestamp + `col` into a float Series on a DatetimeIndex."""
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "timestamp" not in df.columns or col not in df.columns:
        raise ValueError(f"CSV must have columns 'timestamp' and '{col}' (found: {list(df.columns)}).")
    df["timestamp"] = _parse_ts(df["timestamp"])
    s = df.set_index("timestamp")[col].astype(float)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def align_inputs(ghi: pd.Series, wind: pd.Series, load: pd.Series):
    """Inner-join the three hourly series, fill gaps of up to 6 h, trim to whole days.
    Returns (DataFrame[ghi, wind_speed, load_kw], info dict)."""
    df = pd.concat([ghi.rename("ghi"), wind.rename("wind_speed"), load.rename("load_kw")],
                   axis=1, join="inner").sort_index()
    if len(df) < 24 * 30:
        raise ValueError("Fewer than 30 days of overlapping hourly data across solar, wind and load files.")
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    n_missing = len(full) - len(df)
    df = df.reindex(full).interpolate(limit=6, limit_area="inside")
    if df.isna().any().any():
        raise ValueError("The hourly record has gaps longer than 6 hours; please fill them before uploading.")
    start = df.index[df.index.hour == 0][0]
    df = df.loc[start:]
    n_days = len(df) // 24
    if n_days < 30:
        raise ValueError("Fewer than 30 whole days after trimming to midnight-to-midnight.")
    df = df.iloc[: n_days * 24].copy()
    df["ghi"] = df["ghi"].clip(lower=0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0)
    df["load_kw"] = df["load_kw"].clip(lower=0)
    return df, dict(n_days=n_days, n_gap_hours_filled=int(n_missing),
                    start=str(df.index[0]), end=str(df.index[-1]))


def synthetic_year(year: int = 2025, seed: int = 1) -> pd.DataFrame:
    """Demo data used when no CSV is uploaded (clearly labelled synthetic in the app)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    hr = idx.hour.values + idx.minute.values / 60.0
    doy = idx.dayofyear.values
    bell = np.where((hr >= 6) & (hr <= 18.5), np.sin(np.pi * (hr - 6) / 12.5), 0.0)
    n_days = 365
    day_cloud = np.clip(0.80 + 0.20 * np.cos(2 * np.pi * (np.arange(n_days) - 20) / 365)
                        - 0.55 * np.clip(np.sin(np.pi * (np.arange(n_days) - 150) / 120), 0, 1)
                        + rng.normal(0, 0.12, n_days), 0.1, 1.0)
    ghi = 925.0 * bell * np.repeat(day_cloud, 24)
    seasonal = 3.0 + 2.2 * np.exp(-((doy - 195) / 45.0) ** 2)
    diurnal = 1.0 + 0.35 * np.sin(2 * np.pi * (idx.hour.values - 9) / 24)
    noise = np.zeros(8760)
    e = rng.normal(0, 1, 8760)
    for i in range(1, 8760):
        noise[i] = 0.93 * noise[i - 1] + 0.37 * e[i]
    wind = np.clip(seasonal * diurnal * np.exp(0.35 * noise), 0, 20)
    weekend = (idx.dayofweek.values >= 5).astype(float)
    load = (330 + 120 * np.exp(-((idx.hour.values - 20) / 3.5) ** 2) + 60 * np.exp(-((idx.hour.values - 12) / 4.0) ** 2)
            - 40 * weekend + 25 * np.sin(2 * np.pi * (doy - 120) / 365) + rng.normal(0, 12, 8760))
    return pd.DataFrame(dict(ghi=ghi, wind_speed=wind, load_kw=np.clip(load, 150, None)), index=idx)


def wind_power_fraction(v: np.ndarray, p: Params) -> np.ndarray:
    vh = np.asarray(v, float) * (p.z_hub / p.z_ref) ** p.shear
    f = np.zeros_like(vh)
    ramp = (vh >= p.v_in) & (vh < p.v_rated)
    f[ramp] = ((vh[ramp] - p.v_in) / (p.v_rated - p.v_in)) ** 3
    f[(vh >= p.v_rated) & (vh < p.v_out)] = 1.0
    return f


# ----------------------------------------------------------------------
# 2. Representative days, stress days, scenarios
# ----------------------------------------------------------------------
@dataclass
class DayBank:
    Y: float
    L_t: np.ndarray; pv_t: np.ndarray; wt_t: np.ndarray; w_t: np.ndarray; info_t: pd.DataFrame
    L_s: np.ndarray; pv_s: np.ndarray; wt_s: np.ndarray; w_s: np.ndarray; par_s: np.ndarray; info_s: pd.DataFrame
    L_w: np.ndarray | None; pv_w: np.ndarray | None; wt_w: np.ndarray | None
    w_w: np.ndarray | None; par_w: int; info_w: dict
    scen: list
    ver: dict
    n_days: int


def build_bank(df: pd.DataFrame, p: Params, cap_pv_ref: float, cap_wt_ref: float,
               K: int = 5, W: int = 5) -> DayBank:
    n_days = len(df) // 24
    Y = n_days / 365.0
    L = df["load_kw"].values.reshape(n_days, 24)
    pv = (df["ghi"].values / 1000.0 * p.pr).reshape(n_days, 24)
    wt = wind_power_fraction(df["wind_speed"].values, p).reshape(n_days, 24)
    dates = df.index[::24]
    month = dates.month.values
    dtype = (dates.dayofweek.values >= 5).astype(int)

    score = np.maximum(L - cap_pv_ref * pv - cap_wt_ref * wt, 0.0).sum(axis=1)
    taken = np.zeros(n_days, bool)

    win_idx = None
    if W >= 2 and n_days >= W + K + 24:
        cs = np.concatenate([[0.0], np.cumsum(score)])
        ws = cs[W:] - cs[:-W]
        s0 = int(np.argmax(ws))
        win_idx = np.arange(s0, s0 + W)
        taken[win_idx] = True
    order = np.argsort(-score)
    sd = [int(i) for i in order if not taken[i]][: max(K, 0)]
    taken[sd] = True

    sc = lambda a: float(np.std(a)) if np.std(a) > 1e-9 else 1.0
    Z = np.hstack([L / sc(L), pv / sc(pv), wt / sc(wt)])

    rows_t, lab = [], []
    key_to_typ = {}
    for m in range(1, 13):
        for q in (0, 1):
            idx = np.where((month == m) & (dtype == q) & (~taken))[0]
            if len(idx) == 0:
                continue
            zbar = Z[idx].mean(axis=0)
            med = int(idx[np.argmin(((Z[idx] - zbar) ** 2).sum(axis=1))])
            key_to_typ[(m, q)] = len(rows_t)
            rows_t.append(med)
            lab.append(dict(group=f"{pd.Timestamp(2001, m, 1):%b} {'weekend' if q else 'weekday'}",
                            medoid_date=str(dates[med].date()), days_in_group=len(idx),
                            weight_days_per_year=len(idx) / Y))
    if not rows_t:
        raise ValueError("Could not form any typical day from the data.")
    info_t = pd.DataFrame(lab)
    w_t = info_t["weight_days_per_year"].values.astype(float)

    def parent_of(day):
        key = (int(month[day]), int(dtype[day]))
        if key in key_to_typ:
            return key_to_typ[key]
        # nearest month with the same day type, else first
        cands = [(abs(k[0] - key[0]), v) for k, v in key_to_typ.items() if k[1] == key[1]]
        return min(cands)[1] if cands else 0

    par_s = np.array([parent_of(i) for i in sd], dtype=int)
    info_s = pd.DataFrame(dict(date=[str(dates[i].date()) for i in sd],
                               residual_load_score_kWh=[float(score[i]) for i in sd],
                               weight_days_per_year=[1.0 / Y] * len(sd),
                               parent_group=[info_t["group"].iloc[j] for j in par_s]))
    if win_idx is not None:
        L_w = L[win_idx].reshape(1, -1); pv_w = pv[win_idx].reshape(1, -1); wt_w = wt[win_idx].reshape(1, -1)
        w_w = np.array([1.0 / Y]); par_w = parent_of(int(win_idx[0]))
        info_w = dict(start=str(dates[win_idx[0]].date()), end=str(dates[win_idx[-1]].date()), days=int(W))
    else:
        L_w = pv_w = wt_w = w_w = None; par_w = 0; info_w = {}

    # resource scenarios and the chronological verification year
    years = dates.year.values
    uy, cnt = np.unique(years, return_counts=True)
    full_years = [int(y) for y, c in zip(uy, cnt) if c >= 360]
    scen = [("base", 1.0, 1.0, 1.0 - p.p_low),
            ("low-resource (assumed P10)", p.pv_p10, p.wt_p10, p.p_low)]
    if len(full_years) >= 3:
        ypv = np.array([pv[years == y].mean() for y in full_years])
        ywt = np.array([wt[years == y].mean() for y in full_years])
        kpv = float(np.percentile(ypv, 10) / ypv.mean()) if ypv.mean() > 1e-9 else 1.0
        kwt = float(np.percentile(ywt, 10) / ywt.mean()) if ywt.mean() > 1e-9 else 1.0
        scen = [("base", 1.0, 1.0, 1.0 - p.p_low), ("low-resource (P10 year)", kpv, kwt, p.p_low)]
    if full_years:
        ysc = {y: score[years == y].sum() for y in full_years}
        vy = max(ysc, key=ysc.get) if len(full_years) > 1 else full_years[0]
        mask = years == vy
        label = f"{vy}" + (" (lowest-resource year)" if len(full_years) > 1 else "")
    else:
        mask = np.ones(n_days, bool); label = "whole record"
    ver = dict(L=L[mask].reshape(-1), pv=pv[mask].reshape(-1), wt=wt[mask].reshape(-1),
               scale=365.0 / mask.sum(), label=label, n_days=int(mask.sum()))
    return DayBank(Y=Y, L_t=L[rows_t], pv_t=pv[rows_t], wt_t=wt[rows_t], w_t=w_t, info_t=info_t,
                   L_s=L[sd] if sd else np.zeros((0, 24)), pv_s=pv[sd] if sd else np.zeros((0, 24)),
                   wt_s=wt[sd] if sd else np.zeros((0, 24)), w_s=np.full(len(sd), 1.0 / Y),
                   par_s=par_s, info_s=info_s,
                   L_w=L_w, pv_w=pv_w, wt_w=wt_w, w_w=w_w, par_w=par_w, info_w=info_w,
                   scen=scen, ver=ver, n_days=n_days)


# ----------------------------------------------------------------------
# 3. Lifetime schedule: installs, replacements, residual value, critical years
# ----------------------------------------------------------------------
def lifetime_schedule(p: Params) -> dict:
    N = p.N
    sched, rep_times = {}, set()
    for k in TECHS:
        Lk = max(int(round(p.tech[k]["life"])), 1)
        inst = [0] + [j * Lk for j in range(1, N // Lk + 2) if j * Lk < N and (N - j * Lk) >= p.nu * Lk]
        last = max(inst)
        rho = max(0.0, 1.0 - (N - last) / Lk)
        age = np.array([(t - 1) - max(i for i in inst if i <= t - 1) for t in range(1, N + 1)], dtype=float)
        decom = [i for i in inst if i > 0] + ([N] if rho == 0.0 else [])
        sched[k] = dict(life=Lk, inst=inst, last=last, rho=rho, age=age, decom=decom)
        if k in p.active:
            rep_times |= {i for i in inst if i > 0}
    sched["rep_times"] = sorted(rep_times)
    sched["T_crit"] = sorted(rep_times | {N})
    sched["milestones"] = sorted({1, N} | rep_times | {t + 1 for t in rep_times} | set(range(1, N + 1, 5)))
    return sched


# ----------------------------------------------------------------------
# 4. Context: everything that does not depend on the design x
# ----------------------------------------------------------------------
def baseline(segments, p: Params) -> dict:
    """All-grid baseline per year t=1..N from (L[D,T], w[D]) segments (base tariff, no escalation)."""
    N = p.N
    scale = (1 + p.g_load) ** (np.arange(1, N + 1) - 1)
    grid0w = np.zeros(N); unm0 = np.zeros(N); load = np.zeros(N)
    for L, w in segments:
        if L is None or len(L) == 0:
            continue
        D, T = L.shape
        price = np.tile(p.tariff, T // 24)
        Ls = L[None, :, :] * scale[:, None, None]
        g0 = np.minimum(Ls, p.grid_cap_kw)
        wv = np.asarray(w, float)[None, :]
        grid0w += ((g0 * price[None, None, :]).sum(axis=2) * wv).sum(axis=1)
        unm0 += (((Ls - g0).sum(axis=2)) * wv).sum(axis=1)
        load += (Ls.sum(axis=2) * wv).sum(axis=1)
    return dict(grid0w=grid0w, unm0=unm0, load=load)


class Ctx:
    def __init__(self, bank: DayBank, p: Params, milestones=None):
        self.bank, self.p = bank, p
        self.sched = lifetime_schedule(p)
        self.T_crit = np.array(self.sched["T_crit"], dtype=int)
        ms = milestones if milestones is not None else self.sched["milestones"]
        self.years_M = np.array(sorted({int(m) for m in ms}), dtype=int)
        self.M = len(self.years_M)
        t = self.years_M
        age = lambda k: self.sched[k]["age"][t - 1]
        gc = p.g_cases if p.g_cases else [(p.g_load, 1.0)]
        combos = []
        for (g, pg) in gc:
            for (nm, kp, kw, pr) in bank.scen:
                combos.append(dict(name=f"g={g * 100:.1f}% | {nm}", g=float(g), kpv=kp, kwt=kw, prob=pg * pr))
        self.combos = combos
        self.g_s = np.array([c["g"] for c in combos])
        self.load_scale = (1 + self.g_s)[:, None] ** (t - 1)[None, :]          # (S, M)
        self.i_base = int(np.argmax([c["prob"] * (c["kpv"] == 1.0 and c["kwt"] == 1.0) for c in combos]))
        self.a_pv = (1 - p.d_pv) ** age("PV")
        self.a_wt = (1 - p.d_wt) ** age("WT")
        self.eta_el = p.eta_el * (1 - p.d_el) ** age("EL")
        self.eta_fc = p.eta_fc * (1 - p.d_fc) ** age("FC")
        self.S = len(combos)
        self.k_pv = np.array([c["kpv"] for c in combos]); self.k_wt = np.array([c["kwt"] for c in combos])
        self.p_s = np.array([c["prob"] for c in combos]); self.p_s = self.p_s / self.p_s.sum()
        self.blocks = {"typ": self._block(bank.L_t, bank.pv_t, bank.wt_t, bank.w_t)}
        if len(bank.L_s):
            self.blocks["str"] = self._block(bank.L_s, bank.pv_s, bank.wt_s, bank.w_s)
        if bank.L_w is not None:
            self.blocks["win"] = self._block(bank.L_w, bank.pv_w, bank.wt_w, bank.w_w)
        segs = [(bank.L_t, bank.w_t)]
        if len(bank.L_s):
            segs.append((bank.L_s, bank.w_s))
        if bank.L_w is not None:
            segs.append((bank.L_w, bank.w_w))
        self.base = [baseline(segs, p.replace(g_load=g)) for g in self.g_s]

    def _block(self, L, fpv, fwt, w):
        S, M = self.S, self.M
        D, T = L.shape
        Lr = L[None, None] * self.load_scale[:, :, None, None]
        pvr = fpv[None, None] * (self.k_pv[:, None, None, None] * self.a_pv[None, :, None, None])
        wtr = fwt[None, None] * (self.k_wt[:, None, None, None] * self.a_wt[None, :, None, None])
        el = np.broadcast_to(self.eta_el[None, :, None], (S, M, D)).reshape(-1).copy()
        fc = np.broadcast_to(self.eta_fc[None, :, None], (S, M, D)).reshape(-1).copy()
        return dict(L=np.ascontiguousarray(Lr).reshape(-1, T), pv=np.ascontiguousarray(pvr).reshape(-1, T),
                    wt=np.ascontiguousarray(wtr).reshape(-1, T), eta_el=el, eta_fc=fc,
                    D=D, T=T, w=np.asarray(w, float))


# ----------------------------------------------------------------------
# 5. Dispatch (vectorized over rows = scenario x year x day)
# ----------------------------------------------------------------------
def dispatch_core(x, L, pv_kw, wt_kw, eta_el, eta_fc, E0, p: Params, endonly: bool = False):
    """Hourly priority-stack dispatch for R independent rows of T hours each.
    shortfall: fuel cell -> SMR ramp -> grid (cap) -> unmet ; surplus: electrolyzer -> curtail."""
    C_PV, C_WT, P_SMR, P_EL, C_TK, P_FC = x
    R, T = L.shape
    base = p.smr_base * P_SMR
    cap_gap = max(p.smr_avail * P_SMR - base, 0.0)
    ramp = p.smr_ramp * P_SMR
    Gcap = p.grid_cap_kw
    n_all = L - pv_kw - wt_kw - base
    E = np.asarray(E0, float).copy()
    Sprev = np.full(R, base)
    if not endonly:
        fcA = np.zeros((R, T)); xsA = np.zeros((R, T)); gA = np.zeros((R, T)); uA = np.zeros((R, T))
        elA = np.zeros((R, T)); cuA = np.zeros((R, T)); EA = np.zeros((R, T))
    for h in range(T):
        n = n_all[:, h]
        sh = n > 0.0
        fc = np.where(sh, np.minimum(np.minimum(n, P_FC), eta_fc * E), 0.0)
        n1 = n - fc
        lim = np.minimum(cap_gap, Sprev + ramp - base)
        xs = np.where(sh, np.maximum(0.0, np.minimum(n1, lim)), 0.0)
        n2 = n1 - xs
        g = np.where(sh, np.minimum(np.maximum(n2, 0.0), Gcap), 0.0)
        u = np.where(sh, np.maximum(n2 - g, 0.0), 0.0)
        s = np.where(sh, 0.0, -n)
        room = np.maximum(C_TK - E, 0.0)
        el = np.minimum(np.minimum(s, P_EL), room / eta_el)
        cu = s - el
        E = np.clip(E - fc / eta_fc + eta_el * el, 0.0, C_TK)
        Sprev = base + xs
        if not endonly:
            fcA[:, h] = fc; xsA[:, h] = xs; gA[:, h] = g; uA[:, h] = u
            elA[:, h] = el; cuA[:, h] = cu; EA[:, h] = E
    if endonly:
        return E
    return dict(L=L, pv=pv_kw, wt=wt_kw, smr=base + xsA, fc=fcA, g=gA, u=uA, el=elA, curt=cuA, E=EA, E_end=E)


def cyclic_E0(x, blk, p: Params, iters: int = 22):
    """Tank level at the start of the day such that end level = start level
    (bisection; the end level is a non-decreasing function of the start level)."""
    C_TK = x[4]
    R = blk["L"].shape[0]
    if C_TK <= 0:
        return np.zeros(R)
    pv_kw = x[0] * blk["pv"]; wt_kw = x[1] * blk["wt"]
    lo = np.zeros(R); hi = np.full(R, float(C_TK))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        end = dispatch_core(x, blk["L"], pv_kw, wt_kw, blk["eta_el"], blk["eta_fc"], mid, p, endonly=True)
        pos = (end - mid) > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def aggregate(sim, blk, S, M, p: Params, detail: bool = False) -> dict:
    D, T = blk["D"], blk["T"]
    price = np.tile(p.tariff, T // 24)
    w = blk["w"]

    def red(a):
        return np.einsum("smd,d->sm", a.sum(axis=1).reshape(S, M, D), w)

    out = dict(unm=red(sim["u"]), grid=red(sim["g"]), gridw=red(sim["g"] * price[None, :]),
               smr=red(sim["smr"]), fc=red(sim["fc"]))
    if detail:
        tot = sim["pv"] + sim["wt"] + sim["smr"]
        share = np.divide(sim["curt"], tot, out=np.zeros_like(tot), where=tot > 1e-12)
        out.update(load=red(sim["L"]), pv_gen=red(sim["pv"]), wt_gen=red(sim["wt"]), el=red(sim["el"]),
                   curt=red(sim["curt"]))
        out["pv_used"] = out["pv_gen"] - red(sim["pv"] * share)
        out["wt_used"] = out["wt_gen"] - red(sim["wt"] * share)
        out["smr_used"] = out["smr"] - red(sim["smr"] * share)
    return out


def simulate(x, ctx: Ctx, detail: bool = False, keep: bool = False):
    """Run all blocks for design x. Returns aggregated annual quantities (S, M) and, if keep, raw sims."""
    p = ctx.p
    x = np.asarray(x, float)
    S, M = ctx.S, ctx.M
    sims, E0s = {}, {}
    typ = ctx.blocks["typ"]
    E0_typ = cyclic_E0(x, typ, p)
    E0s["typ"] = E0_typ

    def run(blk, E0):
        return dispatch_core(x, blk["L"], x[0] * blk["pv"], x[1] * blk["wt"], blk["eta_el"], blk["eta_fc"], E0, p)

    sims["typ"] = run(typ, E0_typ)
    E0_t3 = E0_typ.reshape(S, M, typ["D"])
    if "str" in ctx.blocks:
        E0s["str"] = E0_t3[:, :, ctx.bank.par_s].reshape(-1)
        sims["str"] = run(ctx.blocks["str"], E0s["str"])
    if "win" in ctx.blocks:
        E0s["win"] = E0_t3[:, :, ctx.bank.par_w].reshape(-1)
        sims["win"] = run(ctx.blocks["win"], E0s["win"])
    agg = None
    for name, blk in ctx.blocks.items():
        a = aggregate(sims[name], blk, S, M, p, detail)
        agg = a if agg is None else {k: agg[k] + a[k] for k in agg}
    if keep:
        return agg, sims, E0s
    return agg


def annual_series(agg: dict, ctx_or_years, N: int) -> dict:
    years_M = ctx_or_years.years_M if hasattr(ctx_or_years, "years_M") else np.asarray(ctx_or_years)
    yrs = np.arange(1, N + 1)
    return {k: np.vstack([np.interp(yrs, years_M, v[s]) for s in range(v.shape[0])]) for k, v in agg.items()}


# ----------------------------------------------------------------------
# 6. Economics
# ----------------------------------------------------------------------
def _irr(cf: np.ndarray):
    taus = np.arange(len(cf))
    f = lambda rho: float((cf * (1 + rho) ** (-taus)).sum())
    lo, hi = -0.5, 3.0
    flo, fhi = f(lo), f(hi)
    if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def economics(x, ser: dict, base: dict, p: Params, sched: dict) -> dict:
    """ser: dict of (N,) arrays for ONE scenario. base: all-grid baseline dict (N,)."""
    N = p.N
    t = np.arange(1, N + 1)
    taus = np.arange(0, N + 1)
    disc = (1 + p.r) ** (-t.astype(float))
    disc0 = (1 + p.r) ** (-taus.astype(float))
    esc_pi = (1 + p.e_tariff) ** (t - 1)
    esc_f = (1 + p.e_fuel) ** (t - 1)
    xk = dict(zip(TECHS, np.asarray(x, float)))
    capex = {k: p.tech[k]["capex"] * xk[k] for k in TECHS}
    om_k = {k: p.tech[k]["om"] * capex[k] for k in TECHS}
    OM = sum(om_k.values())

    grid_t = ser["gridw"] * esc_pi
    fuel_t = p.smr_fuel * esc_f * ser["smr"]
    unm = ser["unm"]
    B_t = base["grid0w"] * esc_pi + p.voll * base["unm0"]
    delta = B_t - (grid_t + fuel_t + OM + p.voll * unm)

    repl_k = {k: np.zeros(N + 1) for k in TECHS}
    dec_k = {k: np.zeros(N + 1) for k in TECHS}
    rv_k = {}
    for k in TECHS:
        s = sched[k]
        for tau in s["inst"]:
            if tau > 0:
                repl_k[k][tau] += capex[k] * (1 + p.eps_capex[k]) ** tau
        for tau in s["decom"]:
            dec_k[k][tau] += p.tech[k]["decomm"] * capex[k]
        rv_k[k] = s["rho"] * capex[k] * (1 + p.eps_capex[k]) ** s["last"]
    cap_events = sum(repl_k[k] + dec_k[k] for k in TECHS)
    RV = sum(rv_k.values())
    I0 = sum(capex.values())
    pv_events = float((cap_events[1:] * disc0[1:]).sum())
    npv = -I0 - pv_events + RV * disc0[N] + float((delta * disc).sum())

    e_srv = base["load"] - unm
    e_srv0 = base["load"] - base["unm0"]
    num_sys = I0 + pv_events - RV * disc0[N] + float(((grid_t + fuel_t + OM) * disc).sum())
    den_sys = float((e_srv * disc).sum())
    den_grid = float((e_srv0 * disc).sum())
    lcoe_sys = num_sys / den_sys if den_sys > 0 else float("nan")
    lcoe_grid = float((base["grid0w"] * esc_pi * disc).sum()) / den_grid if den_grid > 0 else float("nan")

    cf = np.zeros(N + 1)
    cf[0] = -I0
    cf[1:] += delta
    cf -= cap_events
    cf[N] += RV
    dcf = cf * disc0
    cum = np.cumsum(dcf)
    payback = float("nan")
    if I0 > 0:
        idx = np.where(cum[1:] >= 0)[0]
        if len(idx):
            T1 = int(idx[0]) + 1
            prev = cum[T1 - 1]
            payback = (T1 - 1) + (-prev / dcf[T1] if dcf[T1] != 0 else 0.0)
    irr = _irr(cf) if I0 > 0 else float("nan")

    pvcost = {k: capex[k] + om_k[k] * float(disc.sum()) + float(((repl_k[k] + dec_k[k])[1:] * disc0[1:]).sum())
              - rv_k[k] * disc0[N] for k in TECHS}
    safe = lambda a, b: a / b if b > 1e-9 else float("nan")
    out = dict(
        npv=npv, I0=I0, lcoe_sys=lcoe_sys, lcoe_grid=lcoe_grid, payback=payback, irr=irr,
        eens=unm.copy(), delta=delta, cf=cf, dcf=dcf, cum_dcf=cum, grid_t=grid_t, fuel_t=fuel_t, OM=OM,
        pvcost=pvcost, num_sys=num_sys, den_sys=den_sys, B_t=B_t, RV=RV,
        capital_events=cap_events,
    )
    if "pv_used" in ser:
        out["lcoe_k"] = dict(
            PV=safe(pvcost["PV"], float((ser["pv_used"] * disc).sum())),
            WT=safe(pvcost["WT"], float((ser["wt_used"] * disc).sum())),
            SMR=safe(pvcost["SMR"] + float((fuel_t * disc).sum()), float((ser["smr_used"] * disc).sum())),
            H2_LCOS=safe(pvcost["EL"] + pvcost["TK"] + pvcost["FC"], float((ser["fc"] * disc).sum())),
            GRID_avg_price=safe(float((grid_t * disc).sum()), float((ser["grid"] * disc).sum())),
        )
        out["pv_fuel"] = float((fuel_t * disc).sum()); out["pv_grid"] = float((grid_t * disc).sum())
        out["cost_identity_gap"] = abs(sum(pvcost.values()) + out["pv_fuel"] + out["pv_grid"] - num_sys)
    return out


def evaluate(x, ctx: Ctx, detail: bool = False) -> dict:
    """Screening evaluation on typical + stress days. Returns objectives, violation and (optionally) detail."""
    p = ctx.p
    agg = simulate(x, ctx, detail=detail)
    ser = annual_series(agg, ctx, p.N)
    econ = [economics(x, {k: v[s] for k, v in ser.items()}, ctx.base[s], p, ctx.sched) for s in range(ctx.S)]
    npv = float(sum(ctx.p_s[s] * econ[s]["npv"] for s in range(ctx.S)))
    ci = ctx.T_crit - 1
    eens_worst = float(max(e["eens"][ci].max() for e in econ))
    viol = max(0.0, eens_worst / max(p.eens_target, 1e-9) - 1.0)
    out = dict(npv=npv, eens_worst=eens_worst, viol=viol)
    if detail:
        out.update(econ=econ, ser=ser)
    return out


# ----------------------------------------------------------------------
# 7. Constrained NSGA-II (objectives: -NPV, worst-year EENS; constraint: EENS <= target)
# ----------------------------------------------------------------------
def _dom(Fa, va, Fb, vb, tol=1e-12):
    fa, fb = va <= tol, vb <= tol
    if fa and not fb:
        return True
    if (not fa) and fb:
        return False
    if (not fa) and (not fb):
        return va < vb
    return bool(np.all(Fa <= Fb) and np.any(Fa < Fb))


def nd_sort(F, v):
    n = len(F)
    dom = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            if i != j and _dom(F[i], v[i], F[j], v[j]):
                dom[i, j] = True
    cnt = dom.sum(axis=0).astype(int)
    rank = np.zeros(n, int)
    fronts, cur = [], [i for i in range(n) if cnt[i] == 0]
    r = 0
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            rank[i] = r
            for j in np.where(dom[i])[0]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(int(j))
        cur, r = nxt, r + 1
    return fronts, rank


def crowding(F, front):
    d = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: np.inf for i in front}
    for m in range(F.shape[1]):
        srt = sorted(front, key=lambda i: F[i, m])
        d[srt[0]] = d[srt[-1]] = np.inf
        span = F[srt[-1], m] - F[srt[0], m]
        if span < 1e-12:
            continue
        for a in range(1, len(srt) - 1):
            d[srt[a]] += (F[srt[a + 1], m] - F[srt[a - 1], m]) / span
    return d


def _sbx(rng, p1, p2, lb, ub, eta=15.0, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    for i in range(len(p1)):
        if ub[i] <= lb[i] or rng.random() >= 0.5 or abs(p1[i] - p2[i]) <= 1e-12:
            continue
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        r = rng.random()

        def betaq(beta):
            alpha = 2.0 - beta ** (-(eta + 1.0))
            return (r * alpha) ** (1 / (eta + 1)) if r <= 1.0 / alpha else (1.0 / (2.0 - r * alpha)) ** (1 / (eta + 1))

        q1 = betaq(1.0 + 2.0 * (y1 - lb[i]) / (y2 - y1))
        q2 = betaq(1.0 + 2.0 * (ub[i] - y2) / (y2 - y1))
        a = np.clip(0.5 * ((y1 + y2) - q1 * (y2 - y1)), lb[i], ub[i])
        b = np.clip(0.5 * ((y1 + y2) + q2 * (y2 - y1)), lb[i], ub[i])
        if rng.random() < 0.5:
            a, b = b, a
        c1[i], c2[i] = a, b
    return c1, c2


def _mutate(rng, x, lb, ub, eta=20.0, pm=0.2):
    y = x.copy()
    for i in range(len(x)):
        if ub[i] <= lb[i] or rng.random() >= pm:
            continue
        rngw = ub[i] - lb[i]
        d1, d2 = (y[i] - lb[i]) / rngw, (ub[i] - y[i]) / rngw
        u, mp = rng.random(), 1.0 / (eta + 1.0)
        if u < 0.5:
            dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** mp - 1.0
        else:
            dq = 1.0 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** mp
        y[i] = np.clip(y[i] + dq * rngw, lb[i], ub[i])
    return y


def _snap(x, ub, frac=0.005):
    """Capacities below 0.5% of the site maximum are treated as zero."""
    y = x.copy()
    y[y < frac * np.maximum(ub, 1e-12)] = 0.0
    return y


def run_nsga2(ctx: Ctx, lb, ub, pop_size=30, generations=25, seed=0, progress_cb=None):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    n = len(lb)

    def ev(x):
        r = evaluate(x, ctx)
        return np.array([-r["npv"], r["eens_worst"]]), r["viol"]

    pop = [lb + rng.random(n) * (ub - lb) for _ in range(pop_size)]
    pop[0] = lb.copy()                                   # build nothing
    if pop_size > 2:
        pop[1] = ub.copy()                               # build everything
        pop[2] = np.array([ub[0], 0, 0, 0, 0, 0], float) # PV only
    pop = np.array([_snap(x, ub) for x in pop])
    res = [ev(x) for x in pop]
    F = np.array([r[0] for r in res]); v = np.array([r[1] for r in res])
    history = []
    for gen in range(generations):
        fronts, rank = nd_sort(F, v)
        crowd = {}
        for f in fronts:
            crowd.update(crowding(F, f))

        def tour():
            a, b = rng.integers(0, len(pop), 2)
            if rank[a] != rank[b]:
                return pop[a if rank[a] < rank[b] else b].copy()
            return pop[a if crowd.get(a, 0) >= crowd.get(b, 0) else b].copy()

        kids = []
        while len(kids) < pop_size:
            c1, c2 = _sbx(rng, tour(), tour(), lb, ub)
            kids += [_snap(_mutate(rng, c1, lb, ub), ub), _snap(_mutate(rng, c2, lb, ub), ub)]
        kids = np.array(kids[:pop_size])
        kres = [ev(x) for x in kids]
        kF = np.array([r[0] for r in kres]); kv = np.array([r[1] for r in kres])
        P2 = np.vstack([pop, kids]); F2 = np.vstack([F, kF]); v2 = np.concatenate([v, kv])
        fronts, rank2 = nd_sort(F2, v2)
        keep = []
        for f in fronts:
            if len(keep) + len(f) <= pop_size:
                keep += f
            else:
                cd = crowding(F2, f)
                keep += sorted(f, key=lambda i: -cd[i])[: pop_size - len(keep)]
                break
        pop, F, v = P2[keep], F2[keep], v2[keep]
        feas = v <= 1e-12
        history.append(dict(gen=gen + 1, best_npv=float(-F[feas, 0].min()) if feas.any() else float("nan"),
                            min_eens=float(F[:, 1].min()), n_feasible=int(feas.sum()),
                            min_violation=float(v.min())))
        if progress_cb:
            progress_cb(gen + 1, generations)
    fronts, _ = nd_sort(F, v)
    front = fronts[0]
    seen, keep = set(), []
    for i in front:                                   # drop duplicate designs
        key = tuple(np.round(pop[i], 1))
        if key not in seen:
            seen.add(key); keep.append(i)
    return dict(X=pop[keep], F=F[keep], v=v[keep], history=history, all_X=pop, all_F=F, all_v=v)


def select_recommended(res: dict) -> tuple[int, bool]:
    feas = np.where(res["v"] <= 1e-12)[0]
    if len(feas):
        return int(feas[np.argmin(res["F"][feas, 0])]), True       # max NPV among feasible
    return int(np.argmin(res["v"])), False


# ----------------------------------------------------------------------
# 8. Verification on the full 8760-hour record, chronological tank
# ----------------------------------------------------------------------
def verify_design(x, ctx: Ctx) -> dict:
    p, ver = ctx.p, ctx.bank.ver
    x = np.asarray(x, float)
    ms = ctx.years_M
    Mv = len(ms)
    ib = ctx.i_base
    Lr = ver["L"][None, :] * ctx.load_scale[ib][:, None]
    pv_kw = x[0] * ver["pv"][None, :] * ctx.a_pv[:, None]
    wt_kw = x[1] * ver["wt"][None, :] * ctx.a_wt[:, None]
    E0 = np.full(Mv, 0.5 * x[4])
    e1 = dispatch_core(x, Lr, pv_kw, wt_kw, ctx.eta_el, ctx.eta_fc, E0, p, endonly=True)   # warm-up pass
    sim = dispatch_core(x, Lr, pv_kw, wt_kw, ctx.eta_el, ctx.eta_fc, e1, p)
    blk = dict(D=1, T=Lr.shape[1], w=np.array([ver["scale"]]))
    agg = aggregate(sim, blk, 1, Mv, p, detail=True)
    ser = annual_series(agg, ms, p.N)
    base = baseline([(ver["L"][None, :], np.array([ver["scale"]]))], p.replace(g_load=float(ctx.g_s[ib])))
    e = economics(x, {k: v[0] for k, v in ser.items()}, base, p, ctx.sched)
    ci = ctx.T_crit - 1
    return dict(npv=e["npv"], eens=e["eens"], eens_worst=float(e["eens"][ci].max()),
                lcoe_sys=e["lcoe_sys"], lcoe_grid=e["lcoe_grid"], econ=e, label=ver["label"],
                passed=bool(e["eens"][ci].max() <= p.eens_target))


def check_all_years(x, bank: DayBank, p: Params) -> dict:
    """Simulate all N years (screening layer) and compare the worst year with the critical-year maximum."""
    ctxf = Ctx(bank, p, milestones=range(1, p.N + 1))
    agg = simulate(x, ctxf)
    ser = annual_series(agg, ctxf, p.N)
    eens_all = ser["unm"]                         # (S, N)
    crit = ctxf.T_crit - 1
    worst_all = float(eens_all.max()); worst_crit = float(eens_all[:, crit].max())
    return dict(eens_all=eens_all, worst_all=worst_all, worst_crit=worst_crit,
                holds=bool(worst_all <= worst_crit * (1 + 1e-6) + 1e-6))


def full_report(x, bank: DayBank, p: Params) -> dict:
    """Detailed economics for the chosen design at every year (milestones = all years)."""
    ctxf = Ctx(bank, p, milestones=range(1, p.N + 1))
    r = evaluate(x, ctxf, detail=True)
    return dict(ctx=ctxf, i_base=ctxf.i_base, **r)


def day_dispatch(x, bank: DayBank, p: Params, year: int, scen: int | None = None):
    """Hourly arrays for every representative day / stress day in one year (for plotting)."""
    ctx1 = Ctx(bank, p, milestones=[year])
    scen = ctx1.i_base if scen is None else scen
    agg, sims, E0s = simulate(x, ctx1, keep=True)
    out = {}
    for name, sim in sims.items():
        D = ctx1.blocks[name]["D"]
        out[name] = {k: sim[k].reshape(ctx1.S, 1, D, -1)[scen, 0] for k in
                     ("L", "pv", "wt", "smr", "fc", "g", "u", "el", "curt", "E")}
        out[name]["E0"] = E0s[name].reshape(ctx1.S, 1, D)[scen, 0]
    return out


def sensitivity(x, bank: DayBank, p: Params, cases: dict) -> pd.DataFrame:
    """NPV of the fixed design under low/high values of single parameters (or groups)."""
    rows = []
    base_npv = evaluate(x, Ctx(bank, p))["npv"]
    for name, (lo_kw, hi_kw) in cases.items():
        vals = []
        for kw in (lo_kw, hi_kw):
            vals.append(evaluate(x, Ctx(bank, p.replace(**kw)))["npv"])
        rows.append(dict(case=name, npv_low=vals[0], npv_high=vals[1], npv_base=base_npv))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 9. Baseline / index summary (the forecasting framework's decomposition, used to show variability)
# ----------------------------------------------------------------------
def index_summary(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """value = baseline(month, day type, hour) x index. Returns, by month, the mean and spread of the daily-mean index."""
    d = pd.DataFrame(dict(load=df["load_kw"].values, pv=df["ghi"].values / 1000.0 * p.pr,
                          wind=wind_power_fraction(df["wind_speed"].values, p)), index=df.index)
    d["m"] = d.index.month; d["q"] = (d.index.dayofweek >= 5).astype(int); d["h"] = d.index.hour
    rows = {}
    for col in ("load", "pv", "wind"):
        base = d.groupby(["m", "q", "h"])[col].transform("mean")
        idx = (d[col] / base.where(base > 1e-9)).fillna(1.0)
        daily = idx.groupby([d.index.year, d.index.month, d.index.day]).mean()
        mm = daily.groupby(level=1).agg(["mean", "std"])
        rows[f"{col} index mean"] = mm["mean"]; rows[f"{col} index std"] = mm["std"]
    out = pd.DataFrame(rows); out.index.name = "month"
    return out.round(3)
