"""Streamlit app: VPP lifetime sizing (NSGA-II, NPV, EENS every year). Run: streamlit run app_v3.py"""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import vpp_backend_v4 as B

st.set_page_config(page_title="VPP Lifetime Sizing", layout="wide")
st.title("VPP lifetime sizing: PV + wind + SMR + grid (25-year)")
st.caption("Constant rupees, real discount rate. Fuel cell / hydrogen, DSM, EV and Shapley allocation are paused.")

# ---------------- sidebar ----------------
sb = st.sidebar
sb.header("Data (hourly CSVs)")
f_s = sb.file_uploader("Solar: timestamp, ghi (W/m2)", type="csv")
f_w = sb.file_uploader("Wind: timestamp, wind_speed (m/s)", type="csv")
f_l = sb.file_uploader("Load: timestamp, load_kw", type="csv")
sb.header("Site")
z_ref = sb.number_input("Wind measurement height (m)", 1.0, 200.0, 50.0)
z_hub = sb.number_input("Hub height (m)", 1.0, 200.0, 50.0)
shear = sb.number_input("Shear exponent", 0.0, 0.6, 0.14)
use_h2 = sb.checkbox("Include hydrogen chain (needs fuel cell)", value=False)
sb.subheader("Maximum capacities")
ub = [sb.number_input("Max PV (kWp)", 0.0, 1e6, 3000.0),
      sb.number_input("Max wind (kW)", 0.0, 1e6, 1000.0),
      sb.number_input("Max SMR (kW)", 0.0, 1e6, 1000.0)]
if use_h2:
    ub += [sb.number_input("Max electrolyzer (kW)", 0.0, 1e6, 1500.0),
           sb.number_input("Max H2 tank (kWh-eq)", 0.0, 1e7, 20000.0),
           sb.number_input("Max fuel cell (kW)", 0.0, 1e6, 1000.0)]
else:
    ub += [0.0, 0.0, 0.0]
active = tuple(k for k, u in zip(B.TECHS, ub) if u > 0)
sb.header("Economics and reliability")
grid_cap = sb.number_input("Grid import cap (kW)", 0.0, 1e6, 1662.5)
r = sb.number_input("Real discount rate", 0.0, 0.5, 0.08, 0.005)
voll = sb.number_input("Value of lost load (Rs/kWh)", 0.0, 1000.0, 120.0)
eens_t = sb.number_input("EENS target (kWh/yr)", 0.0, 1e9, 50000.0, 1000.0)
N = sb.number_input("Horizon N (years)", 5, 40, 25)
nu = sb.number_input("Replacement threshold nu", 0.0, 1.0, 0.5, 0.05)
with sb.expander("Scenarios (load growth, low-resource year)"):
    gl = st.number_input("Low load growth /yr", 0.0, 0.2, 0.02, 0.005)
    gb = st.number_input("Base load growth /yr", 0.0, 0.2, 0.03, 0.005)
    gh = st.number_input("High load growth /yr", 0.0, 0.2, 0.045, 0.005)
    pg_low = st.number_input("Probability low", 0.0, 1.0, 0.25, 0.05)
    pg_high = st.number_input("Probability high", 0.0, 1.0, 0.25, 0.05)
    pv_p10 = st.number_input("Low-resource PV factor", 0.5, 1.0, 0.96, 0.01)
    wt_p10 = st.number_input("Low-resource wind factor", 0.3, 1.0, 0.90, 0.01)
    st.caption("Factors are used only when the data has fewer than 3 full years; otherwise the P10 year is computed from the data.")
with sb.expander("Lifetime assumptions (placeholders)"):
    g_load = gb
    d_pv = st.number_input("PV degradation /yr", 0.0, 0.05, 0.005, 0.001, format="%.3f")
    d_wt = st.number_input("Wind degradation /yr", 0.0, 0.05, 0.003, 0.001, format="%.3f")
    d_el = st.number_input("Electrolyzer efficiency loss /yr", 0.0, 0.05, 0.008, 0.001, format="%.3f")
    d_fc = st.number_input("Fuel cell efficiency loss /yr", 0.0, 0.05, 0.008, 0.001, format="%.3f")
    e_pi = st.number_input("Tariff escalation /yr", -0.05, 0.2, 0.01, 0.005)
    e_f = st.number_input("Fuel escalation /yr", -0.05, 0.2, 0.005, 0.005)
    ec_el = st.number_input("Electrolyzer capex trend /yr", -0.2, 0.1, -0.04, 0.01)
    ec_fc = st.number_input("Fuel cell capex trend /yr", -0.2, 0.1, -0.03, 0.01)
with sb.expander("Technology table"):
    tdf = pd.DataFrame(B.DEFAULT_TECH).T
    tdf.index.name = "tech"
    tdf = st.data_editor(tdf, key="tech")
with sb.expander("Stress days and optimizer"):
    K = st.number_input("Stress days K", 0, 15, 5)
    W = st.number_input("Stress window W (days)", 0, 10, 5)
    pop = st.number_input("Population", 10, 200, 30)
    gens = st.number_input("Generations", 3, 200, 25)
    seed = st.number_input("Seed", 0, 10**6, 0)
    nver = st.number_input("Designs to verify (8760 h)", 1, 15, 8)

tech = {k: dict(tdf.loc[k]) for k in B.TECHS}
for k in tech:
    tech[k]["life"] = int(tech[k]["life"])
p = B.Params(N=int(N), r=r, nu=nu, voll=voll, eens_target=eens_t, grid_cap_kw=grid_cap, g_load=g_load,
             g_cases=[(gl, pg_low), (gb, max(0.0, 1 - pg_low - pg_high)), (gh, pg_high)],
             pv_p10=pv_p10, wt_p10=wt_p10, active=active,
             d_pv=d_pv, d_wt=d_wt, d_el=d_el, d_fc=d_fc, e_tariff=e_pi, e_fuel=e_f,
             eps_capex=dict(PV=0, WT=0, SMR=0, EL=ec_el, TK=0, FC=ec_fc), z_ref=z_ref, z_hub=z_hub,
             shear=shear, tech=tech)

# ---------------- data ----------------
try:
    if f_s and f_w and f_l:
        df, info = B.align_inputs(B.load_series(f_s, "ghi"), B.load_series(f_w, "wind_speed"),
                                  B.load_series(f_l, "load_kw"))
        st.success(f"Using your CSVs: {info['n_days']} days, {info['start']} to {info['end']}.")
    else:
        df = B.synthetic_year()
        st.warning("Upload all three CSVs to use your data. Showing SYNTHETIC demo data; results are not for your site.")
except Exception as e:
    st.error(f"CSV problem: {e}")
    st.stop()

ref = (0.5 * ub[0], 0.5 * ub[1])
bank = B.build_bank(df, p, ref[0], ref[1], K=int(K), W=int(W))
ctx = B.Ctx(bank, p)
sched = ctx.sched
if grid_cap > 1.0 * df.load_kw.max() * (1 + g_load) ** (int(N) - 1):
    st.info("The grid cap exceeds the peak load even after growth, so EENS is zero for every design and "
            "reliability will not limit the design. Lower the grid cap to test reliability.")

tabs = st.tabs(["Data & days", "Optimization", "Recommended design", "Reliability by year",
                "Dispatch", "Verification (8760)", "Sensitivity"])

with tabs[0]:
    st.subheader("Raw hourly data")
    st.line_chart(df[["load_kw"]].resample("D").mean())
    c1, c2 = st.columns(2)
    c1.metric("Days in record", bank.n_days)
    c2.metric("Replacement years / critical years", f"{sched['rep_times']} / {sched['T_crit']}")
    st.write("Variability by month: value = baseline(month, day type, hour) x index. Std of the daily-mean index.")
    st.dataframe(B.index_summary(df, p), width="stretch")
    st.write("Typical days (24 groups, weights = days per year)")
    st.dataframe(bank.info_t, width="stretch")
    st.write("Stress days")
    st.dataframe(bank.info_s, width="stretch")
    if bank.info_w:
        st.write("Stress window", bank.info_w)
    st.caption(f"Weights sum to {bank.w_t.sum() + bank.w_s.sum() + (bank.info_w.get('days', 0) * bank.w_w.sum() if bank.w_w is not None else 0):.1f} days.")

with tabs[1]:
    st.write("Objectives: maximize NPV and minimize worst-year EENS, subject to EENS <= target in every critical year.")
    if st.button("Run NSGA-II", type="primary"):
        bar = st.progress(0.0)
        res = B.run_nsga2(ctx, [0] * 6, ub, int(pop), int(gens), int(seed),
                          progress_cb=lambda g, n: bar.progress(g / n))
        st.session_state["res"] = res
        st.session_state["ver"] = None
    res = st.session_state.get("res")
    if res:
        h = pd.DataFrame(res["history"]).set_index("gen")
        c1, c2 = st.columns(2)
        c1.write("Best feasible NPV (Rs)"); c1.line_chart(h["best_npv"])
        c2.write("Feasible designs in population"); c2.line_chart(h["n_feasible"])
        P = pd.DataFrame(res["X"], columns=B.TECHS)
        P["NPV_Rs"] = -res["F"][:, 0]; P["EENS_worst"] = res["F"][:, 1]; P["feasible"] = res["v"] <= 1e-12
        fig = go.Figure(go.Scatter(x=P["EENS_worst"], y=P["NPV_Rs"] / 1e7, mode="markers"))
        fig.update_layout(xaxis_title="Worst-year EENS (kWh)", yaxis_title="NPV (Rs crore)", height=350)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(P.round(1), width="stretch")
    else:
        st.info("Press Run NSGA-II.")

res = st.session_state.get("res")
if not res:
    for t in tabs[2:]:
        with t:
            st.info("Run the optimization first.")
    st.stop()

i, ok = B.select_recommended(res)
x = res["X"][i]
rep = B.full_report(x, bank, p)
ib = rep["i_base"]
e = rep["econ"][ib]

with tabs[2]:
    if not ok:
        st.error("No design meets the EENS target. Showing the least-violating design.")
    st.dataframe(pd.DataFrame({"Capacity": x}, index=[B.TECH_LABELS[k] for k in B.TECHS]).loc[[B.TECH_LABELS[k] for k in active]].round(1))
    a, b, c, d = st.columns(4)
    a.metric("NPV (Rs crore)", f"{e['npv'] / 1e7:.2f}")
    b.metric("System LCOE (Rs/kWh)", f"{e['lcoe_sys']:.2f}")
    c.metric("Grid-only LCOE (Rs/kWh)", f"{e['lcoe_grid']:.2f}")
    d.metric("Payback (yr) / IRR", f"{e['payback']:.1f} / {e['irr'] * 100:.1f}%" if np.isfinite(e["payback"]) else
             f"none / {e['irr'] * 100:.1f}%" if np.isfinite(e["irr"]) else "none")
    st.write("Per-technology diagnostic LCOE (Rs/kWh)")
    if "lcoe_k" in e:
        st.dataframe(pd.Series(e["lcoe_k"], name="Rs/kWh").round(2).to_frame())
    yrs = np.arange(0, p.N + 1)
    f1 = go.Figure(); f1.add_bar(x=yrs, y=e["cf"] / 1e7, name="Cash flow")
    f1.add_scatter(x=yrs, y=e["cum_dcf"] / 1e7, name="Cumulative discounted")
    f1.update_layout(yaxis_title="Rs crore", xaxis_title="Year", height=350)
    st.plotly_chart(f1, width="stretch")
    cost = {**{k: v for k, v in e["pvcost"].items()}, "Fuel": e["pv_fuel"], "Grid energy": e["pv_grid"]}
    st.write("Discounted lifetime cost (Rs crore)")
    st.bar_chart(pd.Series(cost).div(1e7))

with tabs[3]:
    ser_all = rep["ser"]["unm"]
    yrs = np.arange(1, p.N + 1)
    f2 = go.Figure()
    for si in range(ser_all.shape[0]):
        f2.add_scatter(x=yrs, y=ser_all[si], name=ctx.combos[si]["name"], line=dict(width=3 if si == ib else 1))
    f2.add_hline(y=p.eens_target, line_dash="dash", annotation_text="target")
    for t in sched["T_crit"]:
        f2.add_vline(x=t, line_dash="dot")
    f2.update_layout(xaxis_title="Year", yaxis_title="kWh/yr", height=350)
    st.plotly_chart(f2, width="stretch")
    chk = B.check_all_years(x, bank, p)
    st.write("Worst of all 25 years = worst critical year:", chk["holds"])

with tabs[4]:
    yr = st.slider("Year", 1, p.N, 1)
    dd = B.day_dispatch(x, bank, p, yr)
    kind = st.selectbox("Block", list(dd.keys()))
    n = dd[kind]["L"].shape[0]
    k = st.slider("Day index", 0, n - 1, 0)
    d_ = dd[kind]
    fig = go.Figure()
    for key in ("pv", "wt", "smr", "fc", "g", "u"):
        fig.add_bar(x=np.arange(d_["L"].shape[1]), y=d_[key][k], name=key)
    fig.add_scatter(x=np.arange(d_["L"].shape[1]), y=d_["L"][k], name="load", line=dict(color="black"))
    fig.update_layout(barmode="stack", height=380, xaxis_title="Hour")
    st.plotly_chart(fig, width="stretch")
    st.line_chart(pd.Series(d_["E"][k], name="tank kWh"))

with tabs[5]:
    st.write("Chronological 8760-hour check of the Pareto designs")
    order = np.argsort(-(-res["F"][:, 0]))[: int(nver)]
    rows = []
    for j in order:
        v = B.verify_design(res["X"][j], ctx)
        rows.append(dict(**{k: round(res["X"][j][n], 0) for n, k in enumerate(B.TECHS)},
                         NPV_screen=-res["F"][j, 0], NPV_8760=v["npv"], EENS_8760=v["eens_worst"],
                         verdict="feasible" if v["passed"] and v["npv"] > 0 else "fails"))
    st.caption(f"Verification year: {ctx.bank.ver['label']}")
    st.dataframe(pd.DataFrame(rows), width="stretch")

with tabs[6]:
    cases = {"Load growth 1% / 5%": (dict(g_cases=[(0.01, 1.0)]), dict(g_cases=[(0.05, 1.0)])),
             "Discount rate 6% / 10%": (dict(r=0.06), dict(r=0.10)),
             "Tariff escalation 0% / 3%": (dict(e_tariff=0.0), dict(e_tariff=0.03)),
             "PV degradation 0.2% / 1%": (dict(d_pv=0.002), dict(d_pv=0.01))}
    sdf = B.sensitivity(x, bank, p, cases)
    base_npv = sdf["npv_base"].iloc[0]
    fig = go.Figure()
    fig.add_bar(y=sdf["case"], x=(sdf["npv_low"] - base_npv) / 1e7, orientation="h", name="low")
    fig.add_bar(y=sdf["case"], x=(sdf["npv_high"] - base_npv) / 1e7, orientation="h", name="high")
    fig.update_layout(barmode="overlay", xaxis_title="Change in NPV (Rs crore)", height=350)
    st.plotly_chart(fig, width="stretch")
