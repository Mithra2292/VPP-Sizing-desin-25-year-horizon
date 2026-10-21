import time, numpy as np, vpp_backend_v4 as B
g = B.load_series("/root/solar_2025_fixed.csv","ghi"); w = B.load_series("/root/wind_2025_fixed.csv","wind_speed"); l = B.load_series("/root/load_2025_fixed.csv","load_kw")
df,_ = B.align_inputs(g,w,l)
ub = [3000,1000,1000,0,0,0]
p = B.Params(g_cases=[(0.02,.25),(0.03,.5),(0.045,.25)], active=("PV","WT","SMR"))
bank = B.build_bank(df,p,1500,500); ctx = B.Ctx(bank,p)
print("S",ctx.S,"i_base",ctx.i_base,"T_crit",ctx.T_crit,"milestones",ctx.years_M, "probs",ctx.p_s.round(3))
x = np.array([1000,0,0,0,0,0.])
t=time.time(); r=B.evaluate(x,ctx,detail=True); print("eval",round(time.time()-t,3),"npv",r["npv"])
e=r["econ"][ctx.i_base]; print("lcoe",e["lcoe_sys"],e["lcoe_grid"],"gap",e["cost_identity_gap"])
res=B.run_nsga2(ctx,[0]*6,ub,30,25,0); i,ok=B.select_recommended(res); print("rec",np.round(res["X"][i]),ok,-res["F"][i,0])
print("verify",B.verify_design(res["X"][i],ctx)["npv"])
print(B.check_all_years(res["X"][i],bank,p)["holds"])
print(B.index_summary(df,p).head(3))
# tight grid so reliability binds
p2=p.replace(grid_cap_kw=420.0,eens_target=30000.0); c2=B.Ctx(bank,p2)
res=B.run_nsga2(c2,[0]*6,ub,30,25,0); i,ok=B.select_recommended(res); print("tight rec",np.round(res["X"][i]),ok,res["F"][i,1])
v=B.verify_design(res["X"][i],c2); print("tight verify eens",v["eens_worst"],v["passed"])
