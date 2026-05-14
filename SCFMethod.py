# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:44:48 2026

@author: yasmin
"""
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import sys
import math

#%0 GAP 23 OTELLE İLE ÇALIŞAN KOD

# ==========================================
# 1. PARAMETRELER VE VERİ OKUMA
# ==========================================
T_start, a0, b0 = 0, 240, 300
P, hiz, s_h = 10, 60, 15

dosya = "camasirhane_veri.xlsx"
try:
    df_t = pd.read_excel(dosya, sheet_name="Talepler")
    df_dist = pd.read_excel(dosya, sheet_name="Mesafeler", index_col=0)
    df_dist.columns = df_dist.columns.astype(int)
    df_dist.index   = df_dist.index.astype(int)
except Exception as e:
    sys.exit(f"❌ Excel hatası: {e}")

H = df_t["Musteri_ID"].tolist()
N = [0] + H
K = [1, 2]
D = df_t.columns[1:].tolist()

q_hd  = {(h, d): df_t.loc[df_t["Musteri_ID"] == h, d].values[0] for h in H for d in D}
t_ij  = {(i, j): (df_dist.loc[i, j] / hiz) * 60 for i in N for j in N}
c_ij  = {(i, j):  df_dist.loc[i, j]             for i in N for j in N}

# ==========================================
# 2. GREEDY WARM START
# ==========================================
def greedy_warm_start(H_aktif):
    unvisited = set(H_aktif)
    routes = {k: [] for k in K}
    for k in K:
        if not unvisited:
            break
        curr, time_spent = 0, T_start
        while unvisited:
            best, best_cost = None, float('inf')
            for j in unvisited:
                arr = time_spent + t_ij[curr, j]
                if arr + s_h + t_ij[j, 0] <= b0 and c_ij[curr, j] < best_cost:
                    best, best_cost = j, c_ij[curr, j]
            if best is None:
                break
            routes[k].append(best)
            time_spent += t_ij[curr, best] + s_h
            curr = best
            unvisited.discard(best)
    route_x, route_z = {}, {}
    for k in K:
        seq  = routes[k]
        path = [0] + seq + [0]
        for idx in range(len(path) - 1):
            route_x[(path[idx], path[idx+1], k)] = 1
        for h in seq:
            route_z[(h, k)] = 1
    return route_x, route_z

# ==========================================
# 3. ANA MODEL
# ==========================================
def solve_daily_vrp(gun):
    print(f"\n{'='*60}")
    print(f">>> {gun.upper()} — Two-Commodity Flow Formulation")
    print(f"{'='*60}")

    H_aktif = [h for h in H if q_hd[h, gun] > 0]
    N_aktif = [0] + H_aktif
    n       = len(H_aktif)
    print(f"📌 Aktif otel: {n} | Araç: {len(K)}")

    model = gp.Model(f"VRP_2CF_{gun}")
    model.setParam('OutputFlag', 1)

    # =================================================================
    # DEĞİŞKENLER
    # =================================================================
    x = model.addVars(N_aktif, N_aktif, K, vtype=GRB.BINARY, name="x")
    z = model.addVars(H_aktif, K, vtype=GRB.BINARY, name="z")

    f = model.addVars(N_aktif, N_aktif, K,
                      vtype=GRB.CONTINUOUS, lb=0, ub=n, name="f")

    t_otel = model.addVars(H_aktif, K, vtype=GRB.CONTINUOUS, lb=0, name="t_otel")
    for k in K:
        for j in H_aktif:
            t_otel[j, k].lb = t_ij[0, j]

    t_depo = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0, ub=b0, name="t_depo")
    delay  = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0,        name="delay")

    # =================================================================
    # AMAÇ FONKSİYONU
    # =================================================================
    model.setObjective(
        gp.quicksum(c_ij[i, j] * x[i, j, k]
                    for i in N_aktif for j in N_aktif for k in K)
        + gp.quicksum(P * delay[k] for k in K),
        GRB.MINIMIZE
    )

    # =================================================================
    # TEMEL ROTALAMA KISITLARI
    # =================================================================
    for k in K:
        model.addConstr(gp.quicksum(x[0, j, k] for j in H_aktif) == 1, f"Dep_out_{k}")
        model.addConstr(gp.quicksum(x[i, 0, k] for i in H_aktif) == 1, f"Dep_in_{k}")
        for i in N_aktif:
            model.addConstr(x[i, i, k] == 0, f"NoSelf_{i}_{k}")

    for h in H_aktif:
        model.addConstr(gp.quicksum(z[h, k] for k in K) == 1, f"Visit_{h}")

    for k in K:
        for j in H_aktif:
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in N_aktif if i != j) == z[j, k],
                f"FlowIn_{j}_{k}")
            model.addConstr(
                gp.quicksum(x[j, i, k] for i in N_aktif if i != j) == z[j, k],
                f"FlowOut_{j}_{k}")

    # =================================================================
    # İKİ-MALLAR AKIŞ KISITLARI (subtour elimination)
    # =================================================================
    for k in K:
        model.addConstr(
            gp.quicksum(f[0, j, k] for j in H_aktif) ==
            gp.quicksum(z[h, k] for h in H_aktif),
            f"SourceFlow_{k}")

        for j in H_aktif:
            model.addConstr(
                gp.quicksum(f[i, j, k] for i in N_aktif if i != j) -
                gp.quicksum(f[j, i, k] for i in N_aktif if i != j)
                == z[j, k],
                f"FlowBalance_{j}_{k}")

            for i in N_aktif:
                if i != j:
                    model.addConstr(f[i, j, k] <= n * x[i, j, k], f"FlowBound_{i}_{j}_{k}")
                    model.addConstr(f[j, i, k] <= n * x[j, i, k], f"FlowBound2_{j}_{i}_{k}")

    # =================================================================
    # ZAMAN AKIŞI KISITLARI
    # =================================================================
    for k in K:
        for j in H_aktif:
            model.addConstr(
                t_otel[j, k] >= T_start + t_ij[0, j] - b0 * (1 - x[0, j, k]),
                f"FirstArr_{j}_{k}")
            for i in H_aktif:
                if i != j:
                    M_ij = b0 + s_h + t_ij[i, j] - t_ij[0, j]
                    
                    model.addConstr(
                        t_otel[j, k] >= t_otel[i, k] + s_h + t_ij[i, j]
                        - M_ij * (1 - x[i, j, k]),
                        f"Trans_{i}_{j}_{k}")
            model.addConstr(
                t_depo[k] >= t_otel[j, k] + s_h + t_ij[j, 0] - b0 * (1 - x[j, 0, k]),
                f"Return_{j}_{k}")
    for k in K:
        model.addConstr(delay[k] >= t_depo[k] - a0, f"Delay_{k}")

    # =================================================================
    # GÜÇLENDİRİCİ KISITLAR
    # =================================================================
    # 2-cycle elimination
    for k in K:
        for i in H_aktif:
            for j in H_aktif:
                if i < j:
                    model.addConstr(x[i,j,k] + x[j,i,k] <= 1, f"2Cyc_{i}_{j}_{k}")

    # Sparsification
    for k in K:
        for i in H_aktif:
            for j in H_aktif:
                if i != j:
                    if t_ij[0,i] + s_h + t_ij[i,j] + s_h + t_ij[j,0] > b0:
                        model.addConstr(x[i,j,k] == 0, f"TimeOut_{i}_{j}_{k}")

    # Yük dengesi
    max_stops = math.ceil(n / len(K)) + 1
    for k in K:
        model.addConstr(
            gp.quicksum(z[h, k] for h in H_aktif) <= max_stops,
            f"MaxStops_{k}")


    for k in range(1, len(K)):
        model.addConstr(
            gp.quicksum(z[h, k]   for h in H_aktif) >=
            gp.quicksum(z[h, k+1] for h in H_aktif),
            f"Simetri_{k}")

    # =================================================================
    # WARM START
    # =================================================================
    route_x, route_z = greedy_warm_start(H_aktif)
    for k in K:
        for i in N_aktif:
            for j in N_aktif:
                x[i, j, k].Start = route_x.get((i, j, k), 0)
        for h in H_aktif:
            z[h, k].Start = route_z.get((h, k), 0)

    # =================================================================
    # GUROBI PARAMETRELERİ
    # =================================================================


    model.setParam('MIPFocus',         1)
    model.setParam('Cuts',             2)
    model.setParam('FlowCoverCuts',    2)
    model.setParam('FlowPathCuts',     2)
    model.setParam('GomoryPasses',     3)
    model.setParam('Heuristics',       0.5)
    model.setParam('RINS',             10)
    model.setParam('SubMIPNodes',      300)
    model.setParam('Presolve',         2)
    model.setParam('PreSparsify',      1)
    model.setParam('MIPGap', 0.001)   
    model.setParam('TimeLimit', 1800) 
    model.optimize()

    # =================================================================
    # SONUÇ
    # =================================================================
    if model.SolCount > 0:
        print(f"\n✅ {gun} Günü Özeti:")
        print(f"   Maliyet: {model.objVal:.2f} | Gap: %{model.MIPGap*100:.2f}")
        for k in K:
            curr, rota = 0, "Depo(0)"
            while True:
                nxt = next(
                    (j for j in N_aktif if curr != j and x[curr, j, k].x > 0.5), 0)
                if nxt == 0:
                    rota += f" -> Depo(0) [{t_depo[k].x:.1f} dk]"
                    break
                rota += f" -> Otel {nxt} [{t_otel[nxt,k].x:.1f} dk]"
                curr = nxt
            print(f"   Araç {k}: {rota}")
    else:
        print(f"❌ {gun} için çözüm bulunamadı.")

# ==========================================
# ÇALIŞTIRMA (TÜM HAFTA İÇİN PERİYODİK ÇÖZÜM)
# ==========================================
print("\n" + "🚀"*15)
print("HAFTALIK PERİYODİK ÇÖZÜM BAŞLATILIYOR")
print("🚀"*15)

for gun in D:
    gunluk_aktif_otel = [h for h in H if q_hd[h, gun] > 0]
    if len(gunluk_aktif_otel) == 0:
        print(f"\nℹ️ {gun} günü için hiçbir otelde talep yok. Pas geçiliyor...")
        continue
    solve_daily_vrp(gun)
    print("\n" + "*"*60 + "\n")

print(f"🎉 TÜM HAFTANIN OPTİMİZASYONU TAMAMLANDI!")
