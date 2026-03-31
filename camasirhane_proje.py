# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# 1. PARAMETRELER
T_start = 480       # 08:00
a0 = 720            # 12:00
b0 = 780            # 13:00
P = 10               # Ceza katsayısı
M = 10000           # Big M
hiz = 60            # 60 km/h (1 km = 1 dk)
s_h = 15            # Sabit servis süresi (Dakika)

# 2. VERİ OKUMA VE KÜMELERİ OLUŞTURMA
dosya = "camasirhane_veri.xlsx"
df_t = pd.read_excel(dosya, sheet_name="Talepler")
df_dist = pd.read_excel(dosya, sheet_name="Mesafeler", index_col=0)


df_dist.columns = df_dist.columns.astype(int)
df_dist.index = df_dist.index.astype(int)

# KÜMELER (Sets)
# D: Günler Kümesi (Excel'deki 1. sütun hariç diğer tüm başlıkları otomatik alır)
D = df_t.columns[1:].tolist() 
# H: Oteller Kümesi (1'den 23'e kadar)
H = df_t["Musteri_ID"].tolist()
# N: Tüm düğümler (0: Depo + H)
N = [0] + H 
# K: Araçlar (1 ve 2)
K = [1, 2] 

# Parametreleri Python sözlüklerine çevirme
q_hd = {}
for h in H:
    for d in D:
        q_hd[h, d] = df_t.loc[df_t["Musteri_ID"] == h, d].values[0]

t_ij = {}
c_ij = {}
for i in N:
    for j in N:
        km = df_dist.loc[i, j]
        t_ij[i, j] = (km / hiz) * 60  
        c_ij[i, j] = km

# KÜMELER (Sets)
D = df_t.columns[1:].tolist() 
# H: Oteller Kümesi (1'den 23'e kadar)
H = df_t["Musteri_ID"].tolist()
# N: Tüm düğümler (0: Depo + H)
N = [0] + H 
# K: Araçlar (1 ve 2)
K = [1, 2] 


q_hd = {}
for h in H:
    for d in D:
        q_hd[h, d] = df_t.loc[df_t["Musteri_ID"] == h, d].values[0]

t_ij = {}
c_ij = {}
for i in N:
    for j in N:
        km = df_dist.loc[i, j]
        t_ij[i, j] = (km / hiz) * 60  
        c_ij[i, j] = km               

# 3. MODEL TANIMLAMA
model = gp.Model("Camasirhane_VRP_Haftalik")

# 4. KARAR DEĞİŞKENLERİ 
x = model.addVars(N, N, K, D, vtype=GRB.BINARY, name="x")
z = model.addVars(H, K, D, vtype=GRB.BINARY, name="z")
t_depo = model.addVars(K, D, vtype=GRB.CONTINUOUS, name="t_depo") 
t_otel = model.addVars(H, K, D, vtype=GRB.CONTINUOUS, name="t_otel") 
delay = model.addVars(K, D, vtype=GRB.CONTINUOUS, name="delay")

# 5. AMAÇ FONKSİYONU (Objective Function)
model.setObjective(
    gp.quicksum(c_ij[i, j] * x[i, j, k, d] for i in N for j in N for k in K for d in D) +
    gp.quicksum(P * delay[k, d] for k in K for d in D),
    GRB.MINIMIZE
)

# 6. KISITLAR (Constraints) 
for d in D:
    for h in H:
        talep_durumu = 1 if q_hd[h, d] > 0 else 0 
        model.addConstr(gp.quicksum(z[h, k, d] for k in K) == talep_durumu, name=f"Talep_{h}_{d}")

    # Kısıt 2: Akış korunumu
    for j in H:
        for k in K:
            model.addConstr(gp.quicksum(x[i, j, k, d] for i in N if i != j) == z[j, k, d], name=f"Giris_{j}_{k}_{d}")
            model.addConstr(gp.quicksum(x[j, i, k, d] for i in N if i != j) == z[j, k, d], name=f"Cikis_{j}_{k}_{d}")

    # Kısıt 3: İlk otele varış
    for j in H:
        for k in K:
            model.addConstr(t_otel[j, k, d] >= T_start + t_ij[0, j] - M * (1 - x[0, j, k, d]), name=f"IlkVaris_{j}_{k}_{d}")

    # Kısıt 4: Oteller arası zaman akışı
    for i in H:
        for j in H:
            if i != j:
                for k in K:
                    model.addConstr(t_otel[j, k, d] >= t_otel[i, k, d] + s_h + t_ij[i, j] - M * (1 - x[i, j, k, d]), name=f"Zaman_{i}_{j}_{k}_{d}")

    # Kısıt 5: Depoya dönüş zamanı
    for i in H:
        for k in K:
            model.addConstr(t_depo[k, d] >= t_otel[i, k, d] + s_h + t_ij[i, 0] - M * (1 - x[i, 0, k, d]), name=f"DepoDonus_{i}_{k}_{d}")

    # Kısıt 6, 7, 8 ve 9
    for k in K:
        model.addConstr(t_depo[k, d] <= b0, name=f"SonSinir_{k}_{d}")
        model.addConstr(delay[k, d] >= t_depo[k, d] - a0, name=f"Gecikme_{k}_{d}")
        model.addConstr(delay[k, d] >= 0, name=f"PozitifGecikme_{k}_{d}")
        model.addConstr(gp.quicksum(x[0, j, k, d] for j in H) <= 1, name=f"DepoCikis_{k}_{d}")


model.setParam('TimeLimit', 3600)  # Maksimum 1 saat (3600 saniye) ara.
model.setParam('MIPGap', 0.05)     # %5'lik eniyilik boşluğuna (Gap) ulaşırsan süreyi beklemeden durması için.

# 7. ÇÖZÜMÜ BAŞLAT
model.optimize()

# 8. SONUÇLARI YAZDIR
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print("\n" + "="*50)
    print("!!! HAFTALIK ROTA BAŞARIYLA BULUNDU !!!")
    print(f"Toplam Maliyet Değeri (Haftalık Z): {model.objVal:.2f}")
    if model.MIPGap > 0:
        print(f"Kabul Edilen Gap Değeri: %{model.MIPGap * 100:.2f}")
    print("="*50)
    
    for d in D:
        print(f"\n--- GÜN: {d.upper()} ---")
        for k in K:
            # Aracın o gün kullanılıp kullanılmadığını kontrol et
            if sum(x[0, j, k, d].x for j in H) > 0.5:
                print(f"  ARAÇ {k} ROTASI:")
                for i in N:
                    for j in N:
                        if x[i, j, k, d].x > 0.5:
                            varis = t_otel[j, k, d].x if j in H else t_depo[k, d].x
                            print(f"    Düğüm {i} -> Düğüm {j} (Varış: {varis:.1f}. dk)")
            else:
                print(f"  ARAÇ {k}: Bu gün depoda yattı (Kullanılmadı).")
else:
    print("Geçerli bir çözüm bulunamadı. (Infeasible: Araçlar 13:00'e kadar yetişemiyor olabilir.)")