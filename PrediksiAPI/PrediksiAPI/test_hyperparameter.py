"""
Pengujian 6 Hyperparameter XGBoost
Evaluasi MAPE & RMSE untuk setiap variasi parameter
Baseline: max_depth=5, learning_rate=0.2, n_estimators=150, gamma=1, reg_lambda=0.01, reg_alpha=10
"""
import pymysql
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

# ========================================
# 1. Koneksi Database
# ========================================
conn = pymysql.connect(host="localhost", user="root", password="", database="prediksi_svr")
query = "SELECT * FROM dataset_xgboost WHERE nama_dataset = 'dataset_training'"
df = pd.read_sql(query, conn)
conn.close()

# ========================================
# 2. Preprocessing
# ========================================
df["Sin_Bulan"] = np.sin(2 * np.pi * df["Bulan"] / 12)
df["Cos_Bulan"] = np.cos(2 * np.pi * df["Bulan"] / 12)

df_grouped = df.groupby(["Kecamatan", "Tahun", "Bulan"]).agg({
    "FASTER": "sum", "ONTIME": "sum", "OVERSLA": "sum",
    "Jumlah_Paket": "sum", "SLA": "sum",
    "Sin_Bulan": "first", "Cos_Bulan": "first"
}).reset_index()

FITUR = ["Bulan", "Sin_Bulan", "Cos_Bulan", "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3"]
kategori_target = ["FASTER", "ONTIME", "OVERSLA"]
daftar_kec = sorted(df_grouped["Kecamatan"].unique())

# ========================================
# 3. Baseline (Tabel 10)
# ========================================
BASELINE = {
    "max_depth": 5,
    "learning_rate": 0.2,
    "n_estimators": 150,
    "gamma": 1,
    "reg_lambda": 0.01,
    "reg_alpha": 10,
}

# ========================================
# 4. Fungsi Evaluasi
# ========================================
def evaluasi(override):
    """Evaluasi model XGBoost dengan parameter override terhadap baseline."""
    params = {
        "objective": "reg:squarederror",
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "random_state": 42,
        "verbosity": 0,
        **BASELINE,
        **override,
    }
    mapes, rmses = [], []
    for kec in daftar_kec:
        dk = df_grouped[df_grouped["Kecamatan"] == kec].sort_values(["Tahun", "Bulan"]).reset_index(drop=True)
        for kat in kategori_target:
            dl = dk.copy()
            dl["Lag_1"] = dl[kat].shift(1)
            dl["Lag_2"] = dl[kat].shift(2)
            dl["Lag_3"] = dl[kat].shift(3)
            dl["Rolling_Mean_3"] = dl[kat].rolling(3, min_periods=1).mean().shift(1)
            dl = dl.dropna()
            n = int(len(dl) * 0.8)
            Xtr, ytr = dl.iloc[:n][FITUR].values, dl.iloc[:n][kat].values
            Xte, yte = dl.iloc[n:][FITUR].values, dl.iloc[n:][kat].values
            if len(Xtr) < 3 or len(Xte) == 0:
                continue
            m = xgb.XGBRegressor(**params)
            m.fit(Xtr, ytr, verbose=False)
            yp = np.maximum(m.predict(Xte), 0)
            rmses.append(float(np.sqrt(mean_squared_error(yte, yp))))
            mask = yte != 0
            mapes.append(float(np.mean(np.abs((yte[mask] - yp[mask]) / yte[mask])) * 100) if np.any(mask) else 0)
    return round(np.mean(mapes), 2), round(np.mean(rmses), 2)

# ========================================
# 5. Variasi Parameter yang Diuji
# ========================================
variasi = {
    "reg_lambda": [0.01, 0.1, 0.5, 1, 3, 5, 8, 10],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "max_depth": [3, 5, 7, 9, 11],
    "n_estimators": [50, 80, 100, 150, 200],
    "gamma": [0, 0.5, 1, 3, 5],
}

KOLOM_URUT = ["reg_lambda", "reg_alpha", "learning_rate", "max_depth", "n_estimators", "gamma"]
sheets = {}

# ========================================
# 6. Jalankan Pengujian
# ========================================
print("=" * 60)
print("PENGUJIAN 6 HYPERPARAMETER (FORMAT TABEL SKRIPSI)")
print("=" * 60)

label_map = {
    "reg_lambda": "Uji Nilai Lambda",
    "reg_alpha": "Uji Nilai Alpha",
    "learning_rate": "Uji Nilai Learning Rate",
    "max_depth": "Uji Nilai Max Depth",
    "n_estimators": "Uji Nilai N Estimators",
    "gamma": "Uji Nilai Gamma",
}

for param_name, values in variasi.items():
    label = label_map[param_name]
    print(f"\n--- {label} ---")
    rows = []

    for val in values:
        override = {param_name: val}
        mape, rmse = evaluasi(override)
        print(f"  {param_name}={val} -> MAPE={mape}%, RMSE={rmse}")

        row = {}
        for col in KOLOM_URUT:
            if col == param_name:
                row[col] = val
            else:
                row[col] = BASELINE[col]
        row["MAPE (%)"] = mape
        row["RMSE"] = rmse
        rows.append(row)

    sheets[label] = pd.DataFrame(rows)

# ========================================
# 7. Export ke Excel
# ========================================
output_file = os.path.join(os.path.dirname(__file__), "hasil_pengujian_parameter.xlsx")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for name, df_sheet in sheets.items():
        df_sheet.to_excel(writer, sheet_name=name[:31], index=False)
        print(f"  Sheet: {name}")

print(f"\nFile disimpan: {output_file}")
print("Selesai!")
