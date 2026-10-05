"""
RF feature grid - the Random Forest cross-dataset table of section 3.3.

  A  last discharge_cycle of every NASA cell (from 4 to 197, median 70)
     -> cycle numbers can mean different things in each lab

  B  3 directions x 4 feature sets = 12 runs (T, T+R, cycle+T, cycle+T+R),
     all on the same rows; plus the share of test rows outside the training range
     for every feature, and MAE inside vs outside that range

  C  max_features: sklearn default (1.0 = all features at every node)
     vs 1 random feature per node (R2 -4.181 -> -3.771, conclusions stay the same)

Input:  output/MASTER_processed.csv
Output: tables in the terminal + results/rf_feature_grid.csv
Run:  python rf_feature_grid.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # also works with python -m
from common import OUT, TARGET, load_master, metrics, rf

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

df = load_master()

# --- A. what discharge_cycle looks like in each lab ---
print("=== A. discharge_cycle in each dataset ===")
for src in ["NASA", "OXFORD", "WARWICK"]:
    d = df[df["dataset_source"] == src]
    # last cycle number of every cell in this lab
    mx = d.groupby("battery_id")["discharge_cycle"].max()
    print(f"{src:8s} cells: {mx.size:3d} | min cycle: {d['discharge_cycle'].min():6.0f} | "
          f"max per cell: from {mx.min():6.0f} to {mx.max():6.0f} (median {mx.median():.0f})")
# full list for NASA: test length is very different between cells
nasa_max = df[df["dataset_source"] == "NASA"].groupby("battery_id")["discharge_cycle"].max()
print("")
print("NASA - last discharge_cycle of each cell:")
print(nasa_max.sort_values().to_string())
print("")

# --- B. 3 directions x 4 feature sets = 12 runs (table in section 3.3) ---
# train on two labs, test on the third one
DIRECTIONS = {
    "NASA+Oxford -> Warwick": (["NASA", "OXFORD"], ["WARWICK"]),
    "NASA+Warwick -> Oxford": (["NASA", "WARWICK"], ["OXFORD"]),
    "Oxford+Warwick -> NASA": (["OXFORD", "WARWICK"], ["NASA"]),
}
# T = temperature, R = resistance_ratio, cycle = discharge_cycle
FEATURE_SETS = {
    "T": ["avg_temperature_c"],
    "T+R": ["avg_temperature_c", "resistance_ratio"],
    "cycle+T": ["discharge_cycle", "avg_temperature_c"],
    "cycle+T+R": ["discharge_cycle", "avg_temperature_c", "resistance_ratio"],
}
ALL_FEATS = sorted({c for f in FEATURE_SETS.values() for c in f})
#drop empty rows 1 time, so all 12 runs use exactly the same rows
data = df.dropna(subset=ALL_FEATS + [TARGET])
print(f"=== B. grid (common dropna: {len(df)} -> {len(data)} rows) ===")

results = []
for direction, (tr_src, te_src) in DIRECTIONS.items():
    train = data[data["dataset_source"].isin(tr_src)]
    test = data[data["dataset_source"].isin(te_src)]
    for set_name, feats in FEATURE_SETS.items():
        model = rf().fit(train[feats], train[TARGET])
        pred = model.predict(test[feats])
        # a test row is outside if a feature value is below or above everything in train
        outside_any = np.zeros(len(test), dtype=bool)
        row = {"direction": direction, "features": set_name, "n_train": len(train), "n_test": len(test),
               **metrics(test[TARGET], pred),
               "mean_soh_real": test[TARGET].mean(), "mean_soh_pred": pred.mean()}
        for c in feats:
            lo, hi = train[c].min(), train[c].max()
            outside = ((test[c] < lo) | (test[c] > hi)).to_numpy()
            outside_any |= outside
            row[f"outside_{c}_%"] = outside.mean() * 100
        row["outside_any_%"] = outside_any.mean() * 100
        # MAE and mean SoH separately for rows inside / outside a training range
        if outside_any.any() and (~outside_any).any():
            y = test[TARGET].to_numpy()
            row["n_outside"] = int(outside_any.sum())
            row["mae_inside"] = metrics(y[~outside_any], pred[~outside_any])["mae"]
            row["mae_outside"] = metrics(y[outside_any], pred[outside_any])["mae"]
            row["real_outside"] = y[outside_any].mean()
            row["pred_outside"] = pred[outside_any].mean()
        results.append(row)

# print the tables (screenshots in 3.3) and save everything to csv
res = pd.DataFrame(results)
cols = ["direction", "features", "rmse", "mae", "r2", "mean_soh_real", "mean_soh_pred", "outside_any_%"]
print(res[cols].round(3).to_string(index=False))
print("")
print("share of test rows outside the training range, per feature:")
cols_out = [c for c in res.columns if c.startswith("outside_") and c != "outside_any_%"]
print(res[["direction", "features"] + cols_out].round(2).to_string(index=False))
extra = [c for c in ["n_outside", "mae_inside", "mae_outside", "real_outside", "pred_outside"] if c in res.columns]
if extra:
    print("")
    print("variants with rows outside the range - error inside vs outside:")
    print(res.dropna(subset=extra[:1])[["direction", "features"] + extra].round(2).to_string(index=False))
res.to_csv(OUT / "rf_feature_grid.csv", index=False)
print(f"\nsaved: {OUT / 'rf_feature_grid.csv'}")

# --- C. if picking one random feature per node change anything ?---
print("")
print("=== C. max_features ===")
print(f"scikit-learn {sklearn.__version__}, default max_features for regression: "
      f"{RandomForestRegressor().max_features!r} (1.0 = all features at every node)")
# checked on one direction only: NASA + Warwick -> Oxford, features T + R
feats = FEATURE_SETS["T+R"]
train = data[data["dataset_source"].isin(["NASA", "WARWICK"])]
test = data[data["dataset_source"] == "OXFORD"]
# 1.0 = all features at every node (sklearn default), 1 = one random feature per node
for mf in [1.0, 1]:
    m = rf(max_features=mf).fit(train[feats], train[TARGET])
    w = metrics(test[TARGET], m.predict(test[feats]))
    print(f"  max_features={mf!r:5}: RMSE {w['rmse']:.2f} | MAE {w['mae']:.2f} | R2 {w['r2']:.3f}")