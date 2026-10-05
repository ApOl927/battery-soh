"""
Split comparison - how the way of splitting train / test changes the score.
Section 3.2 (table with variants A-D).

Same Random Forest (200 trees, depth 12) and the same features
(avg_temperature_c, resistance_ratio), only the split changes:
  A  random rows, 80/20          -> R2 0.964, but only because of leakage
  B  whole cells, 80/20          -> R2 0.257, closer to the truth
  C  NASA + Warwick -> Oxford        -> R2 -4.181, new lab
  D  like B + capacity_ah          -> R2 0.939, leakage again (SoH is computed from capacity)

Input:  output/MASTER_processed.csv
Output: text in the terminal only
Run:  python split_comparison.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FOLDER = Path(__file__).resolve().parent
# CECHY = features, CEL = target (SoH)
CECHY = ["avg_temperature_c", "resistance_ratio"]
CEL = "soh_percentage"

df = pd.read_csv(FOLDER / "output" / "MASTER_processed.csv")


def ucz_i_ocen(nazwa, train, test, cechy):
    """Train the RF on train, check it on test and print the results."""
    # the same RF settings as everywhere in the thesis
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[cechy], train[CEL])
    pred = model.predict(test[cechy])

    rmse = float(np.sqrt(mean_squared_error(test[CEL], pred)))
    mae = float(mean_absolute_error(test[CEL], pred))
    r2 = float(r2_score(test[CEL], pred))

    # cells that are in train and in test at the same time (A has 64, the others 0)
    wspolne = set(train["battery_id"]) & set(test["battery_id"])

    print(f"--- {nazwa} ---")
    print(f"  features: {cechy}")
    print(f"  train rows: {len(train)} | test rows: {len(test)}")
    print(f"  test sources: {test['dataset_source'].value_counts().to_dict()}")
    print(f"  cells shared between train and test: {len(wspolne)}")
    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    print("")

# --- A. random rows: neighbour cycles of one cell land in train and test -> leakage ---
# drop rows with empty features or SoH (A, B and C use these rows)
dane = df.dropna(subset=CECHY + [CEL]).copy()
train_a, test_a = train_test_split(dane, test_size=0.2, random_state=42)
ucz_i_ocen("A. random split by rows", train_a, test_a, CECHY)

# --- B. whole cells: 80/20 of cells, rows end up about 85/15 ---
# groups=battery_id -> all cycles of one cell go together to train or to test
podzial = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(podzial.split(dane, groups=dane["battery_id"]))
train_b = dane.iloc[idx_train]
test_b = dane.iloc[idx_test]
ucz_i_ocen("B. split by cells", train_b, test_b, CECHY)

# --- C. other lab: train on NASA + Warwick, test on all of Oxford ---
train_c = dane[dane["dataset_source"].isin(["NASA", "WARWICK"])]
test_c = dane[dane["dataset_source"] == "OXFORD"]
ucz_i_ocen("C. split between datasets (NASA+Warwick -> Oxford)", train_c, test_c, CECHY)

# --- D. like B + capacity_ah: leakage, because SoH is computed from capacity ---
cechy_d = CECHY + ["capacity_ah"]
dane_d = df.dropna(subset=cechy_d + [CEL]).copy()
# same splitter and seed as in B, so the test cells are the same
idx_train, idx_test = next(podzial.split(dane_d, groups=dane_d["battery_id"]))
ucz_i_ocen("D. same as B, but with capacity_ah as feature", dane_d.iloc[idx_train], dane_d.iloc[idx_test], cechy_d)