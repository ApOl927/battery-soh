"""
Common - small helpers shared by rf_feature_grid.py and model_comparison.py (sections 3.3-3.5).


  load_master()      read output/MASTER_processed.csv
  rf()               the same Random Forest as everywhere in the thesis (200 trees, depth 12)
  metrics()          RMSE, MAE, R2 in one dict
  oxford_measured()  True for the 519 Oxford rows that are real measurements,
                     False for interpolated rows (for "only real measurements" results)
  OUT                folder results/ where scripts save csv and png

Not run on its own, only imported.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# scripts import the other project files (oxford_extractor, lstm_cross_val ...)
HERE = Path(__file__).resolve().parent
FOLDER = HERE
for p in (str(HERE), str(FOLDER)):
    if p not in sys.path:
        sys.path.insert(0, p)
# make the Windows console print UTF-8 (Polish letters) without errors
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#target column and folder for results (made automatically if missing)
TARGET = "soh_percentage"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# Oxford climate chamber is 40 C; a mean discharge temperature outside this band means a broken T series.
OXFORD_T_OK = (30.0, 50.0)


def load_master() -> pd.DataFrame:
    return pd.read_csv(FOLDER / "output" / "MASTER_processed.csv")


def rf(seed: int = 42, **kw) -> RandomForestRegressor:
    """Same Random Forest as in the thesis (200 trees, max depth 12)."""
    return RandomForestRegressor(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1, **kw)


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # R2 makes no sense for fewer than 2 rows
    if y_true.size < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fmt(m: dict[str, float]) -> str:
    return f"RMSE {m['rmse']:6.2f} | MAE {m['mae']:6.2f} | R2 {m['r2']:8.3f}"


# lru_cache: the raw file is read only once. Next calls get the saved result
@lru_cache(maxsize=1)
def _oxford_raw_cycles() -> frozenset | None:
    """Set of (battery_id, cycle) records that really exist in the .mat file (not interpolated)."""
    raw_dir = FOLDER / "OXFORD_RAW"
    if not raw_dir.is_dir():
        return None
    #the same reading functions as in the Oxford extractor
    from scipy.io import loadmat
    from oxford_extractor import (
        _CELL_NAME_RE,
        _cycle_number_from_field_name,
        _extract_q_T_v,
        _get_c1dc_for_cycle,
        _iter_cyc_field_names,
    )

    # collect (cell, cycle) pairs that exist in the raw file - only every 100th cycle
    pairs = set()
    for path in sorted(raw_dir.rglob("*.mat")):
        mat = loadmat(str(path), squeeze_me=True)
        for key in mat:
            if key.startswith("__") or not _CELL_NAME_RE.match(key):
                continue
            for cyc_name in _iter_cyc_field_names(mat[key]):
                nr = _cycle_number_from_field_name(cyc_name)
                c1dc = _get_c1dc_for_cycle(mat[key], cyc_name)
                if nr is None or c1dc is None or _extract_q_T_v(c1dc) is None:
                    continue
                # "OXFORD_" prefix, because battery_id in MASTER has it (data_merger.py)
                pairs.add(("OXFORD_" + key, int(nr)))
    return frozenset(pairs)


def oxford_measured(df: pd.DataFrame) -> pd.Series:
    """True for Oxford rows that come from a real measurement, False for interpolated rows."""
    is_ox = df["dataset_source"].astype(str).str.upper() == "OXFORD"
    # 1) if MASTER already has an is_measured column - use it
    if "is_measured" in df.columns:
        return is_ox & df["is_measured"].fillna(False).astype(bool)
    # 2) otherwise check every row against the pairs from the raw .mat file
    pairs = _oxford_raw_cycles()
    if pairs is None:
        # no raw file: guess - the stored cycles are multiples of 100
        print("WARNING: no OXFORD_RAW folder - approximation: measured = discharge_cycle % 100 == 0")
        return is_ox & (df["discharge_cycle"] % 100 == 0)
    keys = zip(df["battery_id"].astype(str), df["discharge_cycle"].round().astype(int))
    return is_ox & pd.Series([k in pairs for k in keys], index=df.index)