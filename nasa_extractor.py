"""
NASA extractor - raw NASA .mat files -> output/NASA_processed.csv.
Sections 3.1.2 (feature extraction) and 3.1.3 (data cleaning).

One row = one discharge cycle of one cell:
  - capacity_ah: ready Capacity value from the file
  - soh_percentage: capacity_ah / 2.0 Ah * 100
  - avg_temperature_c, temp_variance, voltage_variance: from the measured series
  - resistance_ratio: Re / R0, Re comes from the impedance cycles, R0 = Re of the first discharge

Cleaning (3.1.3):
  - discharges outside 0.5-2.2 Ah are skipped (231 broken cycles, all in B0033 and later)
  - only Re > 0 is used (B0050 had negative Re -> resistance_ratio -65454)

Input:  NASA_RAW/*.mat (one file per cell, 34 cells, chamber about 4 / 24 / 43 C)
Output: output/NASA_processed.csv
Run:    python nasa_extractor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
# NASA cells: 18650, nominal 2.0 Ah, cylindrical -> form_factor = 1
NOMINAL_CAPACITY_AH: float = 2.0
FORM_FACTOR: int = 1
# output columns, one row = one discharge (table in section 3.1.2)
RESULT_COLUMNS: tuple[str, ...] = (
    "battery_id",
    "discharge_cycle",
    "avg_temperature_c",
    "resistance_ratio",
    "capacity_ah",
    "nominal_capacity",
    "form_factor",
    "temp_variance",
    "voltage_variance",
    "soh_percentage",
)

# helpers below open the nested MATLAB boxes: battery -> cycle -> data -> series
def _cycle_type_string(raw: np.ndarray) -> str:
    #normalize cycle "type" field to a plain string.
    x = np.squeeze(raw)
    if x.shape == ():
        item = x.item()
    else:
        item = x.flat[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace").strip()
    if isinstance(item, str):
        return item.strip()
    return str(item).strip()


def _unwrap_to_scalar(obj: object) -> float | None:
    # unwrap MATLAB-style (1,1) object/float arrays to a single float.
    cur: object = obj
    for _ in range(32):
        if isinstance(cur, (float, int, np.floating, np.integer)):
            return float(cur)
        if not isinstance(cur, np.ndarray):
            return None
        if cur.size == 0:
            return None
        if cur.dtype == object and cur.shape == (1, 1):
            cur = cur[0, 0]
            continue
        if cur.dtype.names:
            return None
        if np.issubdtype(cur.dtype, np.number):
            val = np.squeeze(cur)
            if np.iscomplexobj(val):
                return float(np.real(val))
            return float(val)
        if cur.dtype == object:
            cur = cur.flat[0]
            continue
        val = np.squeeze(cur)
        if np.iscomplexobj(val):
            return float(np.real(val))
        return float(val)
    return None


def _data_struct_from_cycle(cycle: np.void) -> np.void | None:
    #return the inner " data" struct (numpy.void) or None.
    try:
        data_wrapped = cycle["data"]
    except (KeyError, ValueError, IndexError, TypeError):
        return None

    if not isinstance(data_wrapped, np.ndarray):
        return None

    ds: object = data_wrapped
    if isinstance(ds, np.ndarray):
        if ds.shape == (1, 1):
            ds = ds[0, 0]
        elif ds.size == 1:
            ds = ds.flat[0]

    if not isinstance(ds, np.void):
        return None
    return ds


def _extract_scalar_field(data_struct: np.void, field: str) -> float | None:
    #Read a numeric scalar from a structured "data" void (Capacity, Re, etc.)
    names = data_struct.dtype.names
    if names is None or field not in names:
        return None
    return _unwrap_to_scalar(data_struct[field])


def _extract_discharge_capacity_ah(data_struct: np.void) -> float | None:
    return _extract_scalar_field(data_struct, "Capacity")


def _extract_re_ohm(data_struct: np.void) -> float | None:
   #Estimated electrolyte resistance "Re" (Ohm) on impedance "data" struct.
    return _extract_scalar_field(data_struct, "Re")


def _extract_temperature_mean_c(data_struct: np.void) -> float | None:
    #  mean of "Temperature_measured" time series (deg C) on discharge "data" struct.
    names = data_struct.dtype.names
    if names is None or "Temperature_measured" not in names:
        return None
    tm = data_struct["Temperature_measured"]
    arr = _to_float_(tm)
    if arr is None or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
    return float(np.mean(arr))


def _array_from_field(data_struct: np.void, field: str) -> np.ndarray | None:
    names = data_struct.dtype.names
    if names is None or field not in names:
        return None
    arr = _to_float_(data_struct[field])
    if arr is None or arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr


def _scalar_spread(arr: np.ndarray | None) -> float:
    """
    Spread metric for a 1D array: sample variance if available else max-min delta.
    Missing / empty input -> NaN.
    """ #Variance of the points; no points -> NaN, one point -> 0
    if arr is None or arr.size == 0:
        return float(np.nan)
    if arr.size == 1:
        return 0.0
    v = float(np.var(arr))
    if np.isfinite(v):
        return v
    return float(np.ptp(arr)) if np.isfinite(np.ptp(arr)) else float(np.nan)


def get_temp_var(data_struct: np.void) -> float:
    arr = _array_from_field(data_struct, "Temperature_measured")
    return _scalar_spread(arr)


def _voltage_variance_discharge(data_struct: np.void) -> float:
    for field in ("Voltage_measured", "Voltage"):
        arr = _array_from_field(data_struct, field)
        if arr is not None and arr.size > 0:
            return _scalar_spread(arr)
    return float(np.nan)


def _to_float_(obj: object) -> np.ndarray | None:
    """Unwrap nested object arrays to a 1D float array."""
    cur: object = obj
    for _ in range(32):
        if isinstance(cur, np.ndarray):
            if cur.size == 0:
                return None
            if cur.dtype == object and cur.shape == (1, 1):
                cur = cur[0, 0]
                continue
            if cur.dtype == object:
                cur = cur.flat[0]
                continue
            if np.issubdtype(cur.dtype, np.number):
                a = np.asarray(cur, dtype=np.float64).ravel()
                if np.iscomplexobj(a):
                    a = np.real(a)
                return a.astype(np.float64, copy=False)
            return None
        return None
    return None


def _battery_key(mat: dict[str, object], mat_path: Path) -> str:
    keys = [k for k in mat if not k.startswith("__")]
    stem = mat_path.stem
    if stem in keys:
        return stem
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"{mat_path}: cannot pick battery key; stem={stem!r}, keys={keys!r}"
    )


class NASABatteryParser:
    """Parse NASA PCoE .mat battery files into tabular features per discharge cycle."""

    def __init__(self, raw_dir: str | Path) -> None:
        self._raw_dir = Path(raw_dir)

    def parse_all(self) -> pd.DataFrame:
        #Read every .mat file, one row per good discharge cycle.
        
        rows: list[dict[str, float | str | int]] = []
        if not self._raw_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self._raw_dir.resolve()}")

        for mat_path in sorted(self._raw_dir.glob("*.mat")):
            try:
                mat = loadmat(str(mat_path), struct_as_record=True, squeeze_me=False)
                battery_id = _battery_key(mat, mat_path)
                root = mat[battery_id][0, 0]
                cycle_block = root["cycle"]
                cycle_array = cycle_block[0]
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                print(
                    f"Skip {mat_path.name}: cannot open root/cycle ({exc})",
                    file=sys.stderr,
                )
                continue
                 # last known Re; NaN until the first impedance (B0005: index 40, after ~20 discharges)
            current_resistance = np.nan
            discharge_counter = 0
            n = int(cycle_array.shape[0])

            for i in range(n):
                ci = cycle_array[i]
                if not isinstance(ci, np.void):
                    continue

                try:
                    ctype = _cycle_type_string(ci["type"])
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
                # cycles go in time order: impedance only updates Re. Discharge makes a row, charge is skipped
                if ctype == "impedance":
                    ds = _data_struct_from_cycle(ci)
                    if ds is None:
                        continue
                    re_val = _extract_re_ohm(ds)
                    
                    
                    if re_val is not None and np.isfinite(re_val) and re_val > 0:
                        current_resistance = float(re_val)
                    continue

                if ctype != "discharge":
                    continue
                
                discharge_counter += 1

                ds = _data_struct_from_cycle(ci)
                if ds is None:
                    continue

                cap = _extract_discharge_capacity_ah(ds)
                avg_temp = _extract_temperature_mean_c(ds)
                temp_variance = get_temp_var(ds)
                voltage_variance = _voltage_variance_discharge(ds)
                if cap is None or avg_temp is None:
                    continue
                
                
                if not (0.5 <= cap <= 2.2):
                    continue
                    # SoH = capacity / nominal capacity * 100
                soh = (cap / NOMINAL_CAPACITY_AH) * 100.0
                rows.append(
                    {
                        "battery_id": battery_id,
                        "discharge_cycle": discharge_counter,
                        "avg_temperature_c": avg_temp,
                        "internal_resistance_ohm": current_resistance,
                        "capacity_ah": cap,
                        "temp_variance": temp_variance,
                        "voltage_variance": voltage_variance,
                        "soh_percentage": soh,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=list(RESULT_COLUMNS))

        df = df.sort_values(["battery_id", "discharge_cycle"], kind="mergesort").reset_index(
            drop=True
        )
        # first discharges have no Re yet -> they get the first measured Re, so their ratio = 1
        df["internal_resistance_ohm"] = df.groupby("battery_id", sort=False)[
            "internal_resistance_ohm"
        ].transform(lambda s: s.ffill().bfill())
        r0 = df.groupby("battery_id", sort=False)["internal_resistance_ohm"].transform("first") # R0 = Re of the first discharge of this cell, ratio = Re / R0
        df["resistance_ratio"] = df["internal_resistance_ohm"] / r0.replace(0.0, np.nan)
          # constant columns, the same in every NASA row: they tell the model the cell type
        df["nominal_capacity"] = NOMINAL_CAPACITY_AH
        df["form_factor"] = FORM_FACTOR
        return df[list(RESULT_COLUMNS)]


def main() -> None:
    root = Path(__file__).resolve().parent
    raw = root / "NASA_RAW"
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "NASA_processed.csv"

    parser = NASABatteryParser(raw)
    df = parser.parse_all()
    print(df.head(5).to_string())
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()

