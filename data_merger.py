"""
Data merger - joins the three extractor tables into output/MASTER_processed.csv.
Sections 3.1.2 (MASTER table, dataset_source column) and 3.2 (unique battery_id).

  - checks that every table has resistance_ratio (not ohms) and the other needed columns
  - renames Oxford cycle_index -> discharge_cycle
  - adds dataset_source = NASA / OXFORD / WARWICK (not a feature, only for splitting the data)
  - adds the lab to battery_id (OXFORD_Cell2 vs WARWICK_Cell2), otherwise 7 ids repeat

Input:  output/NASA_processed.csv, OXFORD_processed.csv, WARWICK_processed.csv
Output: output/MASTER_processed.csv (58083 rows: NASA 2538, Oxford 55308, Warwick 237)
Run:    python data_merger.py   (warwick_extractor.py also calls it at the end)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_datasets(output_dir: str | Path) -> pd.DataFrame:
    """Read the three tables, check them, join them and save MASTER."""
    #paths to the three extractor tables and to the final MASTER file
    out = Path(output_dir)
    nasa_path = out / "NASA_processed.csv"
    oxford_path = out / "OXFORD_processed.csv"
    warwick_path = out / "WARWICK_processed.csv"
    master_path = out / "MASTER_processed.csv"

    nasa = pd.read_csv(nasa_path)
    oxford = pd.read_csv(oxford_path)
    warwick = pd.read_csv(warwick_path)

    nasa = nasa.copy()
    oxford = oxford.copy()
    warwick = warwick.copy()
    # remove an old dataset_source column if an extractor already wrote one
    for _, frame in (("NASA", nasa), ("Oxford", oxford), ("Warwick", warwick)):
        if "dataset_source" in frame.columns:
            frame.drop(columns=["dataset_source"], inplace=True)
    # Oxford calls the cycle column cycle_index ->   same name as in NASA and Warwick
    if "cycle_index" in oxford.columns:
        oxford = oxford.rename(columns={"cycle_index": "discharge_cycle"})
    # stop if a table is old. Ohms instead of resistance_ratio or a missing column
    for label, frame in (("NASA", nasa), ("Oxford", oxford), ("Warwick", warwick)):
        if "internal_resistance_ohm" in frame.columns:
            raise ValueError(f"{label}_processed.csv must use resistance_ratio, not internal_resistance_ohm")
        if "resistance_ratio" not in frame.columns:
            raise ValueError(f"{label}_processed.csv is missing resistance_ratio")
        for req in ("nominal_capacity", "form_factor", "temp_variance", "voltage_variance"):
            if req not in frame.columns:
                raise ValueError(f"{label}_processed.csv is missing {req}")
    #lab name for every row (not a model feature, only for train/test splits )
    nasa["dataset_source"] = "NASA"
    oxford["dataset_source"] = "OXFORD"
    warwick["dataset_source"] = "WARWICK"

    # Cell2 -Cell8 exist in Oxford and in Warwick -> add the lab to battery_id: OXFORD_Cell2
    for frame in (nasa, oxford, warwick):
        frame["battery_id"] = frame["dataset_source"] + "_" + frame["battery_id"].astype(str)

    # one big table: rows of all three labs under each other
    master = pd.concat([nasa, oxford, warwick], ignore_index=True)
    # make sure the names are exactly NASA / OXFORD / WARWICK
    master["dataset_source"] = master["dataset_source"].astype(str).str.strip().str.upper()
    master.to_csv(master_path, index=False)
    return master


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "output"
    df = merge_datasets(out)
    # quick check: total rows and rows per lab. Should be 58083: NASA 2538, OXFORD 55308, WARWICK 237
    print(f"MASTER_processed.csv rows: {len(df)}")
    print(df["dataset_source"].value_counts().to_string())
    print()
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()