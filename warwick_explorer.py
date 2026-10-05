"""
Warwick explorer - look inside one raw Warwick csv file before writing the extractor.
Section 3.1.1 (warwick_explorer.py part).

What it prints:
  - how many csv files there are (one file = one capacity check of one cell)
  - the info lines above the real table header (why a plain pd.read_csv does not work)
  - the columns we use: Voltage, Current, AhAccu, LogTemp001
  - discharge rows, found by the sign of the current (no column says "discharge")
  - capacity of the discharge from AhAccu

Input:  data/WARWICK_RAW/*.csv or WARWICK_RAW/*.csv (Digatron cycler export)
Output: text in the terminal only
Run:    python warwick_explorer.py
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent # data_teleport.py copies the raw files into data/WARWICK_RAW
RAW_DIR_NAMES = ("data/WARWICK_RAW", "WARWICK_RAW")

# these are the columns the extractor needs
KEY_COLUMNS = ["Voltage", "Current", "AhAccu", "LogTemp001"]
UNITS = {"Voltage": "V", "Current": "A", "AhAccu": "Ah", "LogTemp001": "C"}


def find_raw_dir():
    for name in RAW_DIR_NAMES:
        path = (SCRIPT_DIR / name).resolve()
        if path.is_dir():
            return path
    return None


def find_header_line(lines):
    ##The real header is not the first line, we have to look for it.
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("Step,Status"):
            return i
        if "AhAccu" in line and "Step" in line:
            return i
    return None


def main():
    raw_dir = find_raw_dir()
    if raw_dir is None:
        print("No Warwick data directory found", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(raw_dir.rglob("*.csv"))
    if len(csv_files) == 0:
        print("No .csv files in " + str(raw_dir), file=sys.stderr)
        sys.exit(1)

    print("=== how the data is stored ===")
    print("")
    print("directory: " + str(raw_dir))
    print("number of csv files: " + str(len(csv_files)))
    print("first file names:")
    for path in csv_files[:5]:
        print("  " + path.name)
    print("")
    print("the file name carries the cell number and the cycle number,")
    print("so in Warwick one file is one measurement of one cell, not a whole battery history")
    print("")

    # look inside one file
    file_path = csv_files[0]
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_line = find_header_line(lines)
    # about 15 info lines are above the real table, we have to find the header row
    size_kb = file_path.stat().st_size / 1000
    print("=== inside one file: " + file_path.name + " (" + str(round(size_kb)) + " kB) ===")
    print("")
    print("total lines in file: " + str(len(lines)))
    print("the header row is line number: " + str(header_line))
    print("")
    print("what is written above the header (this is why pandas alone is not enough):")
    for i in range(min(header_line, 12)):
        print("  line " + str(i) + ": " + lines[i][:90])
    print("")

    # read the table, skipping everything above the header
    body = "\n".join(lines[header_line:])
    table = pd.read_csv(StringIO(body), header=0, low_memory=False) # column names come with extra spaces, strip them
    table.columns = [str(c).strip() for c in table.columns]

    print("columns in the table (" + str(len(table.columns)) + "):")
    print("  " + str(list(table.columns)))
    print("number of rows: " + str(len(table)))
    print("")

    print("=== the columns we use ===")
    print("")
    for name in KEY_COLUMNS:
        if name in table.columns:
            values = pd.to_numeric(table[name], errors="coerce").dropna()
            unit = UNITS.get(name, "")
            print("  " + name.ljust(12) + str(values.size).rjust(6) + " values, from " +
                  str(round(values.min(), 2)) + " to " + str(round(values.max(), 2)) + " " + unit)
        else:
            print("  " + name.ljust(12) + " NOT FOUND in this file")
    print("")

    # discharge is not marked, we find it by the sign of the current
    current = pd.to_numeric(table["Current"], errors="coerce").fillna(0.0)
    discharge_rows = (current < 0).sum()  # Current < 0 -> discharge, > 0 -> charge, = 0 -> rest
    print("=== finding the discharge ===")
    # count how many separate discharge segments are in this file
    is_discharge = current < 0
    segments = 0
    previous = False
    for value in is_discharge:
        if value and not previous:
            segments = segments + 1
        previous = value
    print("number of separate discharge segments in this file: " + str(segments))
    print("")
    print("there is no column that says 'this is a discharge'")
    print("rows with negative current (discharge): " + str(discharge_rows))
    print("rows with positive current (charge):    " + str((current > 0).sum()))
    print("rows with zero current (rest):          " + str((current == 0).sum()))
    print("so the discharge is found by the sign of the current")
    print("")
    is_discharge = current < 0
    segments = 0
    previous = False
    for value in is_discharge:
        if value and not previous:
            segments = segments + 1
        previous = value
    print("number of separate discharge segments in this file: " + str(segments))

    # capacity from the accumulated charge during the discharge
    if "AhAccu" in table.columns and discharge_rows > 0:
        ah = pd.to_numeric(table["AhAccu"], errors="coerce")
        ah_discharge = ah[current < 0].dropna()
        capacity = abs(ah_discharge.max() - ah_discharge.min())
        print("capacity of the discharge, computed as max - min of AhAccu: " +
              str(round(capacity, 3)) + " Ah")
        print("")

    print("=== what is stored in the file and what we compute ===")
    print("")
    print("stored in the file:")
    print("  Voltage, Current, AhAccu, LogTemp001   columns of the measurement table")
    print("  cell number and cycle number           in the file name")
    print("")
    print("computed by the extractor:")
    print("  discharge segments    rows where Current < 0")
    print("  capacity_ah           max - min of AhAccu inside the segment")
    print("  soh_percentage        capacity_ah / 5.0 Ah * 100")
    print("  avg_temperature_c     mean of LogTemp001")
    print("  temp_variance         variance of LogTemp001")
    print("  voltage_variance      variance of Voltage")
    print("  resistance            voltage drop at the start of the segment divided by current")
    print("  resistance_ratio      resistance divided by the first resistance of this cell")
    print("")
    print("Warwick stores no ready capacity and no ready resistance, both have to be computed")


if __name__ == "__main__":
    main()