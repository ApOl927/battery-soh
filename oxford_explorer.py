"""
Oxford explorer - look inside the raw Oxford file before writing the extractor.
Section 3.1.1 (oxford_explorer.py part).

What it prints:
  - how the data is nested: file -> cell -> cycle -> block -> series -> numbers
  - which cycles are stored: only every 100th (cyc0000, cyc0100, ... cyc8200)
  - what is inside the C1dc block (1C discharge): t, v, q, T
  - capacity of the first discharge, computed from q (there is no ready Capacity)

Input:  OXFORD_RAW/Oxford_Battery_Degradation_Dataset_1.mat (all 8 cells in one file)
Output: text in the terminal only
Run:    python oxford_explorer.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
# Oxford keeps all 8 cells in one .mat file (NASA has one file per cell)
OXFORD_RAW_DIR = Path(__file__).resolve().parent / "OXFORD_RAW"
PREFERRED_NAME = "Oxford_Battery_Degradation_Dataset_1.mat"
# t is a date (MATLAB datenum, in days), not seconds from the start like in NASA
UNITS = {"t": "days (MATLAB datenum)", "v": "V", "q": "mAh", "T": "C"}


def pick_file():
    """Take the preferred file if it exists, otherwise the first .mat file."""
    all_files = sorted(OXFORD_RAW_DIR.rglob("*.mat"))
    if len(all_files) == 0:
        return None
    for path in all_files:
        if path.name == PREFERRED_NAME:
            return path
    return all_files[0]


def unwrap(obj):
    """Open the MATLAB boxes until we reach something we can read."""
    for step in range(64):
        if isinstance(obj, np.ndarray):
            if obj.shape == ():
                obj = obj[()]
                continue
            if obj.size == 1:
                obj = obj.flat[0]
                continue
        break
    return obj


def get_field_names(obj):
    """to understand what fields areinside a MATLAB structure."""
    obj = unwrap(obj)
    if isinstance(obj, np.void) and obj.dtype.names is not None:
        return list(obj.dtype.names)
    if hasattr(obj, "_fieldnames"):
        return list(obj._fieldnames)
    return []


def get_field(obj, name):
    """One field of a MATLAB structure, by its name."""
    obj = unwrap(obj)
    if isinstance(obj, np.void) and obj.dtype.names is not None and name in obj.dtype.names:
        return obj[name]
    return getattr(obj, name)


def get_numbers(field):
    """Turn a field into a plain list of numbers."""
    values = np.asarray(unwrap(field), dtype=np.float64).ravel()
    return values

# Structure of MATLAB file
def show_structure(mat):
    #Go down the file tree, one level at a time, and print what we find.
    print("=== how the data is nested in the .mat file ===")
    print("")

    # level 1: the file itself
    cell_names = []
    for key in mat:
        if not key.startswith("__"):
            cell_names.append(key)
    print("level 1  file          -> " + str(len(cell_names)) + " cells: " + str(cell_names))

    # level 2: one cell holds many cycles
    cell = mat[cell_names[0]]
    cycle_names = []
    for name in get_field_names(cell):
        if name.startswith("cyc"):
            cycle_names.append(name)
    print("level 2  " + cell_names[0].ljust(13) + " -> " + str(len(cycle_names)) +
          " cycles: " + str(cycle_names[:3]) + " ... " + cycle_names[-1])

    # level 3: one cycle holds several measurement blocks
    cycle = get_field(cell, cycle_names[0])
    block_names = get_field_names(cycle)
    print("level 3  " + cycle_names[0].ljust(13) + " -> blocks: " + str(block_names))

    # level 4: one block holds the measurement series
    block = get_field(cycle, "C1dc")
    series_names = get_field_names(block)
    print("level 4  C1dc          -> series: " + str(series_names))

    # level 5: the numbers
    voltage = get_numbers(get_field(block, "v"))
    print("level 5  C1dc.v        -> " + str(voltage.size) + " numbers, first three: " +
          str(voltage[:3].round(3)))
    print("")
    print("so to read one voltage value we go: file -> cell -> cycle -> block -> series -> number")
    print("")

def show_what_is_ready():
    """Which values are stored in the file and which ones need to be computed."""
    print("=== what is stored in the file and what we compute ===")
    print("")
    print("stored in the file:")
    print("  Capacity             one ready value in every discharge cycle [Ah]")
    print("  Re                   one ready value in every impedance cycle [Ohm]")
    print("  Voltage_measured     series of a few hundred points [V]")
    print("  Temperature_measured series of a few hundred points [C]")
    print("")
    print("computed by the extractor:")
    print("  discharge_cycle      our own counter of discharge cycles")
    print("  soh_percentage       Capacity / 2.0 Ah * 100")
    print("  avg_temperature_c    mean of Temperature_measured")
    print("  temp_variance        variance of Temperature_measured")
    print("  voltage_variance     variance of Voltage_measured")
    print("  resistance_ratio     Re divided by the first Re of this battery")
    print("")
    print("NASA stores ready Capacity and Re -> we do not compute them ourselves")
    print("")

def main():
    if not OXFORD_RAW_DIR.is_dir():
        print("Directory not found: " + str(OXFORD_RAW_DIR), file=sys.stderr)
        sys.exit(1)

    file_path = pick_file()
    if file_path is None:
        print("No .mat files in " + str(OXFORD_RAW_DIR), file=sys.stderr)
        sys.exit(1)

    size_mb = file_path.stat().st_size / 1e6
    print("file: " + file_path.name + " (" + str(round(size_mb, 1)) + " MB)")
    
     # squeeze_me=True removes the extra 1x1 boxes, so less unwrapping than in NASA
    mat = loadmat(str(file_path), squeeze_me=True)
    show_structure(mat)
    #top level: the cells stored in this file
    cell_names = []
    for key in mat:
        if not key.startswith("__"):
            cell_names.append(key)
    print("cells in this file: " + str(cell_names))
    print("")

    #look inside the first cell
    cell_name = cell_names[0]
    cell = mat[cell_name]
    cycle_names = []
    for name in get_field_names(cell):
        if name.startswith("cyc"):
            cycle_names.append(name)

    print("cell: " + cell_name)
    print("number of stored cycles: " + str(len(cycle_names)))
    print("first cycle names: " + str(cycle_names[:5]))
    print("last cycle name: " + str(cycle_names[-1]))

    # the cycle names contain the cycle number
    numbers = []
    for name in cycle_names:
        numbers.append(int(name.replace("cyc", "")))
    if len(numbers) > 1:
        print("cycle numbers go from " + str(numbers[0]) + " to " + str(numbers[-1]) +
              ", step " + str(numbers[1] - numbers[0]))
        print("so the cell was measured only every " + str(numbers[1] - numbers[0]) + " cycles")
    print("")

    # look inside the first cycle
    first_cycle_name = cycle_names[0]
    first_cycle = get_field(cell, first_cycle_name)
    block_names = get_field_names(first_cycle)
    print("cycle " + first_cycle_name + " contains blocks: " + str(block_names))
    print("C1dc is the 1C discharge, this is the block the extractor uses")
    print("")

    # look inside the discharge block
    discharge = get_field(first_cycle, "C1dc")
    print("fields inside C1dc:")
    for name in get_field_names(discharge):
        values = get_numbers(get_field(discharge, name))
        unit = UNITS.get(name, "")
        print("  " + name.ljust(6) + str(values.size).rjust(6) + " points, from " +
              str(round(values.min(), 2)) + " to " + str(round(values.max(), 2)) + " " + unit)

    # there is no ready capacity value here, it has to be computed from q
    q = get_numbers(get_field(discharge, "q")) # q goes from 0 to about -739 mAh, so |max - min| = charge taken out in this discharge
    capacity_mah = abs(q.max() - q.min())
    print("")
    print("there is no ready Capacity field in Oxford data")
    print("capacity has to be computed from q: max - min = " +
          str(round(capacity_mah, 1)) + " mAh = " + str(round(capacity_mah / 1000, 3)) + " Ah")

    show_what_is_ready()

if __name__ == "__main__":
    main()