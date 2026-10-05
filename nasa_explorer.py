"""
NASA explorer - look inside one raw NASA file (B0005.mat) before writing the extractor.
Section 3.1.1 (figures with the structure of B0005.mat).

What it prints:
  - how the data is nested: file -> battery -> cycle -> data -> series -> numbers
  - how many charge / discharge / impedance cycles there are and where each type starts
    (the first impedance is only at index 40 -> no Re for the first discharges)
  - one example cycle of each type: number of points and min / max of every series

Input:  NASA_RAW/B0005.mat
Output: text in the terminal only
Run:    python nasa_explorer.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
# one file is enough to see the structure, all NASA files look the same
NASA_RAW_DIR = Path(__file__).resolve().parent / "NASA_RAW"
FILE_PATH = NASA_RAW_DIR / "B0005.mat"

UNITS = {
    "Voltage_measured": "V",
    "Current_measured": "A",
    "Temperature_measured": "C",
    "Voltage_charge": "V",
    "Current_charge": "A",
    "Voltage_load": "V",
    "Current_load": "A",
    "Time": "s",
    "Capacity": "Ah",
    "Re": "Ohm",
    "Rct": "Ohm",
}


def get_type_name(raw_type):
    """The 'type' field as a normal string. for ex'discharge'."""
    x = np.squeeze(raw_type)
    if x.shape == ():
        item = x.item()
    else:
        item = x.flat[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace").strip()
    return str(item).strip()


def get_numbers(field):
    """Open the MATLAB boxes until reach the real numbers."""
    while isinstance(field, np.ndarray) and field.dtype == object and field.size >= 1:
        field = field.flat[0]
    if isinstance(field, np.ndarray):
        values = field.ravel()
        if np.iscomplexobj(values):
            values = np.real(values)   # impedance fields are complex. Need the real part
        return values.astype(float)
    if isinstance(field, complex):
        return np.array([field.real])
    return np.array([float(field)])


def print_one_cycle(cycle, index, type_name):
    """Print a readable summary of one cycle."""
    print("--- cycle type '" + type_name + "', index " + str(index) + " ---")

    ambient = get_numbers(cycle["ambient_temperature"])
    print("ambient temperature: " + str(round(ambient[0])) + " C")
# 'time' is the start date: year, month, day, hour, minute, second
    start = get_numbers(cycle["time"])
    print("start date: year " + str(int(start[0])) + ", month " + str(int(start[1])) +
          ", day " + str(int(start[2])))
    # [0, 0] opens the 1x1 MATLAB box around the data struct
    data = cycle["data"][0, 0]

    # first group: fields with many numbers, they are measurement series
    print("measurement series:")
    for name in data.dtype.names:
        values = get_numbers(data[name])
        if values.size > 1:
            unit = UNITS.get(name, "")
            print("  " + name.ljust(22) + str(values.size).rjust(5) + " points, from " +
                  str(round(values.min(), 2)) + " to " + str(round(values.max(), 2)) + " " + unit)

    # second group: fields with one number they are ready values from NASA
    single_names = []
    for name in data.dtype.names:
        values = get_numbers(data[name])
        if values.size == 1:
            single_names.append(name)

    if len(single_names) == 0:
        print("single values: none in this cycle type")
    else:
        print("single values (already computed by NASA):")
        for name in single_names:
            values = get_numbers(data[name])
            unit = UNITS.get(name, "")
            print("  " + name.ljust(22) + " = " + str(round(values[0], 4)) + " " + unit)

    print("")

def show_structure(mat, battery_name, cycles):
    # Go down the file tree - one level at time, and print what we find.
    print("=== how the data is nested in the .mat file ===")
    print("")

    # level 1: the file holds one battery struct
    names = []
    for key in mat:
        if not key.startswith("__"):
            names.append(key)
    print("level 1  file              -> " + str(names))

    # level 2: the battery struct holds the list of cycles
    print("level 2  " + battery_name.ljust(17) + " -> field 'cycle' with " +
          str(len(cycles)) + " cycles in time order")

    # level 3: one cycle has four fields
    first_cycle = cycles[0]
    print("level 3  cycle[0]          -> fields: " + str(first_cycle.dtype.names))

    #level 4: the 'data' field holds the measurement series
    data = first_cycle["data"][0, 0]
    print("level 4  cycle[0]['data']  -> series: " + str(data.dtype.names))

    # level 5: finally, numbers
    voltage = get_numbers(data["Voltage_measured"])
    print("level 5  Voltage_measured  -> " + str(voltage.size) + " numbers, first three: " +
          str(voltage[:3].round(3)))
    print("")
    print("so to read one voltage value we go: file -> battery -> cycle -> data -> series -> number")
    print("")


def main():
    if not FILE_PATH.is_file():
        print("File not found: " + str(FILE_PATH), file=sys.stderr)
        sys.exit(1)

    mat = loadmat(str(FILE_PATH), struct_as_record=True, squeeze_me=False)

    names = []
    for key in mat:
        if not key.startswith("__"):
            names.append(key)
    battery_name = names[0] # open the battery box and take the row with all cycles (616 for B0005)
    root = mat[battery_name][0, 0]
    cycles = root["cycle"][0]
    show_structure(mat, battery_name, cycles)

    size_mb = FILE_PATH.stat().st_size / 1e6
    print("file: " + FILE_PATH.name + " (" + str(round(size_mb, 1)) + " MB)")
    print("battery: " + battery_name)
    print("number of cycles: " + str(len(cycles)))
    print("")

    #count the cycles, remember where the first cycle of each type is
    count_charge = 0
    count_discharge = 0
    count_impedance = 0
    first_charge = -1
    first_discharge = -1
    first_impedance = -1

    for i in range(len(cycles)):
        type_name = get_type_name(cycles[i]["type"])
        if type_name == "charge":
            count_charge = count_charge + 1
            if first_charge == -1:
                first_charge = i
        if type_name == "discharge":
            count_discharge = count_discharge + 1
            if first_discharge == -1:
                first_discharge = i
        if type_name == "impedance":
            count_impedance = count_impedance + 1
            if first_impedance == -1:
                first_impedance = i

    print("cycle types (they are mixed together in time order):")
    print("  charge     " + str(count_charge).rjust(4) + "   first at index " + str(first_charge))
    print("  discharge  " + str(count_discharge).rjust(4) + "   first at index " + str(first_discharge))
    print("  impedance  " + str(count_impedance).rjust(4) + "   first at index " + str(first_impedance))
    print("")

    # show how deep the data is nested, this is why the extractor needs unpacking
    first_data = cycles[first_charge]["data"]
    print("nesting example: cycle['data'] has shape " + str(first_data.shape) +
          ", inside it 'Voltage_measured' has shape " + str(first_data[0, 0]["Voltage_measured"].shape) +
          ", and only inside that are the numbers")
    print("")
    # one example of each type: discharge has a ready Capacity, impedance a ready Re
    print_one_cycle(cycles[first_charge], first_charge, "charge")
    print_one_cycle(cycles[first_discharge], first_discharge, "discharge")
    print_one_cycle(cycles[first_impedance], first_impedance, "impedance")


if __name__ == "__main__":
    main()