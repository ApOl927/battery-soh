"""
Plot pairs - draws the box plot of section 3.5:
R2 on the Oxford test cells for all 28 pairs of fine-tuning cells for every model.

Reads the results of: python model_comparison.py --pairs all
Has Polish labels.

Input:  results/comparison_all_pairs.csv
Output: results/pary_ogniw.png
Run:    python plot_pairs.py (from the project root after model_comparison.py)
"""
import pandas as pd
import matplotlib.pyplot as plt
# model name in the csv -> Polish label on the x axis (dict order = box order on the plot)
NAZWY = {
    "RF_no_oxford": "RF\n(NASA)",
    "RF_nasa_oxford": "RF\n(NASA +\n2 ogniwa Oxford)",
    "RF_oxford_only": "RF\n(2 ogniwa Oxford)",
    "cycle_only": "Tylko numer\ncyklu",
    "LSTM_pretrained": "LSTM\nz pre-treningiem",
    "LSTM_random_frozen": "LSTM\nz losowymi\nwagami",
    "LSTM_oxford_only": "LSTM\nbez pre-treningu",
}
res = pd.read_csv("results/comparison_all_pairs.csv")
kolejnosc = list(NAZWY)
fig, ax = plt.subplots(figsize=(11, 5))
ax.boxplot([res.loc[res["model"] == v, "r2"] for v in kolejnosc], showmeans=True)
ax.set_xticks(range(1, len(kolejnosc) + 1), [NAZWY[v] for v in kolejnosc])
ax.set_ylabel("R² (testowe ogniwa Oxford)")
ax.set_title("R² dla 28 par ogniw Oxford użytych do douczania")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/pary_ogniw.png", dpi=150)