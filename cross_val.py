"""
Cross val - Random Forest trained on NASA + Warwick, tested on the whole Oxford set.
Sections 3.2 (variant C) and 3.3 (figure: actual vs predicted SoH for Oxford).

Features: avg_temperature_c + resistance_ratio (T + R), RF 200 trees, depth 12, seed 42.
Result: RMSE 14.68, MAE 12.89, R2 -4.181 - predictions stay around 60-80 %,
while the real SoH goes from 61.6 to 99.9 %.

Input: output/MASTER_processed.csv
Output: metrics in the terminal + triple_cross_val_plot.png
Run:    python cross_val.py
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# features: temperature + resistance ratio (T + R), target: SoH
X_COLS: tuple[str, ...] = (
    "avg_temperature_c",
    "resistance_ratio",
)
Y_COL: str = "soh_percentage"
# Warwick name in the dataset_source column
WARWICK_SOURCE: str = "WARWICK"


def main() -> None:
    root = Path(__file__).resolve().parent
    master_path = root / "output" / "MASTER_processed.csv"

    df = pd.read_csv(master_path).copy()

    # train = NASA + Warwick, test = all of Oxford (the model does not see Oxford cell)
    train_data = df[df["dataset_source"].isin(("NASA", WARWICK_SOURCE))].copy()
    test_data = df[df["dataset_source"] == "OXFORD"].copy()

    # drop rows where a feature or SoH is empty
    cols = list(X_COLS) + [Y_COL]
    train_data = train_data.dropna(subset=cols)
    test_data = test_data.dropna(subset=cols)

    X_train = train_data.loc[:, list(X_COLS)]
    y_train = train_data[Y_COL].to_numpy()
    X_test = test_data.loc[:, list(X_COLS)]
    y_test = test_data[Y_COL].to_numpy()

    # the same RF settings as everywhere
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # R2 < 0 = worse than always predicting the mean SoH
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("Triple cross-dataset validation: train on NASA + Warwick, test on Oxford")
    print(f"  Train rows: {len(train_data)} (NASA + Warwick) | Test rows: {len(test_data)} (Oxford)")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    #one dot = one Oxford row. Real SoH on x, predicted SoH on y
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.15, s=8, edgecolors="none")
    ax.set_xlabel("Actual SoH (Oxford) [%]")
    ax.set_ylabel("Predicted SoH (NASA + Warwick train) [%]")
    ax.set_title("NASA + Warwick → Oxford: actual vs predicted SoH (temp + R_ratio)")

    # red dashed line y = x: perfect model puts every point in it
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x (perfect)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # summary numbers on the plot, so figure matches the text
    summary = (
        f"actual SoH:    {y_test.min():.1f} - {y_test.max():.1f} %  (mean {y_test.mean():.1f})\n"
        f"predicted SoH: {y_pred.min():.1f} - {y_pred.max():.1f} %  (mean {y_pred.mean():.1f})\n"
        f"RMSE {rmse:.2f} | MAE {mae:.2f} | R2 {r2:.3f}"
    )
    ax.text(
        0.03, 0.03, summary,
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    out_png = root / "triple_cross_val_plot.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {out_png.resolve()}")


if __name__ == "__main__":
    main()