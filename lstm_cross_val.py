"""
LSTM cross val - the main LSTM model of the thesis.
Sections 3.4 (network, input windows) and 3.5 (training and comparison).

  input:    10 discharge cycles in a row of one cell 6 features (table in 3.4)
  target:   SoH 21 cycles after the last cycle of the window (future state, not current one)
  network:  2 LSTM layers x 64, dropout 0.2, one linear layer at the end

  training whole network on the 2 Oxford cells with the lowest SoH
            (no pre-training on NASA and no discharge_cycle -> see section 3.5 and model_comparison.py)
  test      the other 6 Oxford cells

Input:  output/MASTER_processed.csv
Output: metrics in the terminal + lstm_plot.png
Run:    python lstm_cross_val.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# fixed seed -> every run gives the same result (R^2 changes with the seed, section 3.7)
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# the 6 input features (table in section 3.4)
# discharge_cycle is not a feature. It means something different in each dataset (sections 3.3 and 3.5)
FEATURE_COLUMNS: list[str] = [
    "avg_temperature_c",
    "resistance_ratio",
    "nominal_capacity",
    "form_factor",
    "temp_variance",
    "voltage_variance",
]
assert "discharge_cycle" not in FEATURE_COLUMNS, "discharge_cycle must not be a model feature (section 3.5)."

# rows with an empty value in these columns are dropped
_BASE_FEATURE_COLS: list[str] = [
    "avg_temperature_c",
    "resistance_ratio",
    "nominal_capacity",
    "form_factor",
    "temp_variance",
    "voltage_variance",
]
TARGET_COLUMN: str = "soh_percentage"
# window = 10 cycles in a row, target= SoH 20 cycles after window
WINDOW_SIZE: int = 10
TARGET_HORIZON: int = 20
BATCH_SIZE: int = 64

#training on the 2 Oxford cells: Adam; 50 epochs, learning rate 0.001
EPOCHS: int = 50
LEARNING_RATE: float = 0.001

# network size: 2 LSTM layers x 64 neurons, dropout 0.2 between them (section 3.4)
LSTM_HIDDEN_SIZE: int = 64
LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2



def _oxford_adaptation_battery_ids(oxford_df: pd.DataFrame, k: int = 2) -> list[str]:
    """Pick the k Oxford cells with the lowest SoH for fine-tuning."""
    min_soh = (
        oxford_df.groupby("battery_id", sort=False)["soh_percentage"]
        .min()
        .rename("min_soh")
        .reset_index()
    )
    # if two cells have the same min SoH, sort by name, so the choice is always the same
    min_soh = min_soh.sort_values(["min_soh", "battery_id"], ascending=[True, True], kind="mergesort")
    return min_soh.head(k)["battery_id"].astype(str).tolist()


def create_sequences(
    df: pd.DataFrame, window_size: int = 10, horizon: int = TARGET_HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    #is 21 cycles after the last window row.
    x_list: list[np.ndarray] = []
    y_list: list[float] = []

    dc_col = "_dc_orig" if "_dc_orig" in df.columns else "discharge_cycle"
    # window = rows i..i+9, target = row i+30 (20 rows skipped after the window)
    span = window_size + horizon

    # one cell at a time + cycles in order
    for _, group in df.groupby("battery_id", sort=True):
        g = group.sort_values("discharge_cycle", kind="mergesort")
        x = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = g[TARGET_COLUMN].to_numpy(dtype=np.float32)
        dc = g[dc_col].to_numpy(dtype=np.float64)
        if len(g) < span + 1:
            continue
        # the window moves by one cycle -> the next exanple
        for i in range(len(g) - span):
            label_idx = i + window_size + horizon
            # a gap in the cycle numbers (removed cycle, Warwick step 20) -> drop this window
            path_dc = dc[i : label_idx + 1]
            if path_dc.size != span + 1:
                continue
            if np.any(np.diff(path_dc) > 1.0):
                continue
            j = i + window_size
            x_list.append(x[i:j])
            y_list.append(float(y[label_idx]))

    # no windows at all -> return empty arrays of the right shape
    if not x_list:
        return (
            np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return np.stack(x_list).astype(np.float32), np.asarray(y_list, dtype=np.float32)


class LSTMRegressor(nn.Module):
    """2 LSTM layers + 1 linear layer: last LSTM output -> one SoH number."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        # batch_first=True: input shape is (windows, 10 cycles, 7 features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # keep only the output after the last (10th) cycle of the window
        last = out[:, -1, :]
        pred = self.fc(last).squeeze(-1)
        return pred


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    # one pass for all batches: with optimizer = training, without = only the loss
    train_mode = optimizer is not None
    if train_mode:
        model.train()
        # LSTM frozen (used by model_comparison.py) -> keep it in eval mode. Dropout stays off
        lstm_has_grad = any(p.requires_grad for p in model.lstm.parameters())
        if not lstm_has_grad:
            model.lstm.eval()
    else:
        model.eval()
    total_loss = 0.0
    n = 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            if train_mode:
                optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if train_mode:
                loss.backward()
                optimizer.step()
            # mean loss over all windows
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total_loss / max(n, 1)


def main() -> None:
    root = Path(__file__).resolve().parent
    master_path = root / "output" / "MASTER_processed.csv"
    df = pd.read_csv(master_path).copy()

    src = df["dataset_source"].astype(str).str.upper()

    # Oxford cells: 2 for fine-tuning, the rest for the test. At least 3 are needed
    oxford_df = df[src == "OXFORD"].copy()
    oxford_ids = sorted(oxford_df["battery_id"].dropna().unique().tolist())
    if len(oxford_ids) < 3:
        raise RuntimeError("Need at least 3 Oxford batteries for 2-shot adaptation + held-out test.")
    adapt_ids = _oxford_adaptation_battery_ids(oxford_df, k=2)
    min_soh_by_cell = oxford_df.groupby("battery_id", sort=False)["soh_percentage"].min()
    adapt_min_soh = {bid: float(min_soh_by_cell[bid]) for bid in adapt_ids}

    # NASA + Warwick + 2 Oxford cells -> to fit the scaler (it's same preprocessing as in model_comparison.py)
    # network is trained on the 2 Oxford cells only; test = other 6 Oxford cells
    df_train = pd.concat(
        [
            df[src.isin(("NASA", "WARWICK"))].copy(),
            oxford_df[oxford_df["battery_id"].isin(adapt_ids)].copy(),
        ],
        ignore_index=True,
    )
    df_test = oxford_df[~oxford_df["battery_id"].isin(adapt_ids)].copy()

    required = _BASE_FEATURE_COLS + [TARGET_COLUMN, "battery_id", "discharge_cycle"]
    df_train = df_train.dropna(subset=required)
    df_test = df_test.dropna(subset=required)


    # keep the real cycle numbers before scaling, gap check needs them
    df_train["_dc_orig"] = df_train["discharge_cycle"].astype(np.float64)
    df_test["_dc_orig"] = df_test["discharge_cycle"].astype(np.float64)

    for col in FEATURE_COLUMNS:
        df_train[col] = df_train[col].astype(np.float64)
        df_test[col] = df_test[col].astype(np.float64)

    # scale features to 0-1; the scale is learned on train data only, test just uses it (3.4)
    scaler = MinMaxScaler()
    df_train.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(df_train[FEATURE_COLUMNS])
    df_test.loc[:, FEATURE_COLUMNS] = scaler.transform(df_test[FEATURE_COLUMNS])

    # SoH is scaled too, at the end predictions go back to %
    target_scaler = MinMaxScaler()
    df_train.loc[:, [TARGET_COLUMN]] = target_scaler.fit_transform(df_train[[TARGET_COLUMN]])
    df_test.loc[:, [TARGET_COLUMN]] = target_scaler.transform(df_test[[TARGET_COLUMN]])

    src_tr = df_train["dataset_source"].astype(str).str.upper()
    # training data: only the 2 Oxford cells
    df_ox_train = df_train[(src_tr == "OXFORD") & (df_train["battery_id"].isin(adapt_ids))].copy()

    X_tr_np, y_tr_np = create_sequences(df_ox_train, window_size=WINDOW_SIZE)
    X_test_np, y_test_np = create_sequences(df_test, window_size=WINDOW_SIZE)

    if X_tr_np.shape[0] == 0:
        raise RuntimeError("No training sequences (Oxford training cells).")
    if X_test_np.shape[0] == 0:
        raise RuntimeError("No test sequences.")

    # numpy -> torch tensors
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    # batches of 64 windows, mixed in a new order every epoch
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    input_size = len(FEATURE_COLUMNS)
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
    # loss = mean squared error on the scaled SoH
    criterion = nn.MSELoss()

    # train whole network on the 2 Oxford cells
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Training (2 Oxford cells, whole network): {len(y_tr_np)} sequences, {EPOCHS} epochs, lr={LEARNING_RATE}")
    for epoch in range(EPOCHS):
        avg_loss = _run_epoch(model, train_loader, criterion, optimizer)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:02d}/{EPOCHS} - loss: {avg_loss:.4f}")

    # test on the held-out Oxford cells, no training here
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    y_pred_chunks: list[np.ndarray] = []
    y_true_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            pb = model(xb).cpu().numpy()
            y_pred_chunks.append(pb.reshape(-1, 1))
            y_true_chunks.append(yb.cpu().numpy().reshape(-1, 1))
    y_pred_scaled = np.concatenate(y_pred_chunks, axis=0)
    y_true_scaled = np.concatenate(y_true_chunks, axis=0)

    # back from 0-1 to SoH in %
    y_pred = target_scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = target_scaler.inverse_transform(y_true_scaled).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    print("\nLSTM: train on 2 Oxford cells -> test on the other Oxford cells")
    print(f"  Oxford training cells (lowest min SoH): {adapt_ids}  min SoH: {adapt_min_soh}")
    print(f"  Training sequences: {len(y_tr_np)} | Test sequences: {len(y_test_np)}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    # one dot =one test window: real SoH on x, predicted SoH on y; red line = perfect model
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, edgecolors="none")
    ax.set_xlabel("Actual SoH (Oxford held-out) [%]")
    ax.set_ylabel("Predicted SoH (LSTM) [%]")
    ax.set_title("LSTM trained on 2 Oxford cells, test on the other Oxford cells")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x (perfect)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    out_png = root / "lstm_plot.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {out_png.resolve()}")


if __name__ == "__main__":
    main()