"""
Comparison of the LSTM with simpler models (thesis section 3.5).

All models get identical data: the same windows (10 cycles -> SoH 20 cycles later),
the same features, the same scaler and the same Oxford test cells.
Only the model and the training data differ.

  LSTM_pretrained     thesis model: pre-trained on NASA, last layer re-trained on 2 Oxford cells
  LSTM_oxford_only    the same network trained from scratch on the 2 Oxford cells only
  LSTM_random_frozen  like LSTM_pretrained, but random weights instead of NASA pre-training
  RF_nasa_oxford      Random Forest trained on NASA + Warwick + the same 2 Oxford cells
  RF_oxford_only      Random Forest trained on the 2 Oxford cells only
  RF_no_oxford        Random Forest trained on NASA + Warwick only (no Oxford data)
  cycle_only          SoH predicted from the cycle number alone (2nd-degree polynomial, no network)

Used in the thesis: 3.4 (--no_cycle), 3.5 (--pairs all -> box plot in plot_pairs.py),
3.7 (spread over 10 seeds and over the 28 cell pairs).

To run from the project root:
  python model_comparison.py --seeds 0 1 2 3 4 5 6 7 8 9      # 10 seeds, default cell pair
  python model_comparison.py --pairs all                       # all 28 Oxford cell pairs, seed 42
  python model_comparison.py --no_cycle --models LSTM_pretrained LSTM_oxford_only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # also works with python -m
from common import OUT, TARGET, load_master, metrics, oxford_measured, rf

import argparse
import itertools
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# the LSTM, its settings and helpers are taken from lstm_cross_val.py. Then both files use the same model
import lstm_cross_val as L

ALL_MODELS = ["LSTM_pretrained", "LSTM_oxford_only", "LSTM_random_frozen",
              "RF_nasa_oxford", "RF_oxford_only", "RF_no_oxford", "cycle_only"]


def make_windows(df: pd.DataFrame, feats: list[str]):
    """Same logic as L.create_sequences, but also returns a description of every window."""
    X, y, meta = [], [], []
    w, h = L.WINDOW_SIZE, L.TARGET_HORIZON
    span = w + h
    for bid, g in df.groupby("battery_id", sort=True):
        g = g.sort_values("discharge_cycle", kind="mergesort")
        if len(g) < span + 1:
            continue
        x = g[feats].to_numpy(dtype=np.float32)
        t = g[TARGET].to_numpy(dtype=np.float32)
        dc = g["_dc_orig"].to_numpy(dtype=np.float64)
        # _measured: is the target a real Oxford measurement or an interpolated row
        ms = g["_measured"].to_numpy(dtype=bool)
        src = str(g["dataset_source"].iloc[0])
        for i in range(len(g) - span):
            lab = i + span
            # skip windows with a gap in the cycle numbers (same rule as in lstm_cross_val)
            if np.any(np.diff(dc[i:lab + 1]) > 1.0):
                continue
            X.append(x[i:i + w])
            y.append(t[lab])
            # description of the window: which cell, which lab, target cycle, real or interpolated
            meta.append((bid, src, dc[lab], ms[lab]))
    X = np.stack(X).astype(np.float32) if X else np.empty((0, w, len(feats)), np.float32)
    meta = pd.DataFrame(meta, columns=["battery_id", "source", "label_cycle", "label_measured"])
    return X, np.asarray(y, dtype=np.float32), meta


def prepare(df_all: pd.DataFrame, adapt_ids: list[str], feats: list[str]) -> dict:
    """Same split and scaling as in lstm_cross_val.main()."""
    # train = NASA + Warwick + the 2 chosen Oxford cells. Test = the other Oxford cells
    src = df_all["dataset_source"].astype(str).str.upper()
    ox = df_all[src == "OXFORD"]
    tr = pd.concat([df_all[src.isin(["NASA", "WARWICK"])], ox[ox["battery_id"].isin(adapt_ids)]],
                   ignore_index=True)
    te = ox[~ox["battery_id"].isin(adapt_ids)].copy()
    req = L._BASE_FEATURE_COLS + [TARGET, "battery_id", "discharge_cycle"]
    tr = tr.dropna(subset=req)
    te = te.dropna(subset=req)
    # keep the real cycle numbers before scaling, the gap check needs them
    for d in (tr, te):
        d["_dc_orig"] = d["discharge_cycle"].astype(np.float64)
        for c in feats:
            d[c] = d[c].astype(np.float64)
    # scale features and SoH to 0-1, the scale is learned on train only
    sx, sy = MinMaxScaler(), MinMaxScaler()
    tr.loc[:, feats] = sx.fit_transform(tr[feats])
    te.loc[:, feats] = sx.transform(te[feats])
    tr.loc[:, [TARGET]] = sy.fit_transform(tr[[TARGET]])
    te.loc[:, [TARGET]] = sy.transform(te[[TARGET]])
    s_tr = tr["dataset_source"].astype(str).str.upper()
    # pre = pre-training windows, ft = 2 Oxford cells, te = test, sy = to turn SoH back into %
    return {
        "pre": make_windows(tr[s_tr.isin(["NASA", "WARWICK"])], feats),
        "ft": make_windows(tr[s_tr == "OXFORD"], feats),
        "te": make_windows(te, feats),
        "sy": sy,
    }


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_lstm(P: dict, seed: int, mode: str) -> np.ndarray:
    """mode: 'transfer' (as in the thesis), 'random' (no pre-training, fc only), 'ox_only' (whole net from scratch)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    (Xp, yp, _), (Xf, yf, _), (Xt, _, _) = P["pre"], P["ft"], P["te"]

    def loader(X, y):
        return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=L.BATCH_SIZE, shuffle=True)

    pre_loader, ft_loader = loader(Xp, yp), loader(Xf, yf)
    model = L.LSTMRegressor(input_size=Xp.shape[2])
    crit = nn.MSELoss()
    # phase 1 only for 'transfer': train the whole network on pre-training windows
    if mode == "transfer":
        opt = torch.optim.Adam(model.parameters(), lr=L.PRETRAIN_LR)
        for _ in range(L.PRETRAIN_EPOCHS):
            L._run_epoch(model, pre_loader, crit, opt)
    # 'transfer' and 'random': LSTM frozen, only the last linear layer learns
    if mode in ("transfer", "random"):
        for p in model.lstm.parameters():
            p.requires_grad = False
        model.lstm.eval()
        opt = torch.optim.Adam(model.fc.parameters(), lr=L.FINETUNE_LR)
    else:  # ox_only
        opt = torch.optim.Adam(model.parameters(), lr=L.PRETRAIN_LR)
    # phase 2: training on the 2 Oxford cells
    for _ in range(L.FINETUNE_EPOCHS):
        L._run_epoch(model, ft_loader, crit, opt)
    #predict the test windows in parts of 4096
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), 4096):
            out.append(model(torch.tensor(Xt[i:i + 4096])).numpy())
    return np.concatenate(out)


def predict(variant: str, P: dict, seed: int, feats: list[str]) -> np.ndarray:
    (Xp, yp, _), (Xf, yf, _), (Xt, _, _) = P["pre"], P["ft"], P["te"]
    #RF does not read sequences, so a window (10 cycles x features) becomes one flat row
    flat = lambda X: X.reshape(len(X), -1)
    if variant == "RF_no_oxford":
        return rf(seed).fit(flat(Xp), yp).predict(flat(Xt))
    if variant == "RF_nasa_oxford":
        return rf(seed).fit(np.vstack([flat(Xp), flat(Xf)]), np.concatenate([yp, yf])).predict(flat(Xt))
    if variant == "RF_oxford_only":
        return rf(seed).fit(flat(Xf), yf).predict(flat(Xt))
    if variant == "cycle_only":
        # only the cycle number of the last window row;a 2-nd degree curve fitted on the 2 Oxford cells
        k = feats.index("discharge_cycle")
        m = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(Xf[:, -1, [k]], yf)
        return m.predict(Xt[:, -1, [k]])
    # the three LSTM variants differ only in the mode
    mode = {"LSTM_pretrained": "transfer", "LSTM_random_frozen": "random", "LSTM_oxford_only": "ox_only"}[variant]
    return run_lstm(P, seed, mode)


def main() -> None:
    # settings from the command line (see docstring examples)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--pairs", choices=["default", "all"], default="default")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--no_cycle", action="store_true", help="drop discharge_cycle from the features")
    args = ap.parse_args()

    # --no_cycle: no discharge_cycle feature. cycle_only makes no sense and skipped
    feats = [c for c in L.FEATURE_COLUMNS if not (args.no_cycle and c == "discharge_cycle")]
    variants = [v for v in args.models if not (args.no_cycle and v == "cycle_only")]

    df = load_master()
    df["_measured"] = oxford_measured(df)
    ox = df[df["dataset_source"].astype(str).str.upper() == "OXFORD"]
    # default: the 2 Oxford cells with the lowest SoH; all: every pair of the 8 cells (28 pairs)
    if args.pairs == "default":
        pairs = [L._oxford_adaptation_battery_ids(ox, k=2)]
    else:
        pairs = [list(p) for p in itertools.combinations(sorted(ox["battery_id"].unique()), 2)]

    results = []
    for pair in pairs:
        P = prepare(df, pair, feats)
        meta_pre, meta_te = P["pre"][2], P["te"][2]
        #real SoH back in %, for the metrics and the printout
        y_true = P["sy"].inverse_transform(P["te"][1].reshape(-1, 1)).ravel()
        y_ft = P["sy"].inverse_transform(P["ft"][1].reshape(-1, 1)).ravel()
        ms = meta_te["label_measured"].to_numpy()
        print(f"\n=== adaptation pair: {pair} ===")
        print(f"pre-training windows by source: {meta_pre['source'].value_counts().to_dict()}")
        print(f"fine-tune windows: {len(y_ft)} | test windows: {len(y_true)} "
              f"(with measured label: {int(ms.sum())}) | test cells: {sorted(meta_te['battery_id'].unique())}")
        print(f"SoH range: fine-tune {y_ft.min():.1f}-{y_ft.max():.1f} % | test {y_true.min():.1f}-{y_true.max():.1f} %")
        for seed in args.seeds:
            for v in variants:
                if v == "cycle_only" and seed != args.seeds[0]:
                    continue  # deterministic - one run is enough
                pred = P["sy"].inverse_transform(predict(v, P, seed, feats).reshape(-1, 1)).ravel()
                # metrics for all test windows and only for windows with a real measured target
                m_all = metrics(y_true, pred)
                m_ms = metrics(y_true[ms], pred[ms])
                print(f"  seed {seed:3d} | {v:18s} | R2 {m_all['r2']:7.3f} | RMSE {m_all['rmse']:5.2f} | MAE {m_all['mae']:5.2f}")
                results.append({"pair": "+".join(pair), "seed": seed, "model": v, **m_all,
                                "r2_measured": m_ms["r2"]})

    # save every run, then mean / std / worst/ best R2 per model
    res = pd.DataFrame(results)
    name = "comparison" + ("_no_cycle" if args.no_cycle else "") + ("_all_pairs" if args.pairs == "all" else "_default_pair") + ".csv"
    res.to_csv(OUT / name, index=False)
    summary = res.groupby("model", sort=False).agg(
        runs=("r2", "size"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        r2_worst=("r2", "min"),
        r2_best=("r2", "max"),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        r2_real_measurements=("r2_measured", "mean"),
    )
    print("\n=== summary ===")
    print("runs: number of runs (seeds or cell pairs) | r2_std: how much R2 changes between runs (the +- value)")
    print("r2_real_measurements: R2 only for windows whose target is a real Oxford measurement, not interpolated")
    print(summary.round(3).fillna("-").to_string())
    print(f"\nsaved: {OUT / name}")

    # box plot only if there are several runs per model (thesis: plot_pairs.py)
    if len(res) > len(variants):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot([res.loc[res["model"] == v, "r2"].to_numpy() for v in variants], showmeans=True)
        ax.set_xticks(range(1, len(variants) + 1), variants)
        ax.set_ylabel("R2 (Oxford, test cells)")
        ax.set_title(f"R2 spread: seeds {args.seeds}, pairs: {args.pairs}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / name.replace(".csv", ".png"), dpi=150)


if __name__ == "__main__":
    main()