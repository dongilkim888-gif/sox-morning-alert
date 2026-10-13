#!/usr/bin/env python3
"""확장형 SOX 다중변수 모델 walk-forward 백테스트."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sox_feature_model import STOCKS, download_all, build_history, estimate


def classify(ret: float) -> str:
    return "up" if ret > 0.001 else "down" if ret < -0.001 else "flat"


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    actual = df["actual_direction"]
    pred = df["pred_direction"]
    nonflat = actual != "flat"
    return {
        "n": int(len(df)),
        "mae_pct": float(df["abs_error_pct"].mean()),
        "rmse_pct": float(np.sqrt((df["error_pct"] ** 2).mean())),
        "mae_krw": float(df["abs_error_krw"].mean()),
        "direction_accuracy_all": float((pred == actual).mean()),
        "direction_accuracy_nonflat": float((pred[nonflat] == actual[nonflat]).mean()) if nonflat.any() else None,
        "range_68_coverage": float(df["in_range_68"].mean()),
        "baseline_zero_intraday_mae_pct": float(df["baseline_abs_error_pct"].mean()),
        "model_beats_zero_intraday_baseline": bool(df["abs_error_pct"].mean() < df["baseline_abs_error_pct"].mean()),
        "mean_sox_corr": float(df["sox_corr"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SOX 다중변수 모델 walk-forward 백테스트")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--window", type=int, default=504)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--outdir", default="backtest_output")
    args = ap.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    raw = download_all((start - pd.Timedelta(days=2400)).strftime("%Y-%m-%d"), (end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    rows_all: list[dict] = []
    by_stock: dict[str, dict] = {}
    for name, ticker in STOCKS.items():
        if ticker not in raw:
            continue
        hist = build_history(raw[ticker], raw)
        dates = hist.loc[(hist.index >= start) & (hist.index <= end)].index
        rows = []
        for dt in dates:
            try:
                stat = estimate(hist, dt, args.window, args.ridge_alpha)
            except RuntimeError:
                continue
            row = hist.loc[hist.index.normalize() == dt.normalize()].iloc[0]
            open_px = float(row["Open"])
            actual_close = float(row["Close"])
            pred_ret = stat["prediction"]
            predicted_close = open_px * (1 + pred_ret)
            uncertainty = stat["uncertainty"]
            actual_ret = actual_close / open_px - 1
            error_krw = predicted_close - actual_close
            record = {
                "date": dt.strftime("%Y-%m-%d"), "name": name, "ticker": ticker,
                "sox_return": float(row.get("sox_ret", np.nan)), "open": open_px,
                "predicted_intraday": pred_ret, "actual_intraday": actual_ret,
                "predicted_close": predicted_close, "actual_close": actual_close,
                "error_krw": error_krw, "abs_error_krw": abs(error_krw),
                "error_pct": predicted_close / actual_close - 1, "abs_error_pct": abs(predicted_close / actual_close - 1),
                "range_low_68": open_px * (1 + pred_ret - uncertainty), "range_high_68": open_px * (1 + pred_ret + uncertainty),
                "in_range_68": bool(open_px * (1 + pred_ret - uncertainty) <= actual_close <= open_px * (1 + pred_ret + uncertainty)),
                "pred_direction": classify(pred_ret), "actual_direction": classify(actual_ret),
                "sox_corr": stat["corr"], "feature_n": stat["feature_n"], "train_n": stat["train_n"],
                "baseline_abs_error_pct": abs(open_px / actual_close - 1),
            }
            rows.append(record); rows_all.append(record)
        by_stock[name] = summarize(pd.DataFrame(rows))
    result_df = pd.DataFrame(rows_all)
    if result_df.empty:
        raise RuntimeError("백테스트 결과가 없습니다.")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(outdir / "predictions.csv", index=False, encoding="utf-8-sig")
    summary = {"settings": vars(args), "by_stock": by_stock, "overall": summarize(result_df)}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SOX 다중변수 walk-forward 백테스트 ===")
    for name, metrics in by_stock.items():
        print(f"{name}: n={metrics.get('n', 0)}, MAE={metrics.get('mae_pct', float('nan')):.2%}, 방향={metrics.get('direction_accuracy_all', float('nan')):.2%}, 범위={metrics.get('range_68_coverage', float('nan')):.2%}, 기준선 대비 개선={metrics.get('model_beats_zero_intraday_baseline')}")
    print(f"결과 저장: {outdir.resolve()}")


if __name__ == "__main__":
    main()
