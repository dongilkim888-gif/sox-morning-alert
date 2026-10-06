#!/usr/bin/env python3
"""아침 예측 JSON과 장 마감 후 실제 종가를 비교해 오차를 누적 기록한다."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

KST = ZoneInfo("Asia/Seoul")
STOCKS = {"유진테크": "084370.KQ", "원익IPS": "240810.KQ"}
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
EVAL_DIR = Path(os.getenv("EVAL_DIR", "evaluation"))


def get_closes(tickers: list[str], target: pd.Timestamp) -> dict[str, float]:
    start = target.strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False, actions=False,
        progress=False, threads=False, group_by="ticker",
    )
    closes: dict[str, float] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            x = raw[ticker]
            if "Close" in x and not x["Close"].dropna().empty:
                closes[ticker] = float(x["Close"].dropna().iloc[0])
    elif tickers and "Close" in raw and not raw["Close"].dropna().empty:
        closes[tickers[0]] = float(raw["Close"].dropna().iloc[0])
    return closes


def pct(x: float) -> float:
    return round(float(x) * 100, 6)


def main() -> None:
    target = pd.Timestamp(datetime.now(KST).date())
    if target.weekday() not in (0, 1, 2, 3, 4):
        print(f"사후평가 생략: {target.date()}은 평일이 아닙니다.")
        return

    prediction_path = OUTPUT_DIR / f"alert_{target:%Y%m%d}.json"
    if not prediction_path.exists():
        print(f"오늘 아침 예측 파일이 없어 평가를 생략합니다: {prediction_path}")
        return
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = {x["ticker"]: x for x in payload.get("results", [])}
    if not predictions:
        print("예측 결과가 비어 있어 평가를 생략합니다.")
        return

    closes = get_closes(list(STOCKS.values()), target)
    rows: list[dict] = []
    for name, ticker in STOCKS.items():
        if ticker not in predictions:
            print(f"예측에 {name}이 없어 생략합니다.")
            continue
        if ticker not in closes:
            print(f"실제 종가를 받지 못해 생략합니다: {name}")
            continue
        p = predictions[ticker]
        actual_close = closes[ticker]
        predicted_close = float(p["predicted_close"])
        open_px = float(p["today_open"] if "today_open" in p else p["open"])
        predicted_intraday = float(p["predicted_intraday"] if "predicted_intraday" in p else p["prediction"])
        actual_intraday = actual_close / open_px - 1
        error_krw = predicted_close - actual_close
        error_pct = predicted_close / actual_close - 1
        range_low = float(p.get("range_low_68", open_px * (1 + predicted_intraday - p.get("uncertainty", 0))))
        range_high = float(p.get("range_high_68", open_px * (1 + predicted_intraday + p.get("uncertainty", 0))))
        rows.append({
            "date": str(target.date()), "name": name, "ticker": ticker,
            "sox_previous_return_pct": pct(float(p["sox_previous_return"])),
            "open": open_px, "predicted_close": predicted_close, "actual_close": actual_close,
            "error_krw": round(error_krw, 2), "abs_error_krw": round(abs(error_krw), 2),
            "error_pct": pct(error_pct), "abs_error_pct": pct(abs(error_pct)),
            "predicted_intraday_pct": pct(predicted_intraday), "actual_intraday_pct": pct(actual_intraday),
            "range_low_68": range_low, "range_high_68": range_high,
            "in_range_68": bool(range_low <= actual_close <= range_high),
            "pred_direction": "up" if predicted_intraday > 0.001 else "down" if predicted_intraday < -0.001 else "flat",
            "actual_direction": "up" if actual_intraday > 0.001 else "down" if actual_intraday < -0.001 else "flat",
            "confidence": p.get("confidence", ""), "train_n": p.get("train_n", ""),
            "corr": p.get("corr", p.get("historical_corr", "")),
            "evaluated_at_kst": datetime.now(KST).isoformat(),
        })

    if not rows:
        print("평가할 종목이 없습니다.")
        return
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "daily_evaluation.csv"
    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(["date", "ticker"])
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = {"as_of": str(target.date()), "n": int(len(combined)), "by_stock": {}}
    for name, group in combined.groupby("name"):
        abs_err = pd.to_numeric(group["abs_error_pct"], errors="coerce")
        actual = group["actual_direction"].astype(str)
        pred = group["pred_direction"].astype(str)
        nonflat = actual != "flat"
        report["by_stock"][name] = {
            "n": int(len(group)),
            "mae_pct": round(float(abs_err.mean()), 6),
            "mae_krw": round(float(pd.to_numeric(group["abs_error_krw"], errors="coerce").mean()), 2),
            "direction_accuracy_all": round(float((pred == actual).mean()), 6),
            "direction_accuracy_nonflat": round(float((pred[nonflat] == actual[nonflat]).mean()), 6) if nonflat.any() else None,
            "range_68_coverage": round(float(group["in_range_68"].astype(bool).mean()), 6),
            "mean_error_pct": round(float(pd.to_numeric(group["error_pct"], errors="coerce").mean()), 6),
        }
    (EVAL_DIR / "evaluation_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"사후평가 완료: {target.date()}")
    for row in rows:
        print(f"{row['name']}: 예상 {row['predicted_close']:,.0f}원 / 실제 {row['actual_close']:,.0f}원 / 오차 {row['error_krw']:+,.0f}원 ({row['error_pct']:+.2f}%)")
    print(f"누적 평가표본: {len(combined)}개 | 저장: {csv_path}")


if __name__ == "__main__":
    main()
