#!/usr/bin/env python3
"""아침 예측 JSON과 장 마감 후 실제 종가를 비교해 오차를 누적 기록한다."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

KST = ZoneInfo("Asia/Seoul")
STOCKS = {"유진테크": "084370.KQ", "원익IPS": "240810.KQ"}
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
EVAL_DIR = Path(os.getenv("EVAL_DIR", "evaluation"))
CLOSE_RETRIES = int(os.getenv("CLOSE_RETRIES", "3"))
CLOSE_RETRY_SECONDS = int(os.getenv("CLOSE_RETRY_SECONDS", "30"))


def write_status(target: pd.Timestamp, status: str, reason: str) -> None:
    """평가가 생략되거나 실패해도 원인을 저장한다."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"status_{target:%Y%m%d}.json"
    path.write_text(
        json.dumps(
            {"date": str(target.date()), "status": status, "reason": reason},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_closes(tickers: list[str], target: pd.Timestamp) -> dict[str, float]:
    """target 날짜의 실제 종가를 조회한다. 데이터 지연을 고려해 재시도한다."""
    start = target.strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    last_error: Exception | None = None

    for attempt in range(1, CLOSE_RETRIES + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                group_by="ticker",
            )
            closes: dict[str, float] = {}
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for ticker in tickers:
                    if ticker not in level0:
                        continue
                    frame = raw[ticker]
                    if "Close" in frame.columns:
                        series = frame["Close"].dropna()
                        if not series.empty:
                            closes[ticker] = float(series.iloc[0])
            elif tickers and "Close" in raw.columns:
                series = raw["Close"].dropna()
                if not series.empty:
                    closes[tickers[0]] = float(series.iloc[0])

            if closes:
                return closes
            last_error = RuntimeError("조회 결과에 종가가 없습니다.")
        except Exception as exc:  # 네트워크·Yahoo 응답 오류
            last_error = exc
        if attempt < CLOSE_RETRIES:
            time.sleep(CLOSE_RETRY_SECONDS)

    print(f"실제 종가 조회 실패: {last_error}")
    return {}


def pct(x: float) -> float:
    """소수 수익률을 사람이 읽는 퍼센트 포인트로 저장한다. 0.012 -> 1.2"""
    return round(float(x) * 100, 6)


def get_float(row: dict, *keys: str) -> float:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return float(row[key])
    raise KeyError(f"예측 JSON에 필요한 필드가 없습니다: {keys}")


def as_bool(value) -> bool:
    """CSV 재로드 시 문자열 'False'가 True로 변환되는 문제를 방지한다."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def classify(ret: float) -> str:
    if ret > 0.001:
        return "up"
    if ret < -0.001:
        return "down"
    return "flat"


def send_telegram(text: str) -> None:
    """SEND_TELEGRAM=1일 때만 오후 평가 결과를 Telegram으로 발송한다."""
    if os.getenv("SEND_TELEGRAM", "0") != "1":
        return
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
        if response.status >= 300 or '"ok":true' not in body:
            raise RuntimeError(f"Telegram 사후평가 발송 실패: HTTP {response.status}")


def make_telegram_message(target: pd.Timestamp, rows: list[dict], report: dict) -> str:
    lines = [f"[SOX 사후평가] {target.date()}", ""]
    for row in rows:
        lines.extend([
            row["name"],
            f"예상 종가: {row['predicted_close']:,.0f}원",
            f"실제 종가: {row['actual_close']:,.0f}원",
            f"오차: {row['error_krw']:+,.0f}원 ({row['error_pct']:+.2f}%)",
            f"예상범위 포함: {'예' if row['in_range_68'] else '아니오'}",
            "",
        ])
    lines.append(f"누적 평가표본: {report['n']}개")
    for name, metrics in report["by_stock"].items():
        lines.append(
            f"{name}: MAE {metrics['mae_pct_points']:.2f}% | "
            f"방향 적중률 {metrics['direction_accuracy_all']:.1%} | "
            f"범위 포함률 {metrics['range_68_coverage']:.1%}"
        )
    lines.append("")
    lines.append("※ 연구·분석용 결과이며 투자 권유나 주문 신호가 아닙니다.")
    return "\n".join(lines)


def main() -> None:
    global OUTPUT_DIR, EVAL_DIR
    parser = argparse.ArgumentParser(description="아침 예측 JSON과 실제 종가를 비교하는 사후평가")
    parser.add_argument("--date", help="평가 날짜 YYYY-MM-DD. 미지정 시 현재 KST 날짜")
    parser.add_argument("--allow-nonbusiness", action="store_true", help="주말/휴일도 조회를 시도")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="아침 예측 JSON 폴더")
    parser.add_argument("--eval-dir", default=str(EVAL_DIR), help="평가 결과 폴더")
    args = parser.parse_args()

    target = pd.Timestamp(args.date) if args.date else pd.Timestamp(datetime.now(KST).date())
    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir)
    # 수동 옵션을 반영하기 위해 전역 경로를 이번 실행에만 교체한다.
    OUTPUT_DIR = output_dir
    EVAL_DIR = eval_dir

    if target.weekday() not in (0, 1, 2, 3, 4) and not args.allow_nonbusiness:
        print(f"사후평가 생략: {target.date()}은 평일이 아닙니다.")
        return

    prediction_path = OUTPUT_DIR / f"alert_{target:%Y%m%d}.json"
    if not prediction_path.exists():
        reason = f"오늘 아침 예측 파일이 없어 평가를 생략합니다: {prediction_path}"
        print(reason)
        write_status(target, "skipped", reason)
        return

    try:
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        predictions = {x["ticker"]: x for x in payload.get("results", []) if "ticker" in x}
    except Exception as exc:
        reason = f"아침 예측 JSON을 읽지 못해 평가를 생략합니다: {exc}"
        print(reason)
        write_status(target, "error", reason)
        return

    if not predictions:
        reason = "예측 결과가 비어 있어 평가를 생략합니다."
        print(reason)
        write_status(target, "skipped", reason)
        return

    closes = get_closes(list(STOCKS.values()), target)
    rows: list[dict] = []

    for name, ticker in STOCKS.items():
        if ticker not in predictions:
            print(f"예측에 {name}({ticker})이 없어 생략합니다.")
            continue
        if ticker not in closes:
            print(f"실제 종가를 받지 못해 생략합니다: {name}({ticker})")
            continue

        p = predictions[ticker]
        try:
            actual_close = float(closes[ticker])
            predicted_close = get_float(p, "predicted_close")
            open_px = get_float(p, "today_open", "open")
            predicted_intraday = get_float(p, "predicted_intraday", "prediction")
            sox_previous_return = get_float(p, "sox_previous_return", "sox_ret")
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"{name} 예측 JSON 필드 오류: {exc}"
            print(reason)
            write_status(target, "error", reason)
            continue

        actual_intraday = actual_close / open_px - 1
        error_krw = predicted_close - actual_close
        error_pct = predicted_close / actual_close - 1
        uncertainty = float(p.get("uncertainty", 0.0) or 0.0)
        range_low = float(p.get("range_low_68", open_px * (1 + predicted_intraday - uncertainty)))
        range_high = float(p.get("range_high_68", open_px * (1 + predicted_intraday + uncertainty)))

        rows.append({
            "date": str(target.date()),
            "name": name,
            "ticker": ticker,
            "sox_previous_return_pct": pct(sox_previous_return),
            "open": open_px,
            "predicted_close": predicted_close,
            "actual_close": actual_close,
            "error_krw": round(error_krw, 2),
            "abs_error_krw": round(abs(error_krw), 2),
            "error_pct": pct(error_pct),
            "abs_error_pct": pct(abs(error_pct)),
            "predicted_intraday_pct": pct(predicted_intraday),
            "actual_intraday_pct": pct(actual_intraday),
            "range_low_68": range_low,
            "range_high_68": range_high,
            "in_range_68": bool(range_low <= actual_close <= range_high),
            "pred_direction": classify(predicted_intraday),
            "actual_direction": classify(actual_intraday),
            "confidence": p.get("confidence", ""),
            "train_n": p.get("train_n", ""),
            "corr": p.get("corr", p.get("historical_corr", "")),
            "evaluated_at_kst": datetime.now(KST).isoformat(),
        })

    if not rows:
        reason = "실제 종가를 받은 종목이 없어 평가를 생략합니다."
        print(reason)
        write_status(target, "skipped", reason)
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "daily_evaluation.csv"
    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.sort_values(["date", "ticker"])
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = {"as_of": str(target.date()), "n": int(len(combined)), "by_stock": {}}
    for name, group in combined.groupby("name"):
        abs_err = pd.to_numeric(group["abs_error_pct"], errors="coerce")
        actual = group["actual_direction"].astype(str)
        pred = group["pred_direction"].astype(str)
        nonflat = actual != "flat"
        coverage = group["in_range_68"].map(as_bool)
        report["by_stock"][name] = {
            "n": int(len(group)),
            "mae_pct_points": round(float(abs_err.mean()), 6),
            "mae_krw": round(float(pd.to_numeric(group["abs_error_krw"], errors="coerce").mean()), 2),
            "direction_accuracy_all": round(float((pred == actual).mean()), 6),
            "direction_accuracy_nonflat": round(float((pred[nonflat] == actual[nonflat]).mean()), 6) if nonflat.any() else None,
            "range_68_coverage": round(float(coverage.mean()), 6),
            "mean_error_pct_points": round(float(pd.to_numeric(group["error_pct"], errors="coerce").mean()), 6),
        }

    (EVAL_DIR / "evaluation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    telegram_message = make_telegram_message(target, rows, report)
    send_telegram(telegram_message)

    print(f"사후평가 완료: {target.date()}")
    for row in rows:
        print(
            f"{row['name']}: 예상 {row['predicted_close']:,.0f}원 / "
            f"실제 {row['actual_close']:,.0f}원 / "
            f"오차 {row['error_krw']:+,.0f}원 ({row['error_pct']:+.2f}%)"
        )
    print(f"누적 평가표본: {len(combined)}개 | 저장: {csv_path}")
    if os.getenv("SEND_TELEGRAM", "0") == "1":
        print("Telegram 사후평가 발송 완료")


if __name__ == "__main__":
    main()
