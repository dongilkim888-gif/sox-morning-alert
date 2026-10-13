#!/usr/bin/env python3
"""시장 변수 8종 기반 아침 종가예상 Telegram 알림."""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from sox_feature_model import STOCKS, build_history, download_all, estimate

KST = ZoneInfo("Asia/Seoul")
WINDOW = int(os.getenv("MODEL_WINDOW", "504"))
RIDGE_ALPHA = float(os.getenv("RIDGE_ALPHA", "10"))
RETRIES = int(os.getenv("DATA_RETRIES", "10"))
RETRY_SECONDS = int(os.getenv("RETRY_SECONDS", "60"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))


def send_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
        if response.status >= 300 or '"ok":true' not in body:
            raise RuntimeError(f"Telegram 발송 실패: HTTP {response.status}")


def direction(ret: float) -> str:
    if ret > 0.001:
        return "상승 우위"
    if ret < -0.001:
        return "하락 우위"
    return "보합권"


def make_message(target: pd.Timestamp, rows: list[dict]) -> str:
    lines = ["SOX 다중변수 아침 분석 알림", f"대상일: {target.date()} (KST)", ""]
    for row in rows:
        low = row["open"] * (1 + row["prediction"] - row["uncertainty"])
        high = row["open"] * (1 + row["prediction"] + row["uncertainty"])
        lines.extend([
            f"{row['name']} ({row['ticker']})",
            f"전일 SOX: {row['sox_ret']:+.2%}",
            f"당일 시가: {row['open']:,.0f}원",
            f"예상 장중수익률: {row['prediction']:+.2%}",
            f"예상 종가: {row['predicted_close']:,.0f}원",
            f"통계 범위: {low:,.0f}~{high:,.0f}원",
            f"판정: {direction(row['prediction'])}",
            f"모델: Ridge 시장변수 | 변수 {row['feature_n']}개 | SOX 상관 {row['corr']:+.3f}",
            "",
        ])
    lines.extend([
        "사용 변수: SOX·환율·나스닥·VIX·KOSPI·KOSDAQ·삼성전자·SK하이닉스·반도체 ETF",
        "※ 실제 외국인·기관 수급 데이터는 사용하지 않습니다.",
        "※ 연구·분석용 알림이며 투자 권유나 주문 신호가 아닙니다.",
    ])
    return "\n".join(lines)


def main() -> None:
    now = datetime.now(KST)
    target = pd.Timestamp(now.date())
    if target.weekday() not in (1, 2, 3, 4):
        print(f"알림 생략: {target.date()}은 화~금이 아닙니다.")
        return

    start = (target - pd.Timedelta(days=2400)).strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            raw = download_all(start, end)
            rows: list[dict] = []
            for name, ticker in STOCKS.items():
                if ticker not in raw:
                    raise RuntimeError(f"{name} 데이터 없음")
                history = build_history(raw[ticker], raw)
                stat = estimate(history, target, WINDOW, RIDGE_ALPHA)
                today = history.loc[history.index.normalize() == target]
                if today.empty:
                    raise RuntimeError(f"{name} 당일 시가 미수신")
                open_px = float(today.iloc[0]["Open"])
                rows.append({
                    "name": name,
                    "ticker": ticker,
                    "sox_ret": float(today.iloc[0].get("sox_ret", 0.0)),
                    "open": open_px,
                    "today_open": open_px,
                    "prediction": stat["prediction"],
                    "predicted_intraday": stat["prediction"],
                    "predicted_close": open_px * (1 + stat["prediction"]),
                    "uncertainty": stat["uncertainty"],
                    "range_low_68": open_px * (1 + stat["prediction"] - stat["uncertainty"]),
                    "range_high_68": open_px * (1 + stat["prediction"] + stat["uncertainty"]),
                    "corr": stat["corr"],
                    "train_n": stat["train_n"],
                    "feature_n": stat["feature_n"],
                    "feature_names": stat["feature_names"],
                    "model": stat["model"],
                })
            break
        except Exception as exc:
            last_error = exc
            print(f"시도 {attempt}/{RETRIES} 실패: {exc}")
            if attempt < RETRIES:
                time.sleep(RETRY_SECONDS)
    else:
        raise RuntimeError(f"데이터 수집 최종 실패: {last_error}")

    message = make_message(target, rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": now.isoformat(),
        "target_date": str(target.date()),
        "model": "ridge_market_proxy",
        "results": rows,
        "message": message,
    }
    path = OUTPUT_DIR / f"alert_{target:%Y%m%d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(message)
    send_telegram(message)
    print("Telegram 발송 완료")


if __name__ == "__main__":
    main()
