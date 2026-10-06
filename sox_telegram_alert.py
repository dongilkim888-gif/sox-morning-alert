#!/usr/bin/env python3
"""SOX 전일 수익률 -> 유진테크·원익IPS 당일 종가예상 -> Telegram 알림."""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

KST = ZoneInfo("Asia/Seoul")
SOX = "^SOX"
STOCKS = {"유진테크": "084370.KQ", "원익IPS": "240810.KQ"}
WINDOW = int(os.getenv("MODEL_WINDOW", "504"))
RETRIES = int(os.getenv("DATA_RETRIES", "3"))
RETRY_SECONDS = int(os.getenv("RETRY_SECONDS", "60"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))


def download(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers, start=start, end=end, auto_adjust=False, actions=False,
        progress=False, threads=False, group_by="ticker",
    )
    result: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                continue
            x = data[ticker].copy()
            x.columns = [str(c).title() for c in x.columns]
            result[ticker] = x[["Open", "Close"]].dropna()
    else:
        data.columns = [str(c).title() for c in data.columns]
        result[tickers[0]] = data[["Open", "Close"]].dropna()
    for ticker in result:
        result[ticker].index = pd.to_datetime(result[ticker].index).tz_localize(None)
    return result


def attach_sox(stock: pd.DataFrame, sox: pd.DataFrame) -> pd.DataFrame:
    kr = stock.sort_index().copy()
    us = sox.sort_index().copy()
    us["sox_ret"] = us["Close"].pct_change()
    left = kr.reset_index(names="kr_date").sort_values("kr_date")
    right = us[["sox_ret"]].reset_index(names="us_date").sort_values("us_date")
    merged = pd.merge_asof(
        left, right, left_on="kr_date", right_on="us_date",
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.set_index("kr_date")
    merged["intraday_ret"] = merged["Close"] / merged["Open"] - 1
    return merged.dropna(subset=["sox_ret", "intraday_ret"])


def estimate(history: pd.DataFrame, today: pd.Timestamp, sox_ret: float) -> dict:
    train = history.loc[history.index < today].tail(WINDOW).copy()
    train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=["sox_ret", "intraday_ret"])
    if len(train) < 60:
        raise RuntimeError(f"학습 표본 부족: {len(train)}개")

    x = train.sox_ret.to_numpy(float)
    y = train.intraday_ret.to_numpy(float)
    xw = np.clip(x, *np.quantile(x, [0.01, 0.99]))
    yw = np.clip(y, *np.quantile(y, [0.01, 0.99]))
    beta, alpha = np.polyfit(xw, yw, 1)
    ols_pred = float(alpha + beta * sox_ret)
    resid = yw - (alpha + beta * xw)
    resid_std = float(np.std(resid, ddof=1))

    band_width = max(0.005, float(np.std(x) * 0.25))
    band = train.loc[(train.sox_ret - sox_ret).abs() <= band_width]
    if len(band) < 20:
        band = train.loc[np.sign(train.sox_ret) == np.sign(sox_ret)]
    cond_mean = float(band.intraday_ret.mean()) if len(band) else ols_pred
    cond_std = float(band.intraday_ret.std(ddof=1)) if len(band) > 1 else 0.0
    pred = 0.7 * ols_pred + 0.3 * cond_mean
    uncertainty = max(resid_std, cond_std, 0.005)
    corr = float(np.corrcoef(xw, yw)[0, 1])
    confidence = "높음" if len(train) >= 250 and abs(corr) >= 0.20 and len(band) >= 20 else "중간" if len(train) >= 100 else "낮음"
    return {
        "train_n": len(train), "band_n": len(band), "corr": corr,
        "prediction": pred, "uncertainty": uncertainty, "confidence": confidence,
    }


def make_message(target: pd.Timestamp, results: list[dict]) -> str:
    lines = [f"<b>SOX 아침 분석 알림</b>", f"대상일: {target.date()} (KST)", ""]
    for r in results:
        pred = r["prediction"]
        direction = "상승 우위" if pred > 0.001 else "하락 우위" if pred < -0.001 else "보합권"
        low = r["open"] * (1 + pred - r["uncertainty"])
        high = r["open"] * (1 + pred + r["uncertainty"])
        lines += [
            f"<b>{r['name']}</b> ({r['ticker']})",
            f"전일 SOX: {r['sox_ret']:+.2%}",
            f"당일 시가: {r['open']:,.0f}원",
            f"예상 장중수익률: {pred:+.2%}",
            f"예상 종가: <b>{r['predicted_close']:,.0f}원</b>",
            f"통계 범위: {low:,.0f}~{high:,.0f}원",
            f"판정: {direction} / 신뢰도 {r['confidence']}",
            f"학습표본: {r['train_n']}일, 상관: {r['corr']:+.3f}",
            "",
        ]
    lines.append("※ 연구·분석용 알림이며 투자 권유나 주문 신호가 아닙니다.")
    return "\n".join(lines)


def send_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=20) as response:
        body = response.read().decode("utf-8")
        if response.status >= 300 or '"ok":true' not in body:
            raise RuntimeError(f"Telegram 발송 실패: HTTP {response.status}, {body[:500]}")


def main() -> None:
    now = datetime.now(KST)
    target = pd.Timestamp(now.date())
    # 화~금만 실행: 월요일 한국장은 알림 대상에서 제외
    if target.weekday() not in (1, 2, 3, 4):
        print(f"알림 생략: {target.date()}은 화~금이 아닙니다.")
        return

    start = (target - pd.Timedelta(days=2400)).strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            raw = download(list(STOCKS.values()) + [SOX], start, end)
            if SOX not in raw:
                raise RuntimeError("SOX 데이터 없음")
            results = []
            for name, ticker in STOCKS.items():
                if ticker not in raw:
                    raise RuntimeError(f"{name} 데이터 없음")
                hist = attach_sox(raw[ticker], raw[SOX])
                today = hist.loc[hist.index.normalize() == target]
                if today.empty:
                    raise RuntimeError(f"{name} 당일 시가 미수신")
                row = today.iloc[0]
                stat = estimate(hist, target, float(row.sox_ret))
                open_px = float(row.Open)
                results.append({
                    "name": name, "ticker": ticker, "sox_ret": float(row.sox_ret),
                    "open": open_px, "predicted_close": open_px * (1 + stat["prediction"]), **stat,
                })
            break
        except Exception as exc:
            last_error = exc
            print(f"시도 {attempt}/{RETRIES} 실패: {exc}")
            if attempt < RETRIES:
                time.sleep(RETRY_SECONDS)
    else:
        raise RuntimeError(f"데이터 수집 최종 실패: {last_error}")

    message = make_message(target, results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": now.isoformat(), "target_date": str(target.date()), "results": results, "message": message}
    (OUTPUT_DIR / f"alert_{target:%Y%m%d}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(message.replace("<b>", "").replace("</b>", ""))
    send_telegram(message)
    print("Telegram 발송 완료")


if __name__ == "__main__":
    main()
