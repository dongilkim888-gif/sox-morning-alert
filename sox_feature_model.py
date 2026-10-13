#!/usr/bin/env python3
"""SOX 단일변수 모델을 시장·환율·국내 반도체 프록시로 확장한 공통 모델."""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import yfinance as yf

SOX = "^SOX"
STOCKS: Dict[str, str] = {"유진테크": "084370.KQ", "원익IPS": "240810.KQ"}
MARKET_TICKERS: Dict[str, str] = {
    "sox": "^SOX",
    "nasdaq": "^IXIC",
    "usdkrw": "KRW=X",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "kospi": "^KS11",
    "kosdaq": "^KQ11",
    "samsung": "005930.KS",
    "skhynix": "000660.KS",
    "semi_etf": "091160.KS",  # KODEX 반도체 프록시
}


def download_all(start: str, end: str) -> dict[str, pd.DataFrame]:
    tickers = list(STOCKS.values()) + list(MARKET_TICKERS.values())
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False, actions=False,
        progress=False, threads=False, group_by="ticker",
    )
    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for ticker in tickers:
            if ticker not in level0:
                continue
            frame = raw[ticker].copy()
            frame.columns = [str(c).title() for c in frame.columns]
            wanted = [c for c in ["Open", "Close"] if c in frame.columns]
            if wanted:
                result[ticker] = frame[wanted].dropna(how="all")
    else:
        raw.columns = [str(c).title() for c in raw.columns]
        result[tickers[0]] = raw[[c for c in ["Open", "Close"] if c in raw.columns]].dropna(how="all")
    for ticker, frame in result.items():
        idx = pd.to_datetime(frame.index)
        result[ticker].index = idx.tz_localize(None) if idx.tz is not None else idx
        result[ticker] = result[ticker].sort_index()
    return result


def _us_feature(series: pd.DataFrame, kr_dates: pd.Index, name: str) -> pd.Series:
    x = series["Close"].pct_change().dropna().rename(name).rename_axis("us_date").reset_index()
    left = pd.DataFrame(index=kr_dates).reset_index(names="kr_date").sort_values("kr_date")
    x = x.sort_values("us_date")
    merged = pd.merge_asof(left, x, left_on="kr_date", right_on="us_date", direction="backward", allow_exact_matches=False)
    return pd.Series(merged[name].to_numpy(), index=kr_dates, name=name)


def build_history(stock: pd.DataFrame, all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """현재 한국 거래일에 예측시점 이전에 알 수 있는 특징량만 연결한다."""
    kr = stock[[c for c in ["Open", "Close"] if c in stock.columns]].copy().sort_index()
    out = kr.copy()
    kr_dates = kr.index

    # 미국/글로벌 지표: 가장 최근 미국 세션의 수익률을 다음 한국장에 연결
    for name, ticker in MARKET_TICKERS.items():
        frame = all_data.get(ticker)
        if frame is None or "Close" not in frame.columns:
            continue
        if name in {"kospi", "kosdaq", "samsung", "skhynix", "semi_etf"}:
            # 국내 지표: 현재일에는 전일 수익률만 사용(look-ahead 방지)
            out[f"{name}_ret"] = frame["Close"].pct_change().shift(1).reindex(kr_dates)
        else:
            out[f"{name}_ret"] = _us_feature(frame, kr_dates, f"{name}_ret")

    out["intraday_ret"] = out["Close"] / out["Open"] - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def estimate(history: pd.DataFrame, today: pd.Timestamp, window: int = 504, ridge_alpha: float = 10.0) -> dict:
    feature_cols = [c for c in history.columns if c.endswith("_ret") and c != "intraday_ret"]
    train = history.loc[history.index < today].tail(window).copy()
    train = train.dropna(subset=["intraday_ret"])
    if len(train) < 60:
        raise RuntimeError(f"학습 표본 부족: {len(train)}개")
    if not feature_cols:
        raise RuntimeError("사용 가능한 추가 변수 데이터가 없습니다.")

    # 특징량 결측은 학습기간 중앙값으로 보간. 현재일 입력도 같은 기준을 적용.
    medians = train[feature_cols].median().fillna(0.0)
    x = train[feature_cols].fillna(medians).to_numpy(float)
    y = train["intraday_ret"].to_numpy(float)
    y = np.clip(y, *np.quantile(y, [0.01, 0.99]))
    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    z = (x - means) / scales
    # 표준화된 ridge 회귀: 절편은 y 평균으로 별도 처리
    yc = y - y.mean()
    penalty = np.eye(z.shape[1]) * ridge_alpha
    beta = np.linalg.solve(z.T @ z + penalty, z.T @ yc)
    current = history.loc[history.index.normalize() == pd.Timestamp(today).normalize()]
    if current.empty:
        raise RuntimeError("예측 대상일의 시가 데이터가 없습니다.")
    row = current.iloc[0]
    x_now = row[feature_cols].fillna(medians).to_numpy(float)
    prediction = float(y.mean() + ((x_now - means) / scales) @ beta)
    residual = y - (y.mean() + z @ beta)
    uncertainty = max(float(np.std(residual, ddof=1)), 0.005)
    sox_corr = float(np.corrcoef(train["sox_ret"].fillna(0), train["intraday_ret"])[0, 1]) if "sox_ret" in train else float("nan")
    return {
        "prediction": prediction,
        "uncertainty": uncertainty,
        "corr": sox_corr,
        "train_n": int(len(train)),
        "feature_n": int(len(feature_cols)),
        "feature_names": feature_cols,
        "model": "ridge_multi_factor",
        "ridge_alpha": ridge_alpha,
        "imputation": "training_median",
    }
