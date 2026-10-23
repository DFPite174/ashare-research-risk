"""
企业级量化投研平台 - 动态 Barra 风格因子载荷全自动计算器 (Dynamic Barra Factor Calculator)
每个交易日从行情与财务数据中自动滚动计算各标的的 5 大核心 Barra 风格暴露度：
1. SIZE (规模因子): 对数总市值 / 20日成交额规模对数
2. VALUE (价值因子): 账面市值比 (BPS / Close) 或估值修复代理
3. MOMENTUM (动量因子): 过去 20 日时序动量回报率
4. VOLATILITY (波动率因子): 过去 20 日收益率波动率标准差
5. LIQUIDITY (流动性因子): 过去 20 日日均换手/成交额对数

每日截面严格执行 Winsorize 去极值 (3-Sigma/MAD) 与 Z-Score 标准化 (0均值1方差)，
杜绝静态固定矩阵，实现投研端全动态风格监控与风险对齐。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd


class BarraFactorCalculator:
    """动态 Barra 风格特征矩阵自动计算器"""

    BARRA_STYLES = ["SIZE", "VALUE", "MOMENTUM", "VOLATILITY", "LIQUIDITY"]

    def __init__(
        self,
        bars_df: Optional[pd.DataFrame] = None,
        fundamentals_df: Optional[pd.DataFrame] = None,
        lookback_days: int = 20,
    ):
        self.lookback_days = lookback_days
        self._cached_matrices: Dict[str, pd.DataFrame] = {}
        if bars_df is not None and not bars_df.empty:
            self.precompute(bars_df, fundamentals_df)

    @classmethod
    def standardize_cross_section(cls, series: pd.Series) -> pd.Series:
        """截面 3-sigma 去极值与 Z-Score 标准化"""
        s = series.copy().astype(float)
        valid = s.dropna()
        if len(valid) < 2:
            return pd.Series(0.0, index=series.index)

        # 3-sigma 去极值
        mean = valid.mean()
        std = valid.std(ddof=1)
        if std <= 1e-12:
            return pd.Series(0.0, index=series.index)

        clipped = valid.clip(lower=mean - 3.0 * std, upper=mean + 3.0 * std)
        z = (clipped - clipped.mean()) / (clipped.std(ddof=1) + 1e-12)
        return z.reindex(series.index).fillna(0.0)

    def precompute(
        self,
        bars_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        全量预计算历史所有交易日的 Barra 风格特征矩阵并持久化缓存
        """
        pivoted_close = bars_df.pivot(index="trade_date", columns="symbol", values="close").ffill().bfill()
        timeline = sorted(list(pivoted_close.index))
        symbols = sorted(list(pivoted_close.columns))

        # 1. 动量: 20 日收益率
        mom_df = pivoted_close.pct_change(self.lookback_days).fillna(0.0)

        # 2. 波动率: 20 日收益率标准差
        daily_rets = pivoted_close.pct_change().fillna(0.0)
        vol_df = daily_rets.rolling(self.lookback_days, min_periods=5).std().fillna(0.02)

        # 3. 规模 & 流动性: 基于成交额或市值
        if "amount" in bars_df.columns:
            pivoted_amount = bars_df.pivot(index="trade_date", columns="symbol", values="amount").ffill().bfill()
            roll_amt = pivoted_amount.rolling(self.lookback_days, min_periods=1).mean().clip(lower=1000.0)
            size_raw_df = np.log(roll_amt)
            liq_raw_df = np.log(pivoted_amount.clip(lower=1000.0))
        else:
            size_raw_df = pd.DataFrame(0.0, index=timeline, columns=symbols)
            liq_raw_df = pd.DataFrame(0.0, index=timeline, columns=symbols)

        # 4. 价值: 基于 PIT 财务净资产 BPS / Close，无财务时基于长周期价格反转代理
        val_raw_df = pd.DataFrame(0.0, index=timeline, columns=symbols)
        if fundamentals_df is not None and not fundamentals_df.empty and "bps" in fundamentals_df.columns:
            fina = fundamentals_df.copy()
            fina["published_at"] = fina["published_at"].astype(str)
            fina = fina.sort_values(["symbol", "published_at"]).drop_duplicates(["symbol", "published_at"], keep="last")
            pivoted_bps = fina.pivot(index="published_at", columns="symbol", values="bps")
            aligned_bps = pivoted_bps.reindex(timeline).ffill().bfill()
            val_raw_df = (aligned_bps / pivoted_close).fillna(0.0)
        else:
            # 长期估值修复/均值反转代理: -(Close / MA60 - 1)
            ma60 = pivoted_close.rolling(60, min_periods=10).mean()
            val_raw_df = -((pivoted_close - ma60) / ma60).fillna(0.0)

        # 逐日进行横截面标准化
        for dt in timeline:
            row_dict = {}
            for sym in symbols:
                row_dict[sym] = {
                    "SIZE": size_raw_df.loc[dt, sym] if sym in size_raw_df.columns else 0.0,
                    "VALUE": val_raw_df.loc[dt, sym] if sym in val_raw_df.columns else 0.0,
                    "MOMENTUM": mom_df.loc[dt, sym] if sym in mom_df.columns else 0.0,
                    "VOLATILITY": vol_df.loc[dt, sym] if sym in vol_df.columns else 0.0,
                    "LIQUIDITY": liq_raw_df.loc[dt, sym] if sym in liq_raw_df.columns else 0.0,
                }
            df_day = pd.DataFrame.from_dict(row_dict, orient="index")
            # 截面标准化每一列风格
            for col in self.BARRA_STYLES:
                df_day[col] = self.standardize_cross_section(df_day[col])

            self._cached_matrices[str(dt)] = df_day

    def get_style_matrix(
        self,
        trade_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        bars_slice: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        获取指定交易日的截面 Barra 风格矩阵 (自动使用缓存或切片即时推导)
        :param trade_date: 交易日字符串 (YYYY-MM-DD)
        :param symbols: 目标标的列表
        :param bars_slice: 若无预计算缓存，可传入当期 bars 行情切片即时推导
        :return: DataFrame(index=symbols, columns=["SIZE", "VALUE", "MOMENTUM", "VOLATILITY", "LIQUIDITY"])
        """
        dt_key = str(trade_date) if trade_date else ""
        if dt_key in self._cached_matrices:
            mat = self._cached_matrices[dt_key]
            if symbols:
                # 补齐未在矩阵中的标的
                mat = mat.reindex(symbols).fillna(0.0)
            return mat

        # 即时根据切片推导
        sym_list = symbols or (sorted(list(bars_slice["symbol"].unique())) if bars_slice is not None else [])
        if not sym_list:
            return pd.DataFrame(0.0, index=[], columns=self.BARRA_STYLES)

        if bars_slice is not None and not bars_slice.empty:
            slice_data = {}
            for s in sym_list:
                s_bars = bars_slice[bars_slice["symbol"] == s]
                if s_bars.empty:
                    slice_data[s] = {c: 0.0 for c in self.BARRA_STYLES}
                else:
                    amt = float(s_bars["amount"].iloc[-1]) if "amount" in s_bars.columns else 1000000.0
                    close = float(s_bars["close"].iloc[-1])
                    high = float(s_bars["high"].iloc[-1]) if "high" in s_bars.columns else close
                    low = float(s_bars["low"].iloc[-1]) if "low" in s_bars.columns else close
                    open_p = float(s_bars["open"].iloc[-1]) if "open" in s_bars.columns else close
                    ret = (close / open_p) - 1.0 if open_p > 0 else 0.0
                    vol = (high - low) / close if close > 0 else 0.02
                    slice_data[s] = {
                        "SIZE": np.log(max(amt, 1000.0)),
                        "VALUE": 0.0,
                        "MOMENTUM": ret,
                        "VOLATILITY": vol,
                        "LIQUIDITY": np.log(max(amt, 1000.0)),
                    }
            mat = pd.DataFrame.from_dict(slice_data, orient="index")
            for col in self.BARRA_STYLES:
                mat[col] = self.standardize_cross_section(mat[col])
            return mat

        # 兜底全零中性矩阵
        return pd.DataFrame(0.0, index=sym_list, columns=self.BARRA_STYLES)
