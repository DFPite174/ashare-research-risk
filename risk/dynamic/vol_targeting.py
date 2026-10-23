"""
企业级量化投研平台 - 动态波动率目标调仓机制 (Volatility Targeting)
通过追踪组合滚动实现波动率，在市场高波震荡或流动性枯竭时自动等比例降仓，实现风险预算平滑。
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd


class VolatilityTargetingEngine:
    """
    波动率目标风控调节器
    """

    def __init__(
        self,
        target_annual_vol: float = 0.15,      # 目标年化波动率 (默认 15%)
        lookback_window: int = 20,            # 波动率回看窗口 (20 交易日)
        max_leverage: float = 1.0,            # 投研端最大杠杆倍数 (纯多头上限 1.0)
        periods_per_year: int = 252,
    ):
        self.target_annual_vol = target_annual_vol
        self.lookback_window = lookback_window
        self.max_leverage = max_leverage
        self.periods_per_year = periods_per_year

    def compute_scaling_factor(
        self,
        historical_returns: pd.Series,
    ) -> Tuple[float, float]:
        """
        根据历史收益序列计算当前应采取的风险资产缩放倍率
        :param historical_returns: 组合历史日频收益序列
        :return: (仓位缩放系数 scalar in [0, max_leverage], 当前年化实现波动率)
        """
        clean_rets = historical_returns.dropna()
        if len(clean_rets) < self.lookback_window:
            return 1.0, self.target_annual_vol

        recent = clean_rets.iloc[-self.lookback_window:]
        daily_std = float(recent.std(ddof=1))
        ann_vol = daily_std * np.sqrt(self.periods_per_year)

        if ann_vol <= 1e-6:
            scalar = self.max_leverage
        else:
            scalar = self.target_annual_vol / ann_vol
            scalar = float(np.clip(scalar, 0.05, self.max_leverage))

        return scalar, float(ann_vol)

    def apply_vol_targeting(
        self,
        target_weights: pd.Series,
        historical_returns: pd.Series,
    ) -> Tuple[pd.Series, float, float]:
        """
        对拟分配权重执行波动率目标平滑
        :return: (调整后的目标权重, 缩放系数, 当前实现波动率)
        """
        scalar, ann_vol = self.compute_scaling_factor(historical_returns)
        scaled_weights = target_weights * scalar
        return scaled_weights, scalar, ann_vol
