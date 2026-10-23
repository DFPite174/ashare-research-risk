"""
企业级量化投研平台 - Barra 风格因子风险暴露与归因分析器 (Barra Style Risk Attribution)
依据 Barra CNE5/CNE6 架构，对投资组合暴露的风格因子进行标准化诊断：
1. 规模因子 (Size): 对数总市值；
2. 价值因子 (Value): 估值倒数 BP / EP；
3. 动量因子 (Momentum): 中期时序动量；
4. 波动率因子 (Volatility): 历史收益率残差波动率；
5. 流动性因子 (Liquidity): 换手率。
杜绝策略无意裸露极端微盘风险或单边风格押注。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarraStyleExposureReport:
    """组合 Barra 风格暴露度报告"""
    style_exposures: Dict[str, float]       # 组合在各风格因子上的标准化 Z-score 暴露
    dominant_style: str                     # 最大正向暴露风格
    unintended_biases: List[str]            # 异常超标风险暴露 (|Z| >= 0.70)
    is_style_neutral: bool                  # 是否达到相对中性状态 (|Z| <= 0.30)


class BarraRiskAttribution:
    """
    Barra 风格因子风险暴露度测算器
    """

    SUPPORTED_STYLES = ["SIZE", "VALUE", "MOMENTUM", "VOLATILITY", "LIQUIDITY"]

    @classmethod
    def standardize_cross_section(cls, factor_series: pd.Series) -> pd.Series:
        """截面去极值与 Z-Score 标准化"""
        s = factor_series.dropna()
        if len(s) < 2:
            return factor_series
        # 3-sigma 去极值
        mean = s.mean()
        std = s.std(ddof=1)
        if std <= 1e-12:
            return pd.Series(0.0, index=factor_series.index)
        clipped = s.clip(lower=mean - 3.0 * std, upper=mean + 3.0 * std)
        # Z-Score 标准化为 0 均值 1 方差
        z = (clipped - clipped.mean()) / (clipped.std(ddof=1) + 1e-12)
        return z.reindex(factor_series.index).fillna(0.0)

    def compute_portfolio_exposure(
        self,
        weights: pd.Series,
        asset_style_matrix: pd.DataFrame,
    ) -> BarraStyleExposureReport:
        """
        计算投资组合在各 Barra 风格上的暴露
        :param weights: 投资组合持仓权重 Series(symbol -> weight)
        :param asset_style_matrix: 资产风格因子特征矩阵 DataFrame(index=symbol, columns=style_names)
        :return: BarraStyleExposureReport
        """
        common_symbols = weights.index.intersection(asset_style_matrix.index)
        if len(common_symbols) == 0:
            return BarraStyleExposureReport(
                style_exposures={s: 0.0 for s in self.SUPPORTED_STYLES},
                dominant_style="NONE",
                unintended_biases=[],
                is_style_neutral=True,
            )

        w = weights.loc[common_symbols].copy()
        if w.sum() > 0:
            w = w / w.sum()

        exposures: Dict[str, float] = {}
        unintended: List[str] = []

        for col in asset_style_matrix.columns:
            # 截面标准化
            std_factor = self.standardize_cross_section(asset_style_matrix[col])
            # 组合暴露 = w^T * X
            port_exp = float(np.sum(w * std_factor.loc[common_symbols]))
            exposures[col] = round(port_exp, 3)

            if abs(port_exp) >= 0.70:
                direction = "超配 (Overweight)" if port_exp > 0 else "低配 (Underweight)"
                unintended.append(f"{col} 异常暴露: {port_exp:+.2f} ({direction})")

        # 确定主导风格
        if exposures:
            dom = max(exposures.keys(), key=lambda k: abs(exposures[k]))
        else:
            dom = "NONE"

        is_neutral = all(abs(v) <= 0.35 for v in exposures.values())

        return BarraStyleExposureReport(
            style_exposures=exposures,
            dominant_style=dom,
            unintended_biases=unintended,
            is_style_neutral=is_neutral,
        )
