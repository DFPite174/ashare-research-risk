"""
企业级量化投研平台 - 组合集中度合规投影器 (Concentration Limits Projector)
实现：
1. 单票最大权重限制 (如单票 <= 40% 或 <= 20%)；
2. 申万单一行业暴露上限 (如单一行业总持仓 <= 30%)；
3. 采用削峰填谷多维迭代投影算法，确保约束严格满足且权重总和受控。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConcentrationReport:
    """集中度合规投影报告"""
    max_single_stock_weight: float
    max_single_industry_weight: float
    capped_stocks: List[str]
    capped_industries: List[str]
    is_compliant: bool


class ConcentrationProjector:
    """
    单票与行业集中度约束投影器
    """

    def __init__(
        self,
        max_stock_weight: float = 0.40,
        max_industry_weight: float = 0.30,
        max_iterations: int = 30,
    ):
        self.max_stock_weight = max_stock_weight
        self.max_industry_weight = max_industry_weight
        self.max_iterations = max_iterations

    def project_weights(
        self,
        target_weights: pd.Series,
        industry_mapping: Optional[Dict[str, str]] = None,
        total_target_sum: float = 1.0,
    ) -> Tuple[pd.Series, ConcentrationReport]:
        """
        对目标权重执行单票与行业集中度合规投影
        :param target_weights: 原始目标权重 Series(symbol -> weight)
        :param industry_mapping: 行业分类字典 (symbol -> industry_name, 如 "600519.SH" -> "食品饮料")
        :param total_target_sum: 目标总仓位 (默认 1.0)
        :return: (投影后的合规权重, 集中度分析报告)
        """
        w = target_weights.copy().astype(float)
        # 排除负权重
        w = w.clip(lower=0.0)
        if w.sum() <= 1e-12:
            return w, ConcentrationReport(
                max_single_stock_weight=0.0,
                max_single_industry_weight=0.0,
                capped_stocks=[],
                capped_industries=[],
                is_compliant=True,
            )

        # 归一化至 total_target_sum
        w = w * (total_target_sum / w.sum())

        capped_stocks_set = set()
        capped_industries_set = set()

        for _ in range(self.max_iterations):
            changed = False

            # 1. 单票上限约束
            stock_violations = w[w > self.max_stock_weight + 1e-6]
            if len(stock_violations) > 0:
                changed = True
                excess = float((w - self.max_stock_weight).clip(lower=0.0).sum())
                capped_stocks_set.update(stock_violations.index)
                w = w.clip(upper=self.max_stock_weight)
                # 将多余仓位平摊到非上限标的
                uncapped_mask = w < self.max_stock_weight - 1e-6
                if uncapped_mask.any() and w[uncapped_mask].sum() > 1e-12:
                    w[uncapped_mask] += excess * (w[uncapped_mask] / w[uncapped_mask].sum())

            # 2. 行业暴露上限约束
            if industry_mapping is not None:
                ind_series = pd.Series(industry_mapping)
                common_syms = w.index.intersection(ind_series.index)
                if len(common_syms) > 0:
                    df = pd.DataFrame({"weight": w.loc[common_syms], "industry": ind_series.loc[common_syms]})
                    ind_sums = df.groupby("industry")["weight"].sum()
                    over_industries = ind_sums[ind_sums > self.max_industry_weight + 1e-6]

                    if len(over_industries) > 0:
                        changed = True
                        for ind, ind_weight in over_industries.items():
                            capped_industries_set.add(ind)
                            syms_in_ind = df[df["industry"] == ind].index
                            scale_factor = self.max_industry_weight / ind_weight
                            w.loc[syms_in_ind] *= scale_factor

            if not changed:
                break

        # 最终严格保底约束：确保绝无单票或行业上溢
        w = w.clip(lower=0.0, upper=self.max_stock_weight)
        if industry_mapping is not None:
            ind_series = pd.Series(industry_mapping).reindex(w.index)
            for ind, group in w.groupby(ind_series):
                ind_sum = group.sum()
                if ind_sum > self.max_industry_weight + 1e-5:
                    w.loc[group.index] *= (self.max_industry_weight / ind_sum)

        max_stock_val = float(w.max()) if len(w) > 0 else 0.0
        max_ind_val = 0.0
        if industry_mapping is not None:
            df = pd.DataFrame({"weight": w, "industry": pd.Series(industry_mapping).reindex(w.index).fillna("UNKNOWN")})
            max_ind_val = float(df.groupby("industry")["weight"].sum().max())

        report = ConcentrationReport(
            max_single_stock_weight=round(max_stock_val, 4),
            max_single_industry_weight=round(max_ind_val, 4),
            capped_stocks=sorted(list(capped_stocks_set)),
            capped_industries=sorted(list(capped_industries_set)),
            is_compliant=(max_stock_val <= self.max_stock_weight + 1e-4),
        )

        return w, report
