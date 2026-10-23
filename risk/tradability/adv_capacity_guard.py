"""
企业级量化投研平台 - 流动性容量与 ADV 冲击风控器 (ADV Capacity Guard)
吸纳交易端核心流动性约束：
1. 调仓量受标的过去 20 日日均成交额 (ADV_20) 严格约束 (默认单日调仓额 <= 10% ADV)；
2. 基于 Almgren-Chriss 经典平方根冲击法则测算执行滑点与市场冲击损耗；
3. 超限自动截断目标权重，避免回测出现“几十亿买入微盘股”的纸面富贵。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CapacityConstraintReport:
    """流动性容量截断报告"""
    clipped_symbols: Dict[str, float]       # 受到截断的股票及对应削减金额 (CNY)
    market_impact_bps: Dict[str, float]     # 各标的根据平方根模型估算的市场冲击 (bps)
    total_rebalance_amount_cny: float       # 计划调仓总额
    allowed_rebalance_amount_cny: float     # 约束后允许调仓总额
    is_capacity_breached: bool              # 是否发生流动性容量击穿


class ADVCapacityGuard:
    """
    ADV 流动性容量与冲击成本评估器
    """

    def __init__(
        self,
        max_adv_participation: float = 0.10,  # 单日最大成交量占比 (通常 5% ~ 15%)
        impact_coefficient: float = 0.10,     # Almgren-Chriss 平方根冲击系数
    ):
        self.max_adv_participation = max_adv_participation
        self.impact_coefficient = impact_coefficient

    def estimate_market_impact_bps(self, trade_amount_cny: float, adv_cny: float) -> float:
        """
        基于经典平方根法则 (Square-Root Law) 测算市场冲击成本 (bps)
        Impact = eta * sqrt(Trade_Volume / ADV) * 10,000
        """
        if adv_cny <= 1e-6 or trade_amount_cny <= 1e-6:
            return 0.0
        participation_ratio = min(trade_amount_cny / adv_cny, 1.0)
        impact_bps = self.impact_coefficient * np.sqrt(participation_ratio) * 10000.0
        return float(impact_bps)

    def constrain_target_weights(
        self,
        portfolio_equity_cny: float,
        current_weights: pd.Series,
        target_weights: pd.Series,
        adv_20_cny: pd.Series,
    ) -> Tuple[pd.Series, CapacityConstraintReport]:
        """
        根据 ADV 限制约束调仓权重并返回估算冲击
        :param portfolio_equity_cny: 当前账户净资产总额 (如 1000 万)
        :param current_weights: 当前实际持仓权重
        :param target_weights: 目标持仓权重
        :param adv_20_cny: 各标的过去 20 日日均成交额 (以元为单位)
        :return: (约束后的目标权重, 容量分析报告)
        """
        all_symbols = sorted(list(set(current_weights.index).union(set(target_weights.index))))
        curr_aligned = current_weights.reindex(all_symbols).fillna(0.0)
        tgt_aligned = target_weights.reindex(all_symbols).fillna(0.0)
        adv_aligned = adv_20_cny.reindex(all_symbols).fillna(0.0)

        adjusted_weights = tgt_aligned.copy()
        clipped_symbols: Dict[str, float] = {}
        impacts: Dict[str, float] = {}

        total_req_amount = 0.0
        total_allowed_amount = 0.0

        for sym in all_symbols:
            w_curr = float(curr_aligned.loc[sym])
            w_tgt = float(tgt_aligned.loc[sym])
            delta_w = w_tgt - w_curr
            trade_cny = abs(delta_w) * portfolio_equity_cny
            total_req_amount += trade_cny

            adv = float(adv_aligned.loc[sym])
            max_allowed_cny = adv * self.max_adv_participation

            if adv > 0 and trade_cny > max_allowed_cny:
                # 触发流动性容量约束：截断调仓量
                clipped_amount = trade_cny - max_allowed_cny
                clipped_symbols[sym] = float(clipped_amount)

                # 重新计算截断后的目标权重
                allowed_delta_w = np.sign(delta_w) * (max_allowed_cny / portfolio_equity_cny)
                adjusted_weights.loc[sym] = float(w_curr + allowed_delta_w)
                actual_trade_cny = max_allowed_cny
            else:
                actual_trade_cny = trade_cny

            total_allowed_amount += actual_trade_cny
            impacts[sym] = self.estimate_market_impact_bps(actual_trade_cny, adv)

        report = CapacityConstraintReport(
            clipped_symbols=clipped_symbols,
            market_impact_bps=impacts,
            total_rebalance_amount_cny=float(total_req_amount),
            allowed_rebalance_amount_cny=float(total_allowed_amount),
            is_capacity_breached=len(clipped_symbols) > 0,
        )

        return adjusted_weights, report
