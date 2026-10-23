"""
企业级量化投研平台 - 组合净值阶梯熔断与回撤保护器 (Drawdown Circuit Breaker)
实现：
1. 动态维护组合净值最高水位线 (High Water Mark, HWM)；
2. 阶梯式回撤熔断降仓机制：
   - 回撤 >= 8%: 仓位上限降至 50%;
   - 回撤 >= 12%: 仓位上限降至 20%;
   - 回撤 >= 15%: 触发强制熔断清仓 (仓位降至 0%，全部避险持币).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class CircuitBreakerStatus(str, Enum):
    NORMAL = "NORMAL"                 # 正常运行 (回撤 < 8%)
    WARNING_DERISK = "WARNING_DERISK" # 预警降仓 (8% <= 回撤 < 12%)
    CRITICAL_DEFENSE = "CRITICAL_DEFENSE" # 深度防御 (12% <= 回撤 < 15%)
    MELTDOWN_FROZEN = "MELTDOWN_FROZEN"   # 彻底熔断清仓 (回撤 >= 15%)


@dataclass(frozen=True)
class CircuitBreakerReport:
    """熔断器状态快照"""
    current_nav: float
    high_water_mark: float
    current_drawdown: float
    status: CircuitBreakerStatus
    allowed_max_exposure: float
    is_triggered: bool


class DrawdownCircuitBreaker:
    """
    多级净值回撤熔断器
    """

    def __init__(
        self,
        tier1_drawdown: float = 0.08,    # 阶段一回撤阈值 (8%)
        tier1_exposure: float = 0.50,    # 阶段一允许最大仓位 (50%)
        tier2_drawdown: float = 0.12,    # 阶段二回撤阈值 (12%)
        tier2_exposure: float = 0.20,    # 阶段二允许最大仓位 (20%)
        tier3_meltdown: float = 0.15,    # 阶段三清盘熔断阈值 (15%)
    ):
        self.tier1_drawdown = tier1_drawdown
        self.tier1_exposure = tier1_exposure
        self.tier2_drawdown = tier2_drawdown
        self.tier2_exposure = tier2_exposure
        self.tier3_meltdown = tier3_meltdown

    def evaluate_nav(
        self,
        nav_series: pd.Series,
    ) -> CircuitBreakerReport:
        """
        根据历史净值曲线计算当前回撤等级与允许持仓暴露
        """
        clean_nav = nav_series.dropna()
        if len(clean_nav) == 0:
            return CircuitBreakerReport(
                current_nav=1.0,
                high_water_mark=1.0,
                current_drawdown=0.0,
                status=CircuitBreakerStatus.NORMAL,
                allowed_max_exposure=1.0,
                is_triggered=False,
            )

        hwm = float(clean_nav.cummax().iloc[-1])
        curr_nav = float(clean_nav.iloc[-1])
        dd = max((hwm - curr_nav) / (hwm + 1e-12), 0.0)

        if dd >= self.tier3_meltdown:
            status = CircuitBreakerStatus.MELTDOWN_FROZEN
            allowed = 0.0
            triggered = True
        elif dd >= self.tier2_drawdown:
            status = CircuitBreakerStatus.CRITICAL_DEFENSE
            allowed = self.tier2_exposure
            triggered = True
        elif dd >= self.tier1_drawdown:
            status = CircuitBreakerStatus.WARNING_DERISK
            allowed = self.tier1_exposure
            triggered = True
        else:
            status = CircuitBreakerStatus.NORMAL
            allowed = 1.0
            triggered = False

        return CircuitBreakerReport(
            current_nav=round(curr_nav, 4),
            high_water_mark=round(hwm, 4),
            current_drawdown=round(dd, 4),
            status=status,
            allowed_max_exposure=allowed,
            is_triggered=triggered,
        )

    def apply_circuit_breaker(
        self,
        target_weights: pd.Series,
        nav_series: pd.Series,
    ) -> Tuple[pd.Series, CircuitBreakerReport]:
        """
        对目标权重实施熔断保护
        """
        report = self.evaluate_nav(nav_series)
        w = target_weights.copy()
        current_sum = float(w.sum())

        if current_sum > report.allowed_max_exposure:
            scale = report.allowed_max_exposure / (current_sum + 1e-12) if report.allowed_max_exposure > 0 else 0.0
            w = w * scale

        return w, report
