"""
企业级量化投研平台 - 历史极端情景压力测试器 (Historical Scenario Stress Testing)
针对构建出的当前组合权重，回放 A 股历史上最典型的黑天鹅与流动性危机：
1. 2015 股灾踩踏情景 (2015 China Stock Crash): 全市场连续跌停、流动性瞬间丧失；
2. 2020 疫情流动性危机 (2020 Global Covid Shock): 极端波动与恐慌抛售；
3. 2024 年初微盘踩踏事件 (2024 Micro-cap Meltdown): 大小盘分化、DMA/雪球集中平仓带来的微盘流动性冻结。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressScenarioOutcome:
    """单个压力测试情景下的测算表现"""
    scenario_name: str
    scenario_description: str
    estimated_portfolio_drawdown: float    # 预估组合在情景下的回撤幅度
    var_99_daily: float                    # 99% 置信度单日在险价值
    worst_hit_symbol: str                  # 组合内受创最严重标的
    worst_hit_symbol_drop: float           # 该标的单日/期间跌幅
    is_survival_passed: bool               # 组合是否未击穿清盘线 (默认最大回撤 < 25%)


@dataclass(frozen=True)
class StressTestReport:
    """全景压力测试报告"""
    worst_case_drawdown: float
    scenarios: List[StressScenarioOutcome]
    overall_stress_passed: bool


class HistoricalStressTester:
    """
    宏观历史压力测试器
    """

    # 内置四大经典冲击情景参数 (以行业/风格跌幅代理)
    PRESET_SCENARIOS = {
        "2015_MARKET_CRASH": {
            "name": "2015 年 A 股股灾流动性踩踏",
            "desc": "千股跌停、停牌避险潮与杠杆资金强平，高 Beta/高估值成长股单周跌幅超 25%",
            "broad_market_shock": -0.22,
            "style_multiplier": {"SIZE": -0.08, "MOMENTUM": -0.06, "VOLATILITY": -0.05, "VALUE": 0.04},
        },
        "2020_COVID_SHOCK": {
            "name": "2020 年初全球疫情暴发恐慌",
            "desc": "春节后开盘单日近 3000 只股票跌停，随后极端 V 型反弹与高波动",
            "broad_market_shock": -0.08,
            "style_multiplier": {"SIZE": -0.02, "MOMENTUM": 0.03, "VOLATILITY": -0.04, "VALUE": -0.02},
        },
        "2024_MICROCAP_MELTDOWN": {
            "name": "2024 年初微盘股流动性危机与踩踏",
            "desc": "雪球敲入与量化 DMA 降杠杆导致中证 2000 / 微盘股两周暴跌超 30%，沪深 300 逆势上扬",
            "broad_market_shock": -0.05,
            "style_multiplier": {"SIZE": -0.25, "MOMENTUM": -0.15, "VOLATILITY": -0.10, "VALUE": 0.08},
        },
    }

    def __init__(self, survival_drawdown_limit: float = 0.25):
        self.survival_drawdown_limit = survival_drawdown_limit

    def run_stress_test(
        self,
        target_weights: pd.Series,
        asset_betas: Optional[pd.Series] = None,
        style_exposures: Optional[Dict[str, float]] = None,
    ) -> StressTestReport:
        """
        在当前目标权重上模拟极端情景冲击
        """
        w = target_weights.dropna()
        if w.sum() > 0:
            w = w / w.sum()

        betas = asset_betas.reindex(w.index).fillna(1.0) if asset_betas is not None else pd.Series(1.0, index=w.index)
        style_exp = style_exposures or {"SIZE": 0.0, "VALUE": 0.0, "MOMENTUM": 0.0, "VOLATILITY": 0.0}

        outcomes: List[StressScenarioOutcome] = []
        worst_dd = 0.0

        for sc_key, sc_info in self.PRESET_SCENARIOS.items():
            base_shock = sc_info["broad_market_shock"]
            # 各标的跌幅 = beta * base_shock
            asset_drops = betas * base_shock

            # 风格额外叠加冲击
            style_penalty = sum(style_exp.get(st, 0.0) * sc_info["style_multiplier"].get(st, 0.0) for st in style_exp)
            port_drawdown = float(np.sum(w * asset_drops) + style_penalty)
            port_drawdown = min(port_drawdown, 0.0)

            # 最伤标的
            worst_sym = str(asset_drops.idxmin()) if len(asset_drops) > 0 else "UNKNOWN"
            worst_drop = float(asset_drops.min()) if len(asset_drops) > 0 else 0.0

            # 估算 VaR 99%
            var_99 = abs(port_drawdown) * 0.45

            passed = abs(port_drawdown) <= self.survival_drawdown_limit
            if abs(port_drawdown) > worst_dd:
                worst_dd = abs(port_drawdown)

            outcomes.append(StressScenarioOutcome(
                scenario_name=sc_info["name"],
                scenario_description=sc_info["desc"],
                estimated_portfolio_drawdown=round(port_drawdown, 4),
                var_99_daily=round(var_99, 4),
                worst_hit_symbol=worst_sym,
                worst_hit_symbol_drop=round(worst_drop, 4),
                is_survival_passed=passed,
            ))

        overall_passed = all(o.is_survival_passed for o in outcomes)

        return StressTestReport(
            worst_case_drawdown=round(worst_dd, 4),
            scenarios=outcomes,
            overall_stress_passed=overall_passed,
        )
