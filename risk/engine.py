"""
企业级量化投研平台 - 全流程风控执行中枢 (Research Risk Engine)
串接：
1. 可交易性清洗 (涨跌停/停牌/ST黑名单)
2. 流动性容量约束 (ADV 10% 限制与 Almgren-Chriss 冲击成本)
3. 集中度合规投影 (单票 <= 40%, 行业 <= 30%)
4. 现金缓冲垫预留 (法定手续费储备)
5. 事中动态风控 (波动率目标调节 + 净值阶梯熔断)
6. 事后 Barra 风格暴露与极端历史情景压力测试
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from risk.tradability import (
    TradabilityPriceFilter,
    TradabilityConstraintReport,
    BlacklistFilter,
    ADVCapacityGuard,
    CapacityConstraintReport,
)
from risk.allocation import (
    ConcentrationProjector,
    ConcentrationReport,
    CashBufferManager,
)
from risk.dynamic import (
    VolatilityTargetingEngine,
    DrawdownCircuitBreaker,
    CircuitBreakerReport,
)
from risk.attribution import (
    BarraRiskAttribution,
    BarraStyleExposureReport,
    HistoricalStressTester,
    StressTestReport,
)


@dataclass(frozen=True)
class RiskControlledWeightsResult:
    """风控处理后最终可执行目标权重与诊断总览"""
    raw_weights: pd.Series
    final_weights: pd.Series
    cash_weight: float
    tradability_report: Optional[TradabilityConstraintReport] = None
    capacity_report: Optional[CapacityConstraintReport] = None
    concentration_report: Optional[ConcentrationReport] = None
    circuit_breaker_report: Optional[CircuitBreakerReport] = None
    barra_report: Optional[BarraStyleExposureReport] = None
    stress_test_report: Optional[StressTestReport] = None
    is_fully_compliant: bool = True
    diagnostic_summary: str = ""


class RiskEngine:
    """
    投研端全流程风控总引擎
    """

    def __init__(
        self,
        max_stock_weight: float = 0.40,
        max_industry_weight: float = 0.30,
        max_adv_participation: float = 0.10,
        min_cash_buffer_ratio: float = 0.03,
        target_annual_vol: float = 0.15,
    ):
        self.tradability_filter = TradabilityPriceFilter()
        self.blacklist_filter = BlacklistFilter()
        self.capacity_guard = ADVCapacityGuard(max_adv_participation=max_adv_participation)
        self.concentration_projector = ConcentrationProjector(
            max_stock_weight=max_stock_weight,
            max_industry_weight=max_industry_weight,
        )
        self.cash_manager = CashBufferManager(min_cash_buffer_ratio=min_cash_buffer_ratio)
        self.vol_engine = VolatilityTargetingEngine(target_annual_vol=target_annual_vol)
        self.circuit_breaker = DrawdownCircuitBreaker()
        self.barra_attribution = BarraRiskAttribution()
        self.stress_tester = HistoricalStressTester()

    def process_target_weights(
        self,
        raw_target_weights: pd.Series,
        current_weights: pd.Series,
        portfolio_equity_cny: float = 10000000.0,
        bars_slice: Optional[pd.DataFrame] = None,
        adv_20_cny: Optional[pd.Series] = None,
        industry_mapping: Optional[Dict[str, str]] = None,
        blacklist_symbols: Optional[Set[str]] = None,
        portfolio_nav_history: Optional[pd.Series] = None,
        portfolio_returns_history: Optional[pd.Series] = None,
        asset_style_matrix: Optional[pd.DataFrame] = None,
    ) -> RiskControlledWeightsResult:
        """
        贯穿执行投研端六道风控防线
        """
        w = raw_target_weights.copy().fillna(0.0)
        curr_w = current_weights.reindex(w.index).fillna(0.0)

        # 1. 黑名单过滤
        w, removed_bl = self.blacklist_filter.apply_filter(w, dynamic_blacklist=blacklist_symbols)

        # 2. 涨跌停与停牌过滤
        tradability_rep = None
        if bars_slice is not None:
            w, tradability_rep = self.tradability_filter.apply_filter(curr_w, w, bars_slice)

        # 3. ADV 流动性容量约束与冲击削减
        capacity_rep = None
        if adv_20_cny is not None:
            w, capacity_rep = self.capacity_guard.constrain_target_weights(
                portfolio_equity_cny=portfolio_equity_cny,
                current_weights=curr_w,
                target_weights=w,
                adv_20_cny=adv_20_cny,
            )

        # 4. 事中动态风控: 波动率目标调节
        if portfolio_returns_history is not None:
            w, vol_scalar, ann_vol = self.vol_engine.apply_vol_targeting(w, portfolio_returns_history)

        # 5. 事中动态风控: 净值阶梯回撤熔断
        circuit_rep = None
        if portfolio_nav_history is not None:
            w, circuit_rep = self.circuit_breaker.apply_circuit_breaker(w, portfolio_nav_history)

        # 6. 单票与行业集中度合规投影
        w, conc_rep = self.concentration_projector.project_weights(
            target_weights=w,
            industry_mapping=industry_mapping,
            total_target_sum=min(float(w.sum()), 1.0),
        )

        # 7. 预留法定现金缓冲垫
        w, cash_ratio = self.cash_manager.apply_cash_buffer(w)

        # 8. 事后诊断: Barra 风格暴露度分析
        barra_rep = None
        if asset_style_matrix is not None:
            barra_rep = self.barra_attribution.compute_portfolio_exposure(w, asset_style_matrix)

        # 9. 事后诊断: 宏观黑天鹅极端情景压力测试
        style_exp_dict = barra_rep.style_exposures if barra_rep else None
        stress_rep = self.stress_tester.run_stress_test(w, style_exposures=style_exp_dict)

        # 构造诊断文本摘要
        diag_lines = [
            "### 投研端全流程风控审计报告",
            f"- 原始拟分配仓位: `{raw_target_weights.sum():.2%}` ➔ 风控后可执行多头仓位: `{w.sum():.2%}`, 现金缓冲: `{cash_ratio:.2%}`",
            f"- 集中度合规: 单票最高 `{conc_rep.max_single_stock_weight:.1%}` (上限 <= {self.concentration_projector.max_stock_weight:.0%}), 行业最高 `{conc_rep.max_single_industry_weight:.1%}`",
        ]
        if capacity_rep and capacity_rep.is_capacity_breached:
            diag_lines.append(f"- ⚠️ 触发 ADV 容量削减: {len(capacity_rep.clipped_symbols)} 只股票因流动性受限被截断")
        if tradability_rep and tradability_rep.blocked_symbols:
            diag_lines.append(f"- ⚠️ 触发可交易性封锁: {len(tradability_rep.blocked_symbols)} 只股票因涨跌停/停牌锁定持仓")
        if circuit_rep and circuit_rep.is_triggered:
            diag_lines.append(f"- 🚨 触发净值回撤熔断防御: 当前回撤 `{circuit_rep.current_drawdown:.1%}`, 仓位强制限额 `{circuit_rep.allowed_max_exposure:.1%}`")
        if stress_rep:
            diag_lines.append(f"- 极端黑天鹅情景最大预估回撤: `{stress_rep.worst_case_drawdown:.1%}` (压力测试合规: `{'通过' if stress_rep.overall_stress_passed else '超标'}`)")

        return RiskControlledWeightsResult(
            raw_weights=raw_target_weights,
            final_weights=w,
            cash_weight=cash_ratio,
            tradability_report=tradability_rep,
            capacity_report=capacity_rep,
            concentration_report=conc_rep,
            circuit_breaker_report=circuit_rep,
            barra_report=barra_rep,
            stress_test_report=stress_rep,
            is_fully_compliant=conc_rep.is_compliant and (not stress_rep or stress_rep.overall_stress_passed),
            diagnostic_summary="\n".join(diag_lines),
        )
