"""
事中动态风控：波动率目标化 (Vol Targeting) 与净值阶梯熔断单元测试
"""

import pytest
import numpy as np
import pandas as pd

from risk.dynamic import (
    VolatilityTargetingEngine,
    DrawdownCircuitBreaker,
    CircuitBreakerStatus,
)


def test_volatility_targeting_scaling():
    # 目标年化波动率 15%
    engine = VolatilityTargetingEngine(target_annual_vol=0.15, lookback_window=20)

    # 1. 低波动环境下 (年化波幅约 10%)
    np.random.seed(42)
    low_vol_rets = pd.Series(np.random.normal(0.0005, 0.0063, 30))  # 0.0063 * sqrt(252) ≈ 10%
    scalar_low, ann_vol_low = engine.compute_scaling_factor(low_vol_rets)
    # 不加杠杆情况下上限为 1.0
    assert scalar_low == 1.0
    assert ann_vol_low < 0.15

    # 2. 高波动极端环境下 (年化波幅约 30%)
    high_vol_rets = pd.Series(np.random.normal(0.0, 0.022, 30))  # 0.022 * sqrt(252) ≈ 35%
    scalar_high, ann_vol_high = engine.compute_scaling_factor(high_vol_rets)
    # 应等比例自动降仓至约 40%~65%
    assert 0.35 <= scalar_high <= 0.70
    assert ann_vol_high > 0.20


def test_drawdown_circuit_breaker_tiers():
    cb = DrawdownCircuitBreaker(
        tier1_drawdown=0.08, tier1_exposure=0.50,
        tier2_drawdown=0.12, tier2_exposure=0.20,
        tier3_meltdown=0.15,
    )

    # 1. 正常运行期 (回撤 3%)
    nav_normal = pd.Series([1.0, 1.05, 1.08, 1.05])  # HWM 1.08, curr 1.05 (DD ≈ 2.8%)
    rep1 = cb.evaluate_nav(nav_normal)
    assert rep1.status == CircuitBreakerStatus.NORMAL
    assert rep1.allowed_max_exposure == 1.0
    assert rep1.is_triggered is False

    # 2. 阶段一降仓 (回撤 9%)
    nav_t1 = pd.Series([1.0, 1.10, 1.00])  # HWM 1.10, curr 1.00 (DD = 9.1%)
    rep2 = cb.evaluate_nav(nav_t1)
    assert rep2.status == CircuitBreakerStatus.WARNING_DERISK
    assert rep2.allowed_max_exposure == 0.50
    assert rep2.is_triggered is True

    # 3. 阶段二深度防御 (回撤 13%)
    nav_t2 = pd.Series([1.0, 1.10, 0.95])  # HWM 1.10, curr 0.95 (DD = 13.6%)
    rep3 = cb.evaluate_nav(nav_t2)
    assert rep3.status == CircuitBreakerStatus.CRITICAL_DEFENSE
    assert rep3.allowed_max_exposure == 0.20

    # 4. 阶段三清盘熔断 (回撤 16%)
    nav_t3 = pd.Series([1.0, 1.10, 0.92])  # HWM 1.10, curr 0.92 (DD = 16.4%)
    rep4 = cb.evaluate_nav(nav_t3)
    assert rep4.status == CircuitBreakerStatus.MELTDOWN_FROZEN
    assert rep4.allowed_max_exposure == 0.0
