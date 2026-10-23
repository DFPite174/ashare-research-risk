"""
事后风险剖析：Barra 风格因子风险归因与历史宏观情景压力测试单元测试
"""

import pytest
import numpy as np
import pandas as pd

from risk.attribution import (
    BarraRiskAttribution,
    HistoricalStressTester,
)


def test_barra_style_exposure_attribution():
    attribution = BarraRiskAttribution()

    # 4 只股票，特征：
    # 000001: 大市值、低估值、低动量
    # 600519: 超大市值、高估值、高动量
    # SMALL1: 极小市值、高波动
    # SMALL2: 极小市值、高流动性
    style_df = pd.DataFrame({
        "SIZE": [1.2, 2.5, -2.0, -2.1],
        "VALUE": [1.5, -0.8, -0.2, 0.1],
        "MOMENTUM": [-0.5, 1.8, 0.4, -0.2],
        "VOLATILITY": [-0.8, -0.5, 1.9, 1.5],
        "LIQUIDITY": [0.2, 0.4, 1.6, 1.2],
    }, index=["000001.SZ", "600519.SH", "SMALL1", "SMALL2"])

    # 策略重仓持有 SMALL1 和 SMALL2 (微盘策略)
    micro_weights = pd.Series({"SMALL1": 0.50, "SMALL2": 0.50})
    rep_micro = attribution.compute_portfolio_exposure(micro_weights, style_df)

    # 规模因子 (SIZE) 必然为极端负暴露 (严重暴露微盘风险)
    assert rep_micro.style_exposures["SIZE"] < -0.80
    assert len(rep_micro.unintended_biases) > 0

    # 均衡权重组合
    balanced_weights = pd.Series({"000001.SZ": 0.25, "600519.SH": 0.25, "SMALL1": 0.25, "SMALL2": 0.25})
    rep_bal = attribution.compute_portfolio_exposure(balanced_weights, style_df)
    assert abs(rep_bal.style_exposures["SIZE"]) < abs(rep_micro.style_exposures["SIZE"])


def test_historical_stress_testing():
    tester = HistoricalStressTester(survival_drawdown_limit=0.30)

    weights = pd.Series({"000001.SZ": 0.50, "600519.SH": 0.50})
    betas = pd.Series({"000001.SZ": 0.8, "600519.SH": 1.1})

    rep = tester.run_stress_test(weights, asset_betas=betas)

    assert rep.worst_case_drawdown > 0.0
    assert len(rep.scenarios) == 3
    # 检查 2015 股灾情景
    sc15 = [s for s in rep.scenarios if "2015" in s.scenario_name][0]
    assert sc15.estimated_portfolio_drawdown < -0.15
    assert sc15.is_survival_passed is True
    assert rep.overall_stress_passed is True
