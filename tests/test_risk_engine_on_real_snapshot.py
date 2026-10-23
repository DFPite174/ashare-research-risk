"""
全流程风控总引擎 (RiskEngine) 端到端集成测试 (Self-contained Risk Engine Tests)
"""

import pytest
import numpy as np
import pandas as pd

from risk.engine import RiskEngine


def test_risk_engine_end_to_end_pipeline():
    symbols = ["000001.SZ", "600519.SH", "601088.SH", "688981.SH"]
    
    # 模拟切片行情与 20 日均额 ADV
    today_bars = pd.DataFrame([
        {"symbol": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 1000000, "pre_close": 10.0},
        {"symbol": "600519.SH", "open": 1500.0, "high": 1520.0, "low": 1490.0, "close": 1510.0, "volume": 50000, "pre_close": 1500.0},
        {"symbol": "601088.SH", "open": 38.0, "high": 38.5, "low": 37.8, "close": 38.2, "volume": 2000000, "pre_close": 38.0},
        {"symbol": "688981.SH", "open": 50.0, "high": 50.5, "low": 49.5, "close": 50.0, "volume": 1500000, "pre_close": 50.0},
    ])
    adv_20 = pd.Series({
        "000001.SZ": 50000000.0,
        "600519.SH": 200000000.0,
        "601088.SH": 80000000.0,
        "688981.SH": 60000000.0,
    })

    # 1. 模拟一个激进的原始权重 (重仓第一只股票 70%, 第二只 30%)
    raw_weights = pd.Series({
        symbols[0]: 0.70,  # 严重超过单票 40% 上限
        symbols[1]: 0.30,
    })
    curr_weights = pd.Series({s: 0.0 for s in symbols})

    engine = RiskEngine(
        max_stock_weight=0.40,
        max_industry_weight=0.50,
        max_adv_participation=0.10,
        min_cash_buffer_ratio=0.03,
    )

    # 模拟历史净值与收益
    nav_history = pd.Series([1.0, 1.02, 1.05, 1.03])
    ret_history = pd.Series([0.001, 0.002, -0.001, 0.003])

    # 模拟 Barra 风格矩阵
    style_mat = pd.DataFrame({
        "SIZE": np.linspace(1.0, -1.0, len(symbols)),
        "VALUE": np.linspace(-0.5, 0.5, len(symbols)),
        "MOMENTUM": np.linspace(0.8, -0.2, len(symbols)),
    }, index=symbols)

    res = engine.process_target_weights(
        raw_target_weights=raw_weights,
        current_weights=curr_weights,
        portfolio_equity_cny=1000000.0,
        bars_slice=today_bars,
        adv_20_cny=adv_20,
        portfolio_nav_history=nav_history,
        portfolio_returns_history=ret_history,
        asset_style_matrix=style_mat,
    )

    # 验证单票上限合规
    assert res.final_weights.max() <= 0.40 + 1e-4
    # 验证现金缓冲垫保留 (>= 3%)
    assert res.cash_weight >= 0.03
    assert res.final_weights.sum() <= 0.97 + 1e-4
    # 验证诊断报告生成与集中度合规判定
    assert res.concentration_report.is_compliant is True
    assert "投研端全流程风控审计报告" in res.diagnostic_summary
    assert res.barra_report is not None
    assert res.stress_test_report is not None
