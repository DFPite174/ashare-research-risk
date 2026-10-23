"""
组合集中度约束与现金缓冲垫管理单元测试
"""

import pytest
import numpy as np
import pandas as pd

from risk.allocation import ConcentrationProjector, CashBufferManager


def test_concentration_projector_stock_and_industry():
    projector = ConcentrationProjector(max_stock_weight=0.40, max_industry_weight=0.50)

    # 初始权重: A 占 60% (超标), B 占 30%, C 占 10%
    # 行业分类: A, B 同属 "银行" (合计 90%, 严重超标)
    target_w = pd.Series({"STOCK_A": 0.60, "STOCK_B": 0.30, "STOCK_C": 0.10})
    industry_map = {"STOCK_A": "银行", "STOCK_B": "银行", "STOCK_C": "电子"}

    projected_w, rep = projector.project_weights(target_w, industry_mapping=industry_map)

    # 1. 单票上限必须 <= 40%
    assert projected_w["STOCK_A"] <= 0.40 + 1e-4
    # 2. 银行行业合计必须 <= 50%
    bank_weight = projected_w["STOCK_A"] + projected_w["STOCK_B"]
    assert bank_weight <= 0.50 + 1e-4
    # 3. 电子行业单票 <= 40% 且全组合不超过 1.0 (受行业与单票双重约束)
    assert projected_w["STOCK_C"] <= 0.40 + 1e-4
    assert projected_w.sum() <= 1.0 + 1e-4
    assert rep.is_compliant is True


def test_cash_buffer_manager():
    mgr = CashBufferManager(min_cash_buffer_ratio=0.05)  # 5% 现金缓冲

    # 全投资 100% 仓位
    target_w = pd.Series({"000001.SZ": 0.50, "600519.SH": 0.50})
    scaled_w, cash_ratio = mgr.apply_cash_buffer(target_w)

    assert abs(scaled_w.sum() - 0.95) < 1e-4
    assert abs(cash_ratio - 0.05) < 1e-4
    assert abs(scaled_w["000001.SZ"] - 0.475) < 1e-4
