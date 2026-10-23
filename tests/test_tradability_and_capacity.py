"""
可交易性约束 (涨跌停/停牌) 与 ADV 流动性容量风控单元测试
"""

import pytest
import numpy as np
import pandas as pd

from risk.tradability import (
    TradabilityPriceFilter,
    BlacklistFilter,
    ADVCapacityGuard,
)


def test_tradability_limit_up_down_and_suspended():
    filter_engine = TradabilityPriceFilter()

    # 构造测试行情：
    # 000001.SZ: 正常交易
    # 600519.SH: 一字涨停 (open=close=high=low=110, pre_close=100, +10%)
    # 601088.SH: 一字跌停 (open=close=high=low=90, pre_close=100, -10%)
    # 688981.SH: 停牌 (volume=0)
    bars = pd.DataFrame([
        {"symbol": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 100000, "pre_close": 10.0},
        {"symbol": "600519.SH", "open": 110.0, "high": 110.0, "low": 110.0, "close": 110.0, "volume": 50000, "pre_close": 100.0},
        {"symbol": "601088.SH", "open": 90.0, "high": 90.0, "low": 90.0, "close": 90.0, "volume": 20000, "pre_close": 100.0},
        {"symbol": "688981.SH", "open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 0.0, "pre_close": 50.0},
    ])

    curr_weights = pd.Series({"000001.SZ": 0.10, "600519.SH": 0.10, "601088.SH": 0.30, "688981.SH": 0.20})
    # 意图:
    # 000001: 增持到 0.20 (正常)
    # 600519: 增持到 0.40 (涨停无法买入，应锁定为 0.10)
    # 601088: 减持到 0.00 (跌停无法卖出，应锁定为 0.30)
    # 688981: 调整为 0.30 (停牌无法调仓，应锁定为 0.20)
    target_weights = pd.Series({"000001.SZ": 0.20, "600519.SH": 0.40, "601088.SH": 0.00, "688981.SH": 0.30})

    adj_w, rep = filter_engine.apply_filter(curr_weights, target_weights, bars)

    assert abs(adj_w["000001.SZ"] - 0.20) < 1e-4  # 正常执行
    assert abs(adj_w["600519.SH"] - 0.10) < 1e-4  # 涨停封锁
    assert abs(adj_w["601088.SH"] - 0.30) < 1e-4  # 跌停封锁
    assert abs(adj_w["688981.SH"] - 0.20) < 1e-4  # 停牌封锁

    assert rep.blocked_symbols["600519.SH"] == "LIMIT_UP_BUY_BLOCKED"
    assert rep.blocked_symbols["601088.SH"] == "LIMIT_DOWN_SELL_BLOCKED"
    assert rep.blocked_symbols["688981.SH"] == "SUSPENDED_BLOCKED"


def test_blacklist_filter():
    bl_filter = BlacklistFilter(static_blacklist={"000002.SZ", "ST0001"})
    target = pd.Series({"000001.SZ": 0.30, "000002.SZ": 0.30, "600519.SH": 0.40})

    filtered, removed = bl_filter.apply_filter(target, dynamic_blacklist={"600519.SH"})
    assert abs(filtered["000001.SZ"] - 0.30) < 1e-4
    assert filtered["000002.SZ"] == 0.0
    assert filtered["600519.SH"] == 0.0
    assert set(removed) == {"000002.SZ", "600519.SH"}


def test_adv_capacity_guard_and_impact():
    # 假设账户资金 1000 万元
    portfolio_equity = 10000000.0
    guard = ADVCapacityGuard(max_adv_participation=0.10)

    curr_w = pd.Series({"000001.SZ": 0.0, "SMALL_CAP": 0.0})
    # 计划买入 SMALL_CAP 20% (即 200 万元)
    target_w = pd.Series({"000001.SZ": 0.50, "SMALL_CAP": 0.20})

    # 假定 SMALL_CAP 过去 20 日日均成交额仅为 500 万元 (10% ADV 仅允许单日买 50 万元 = 5% 仓位)
    adv_series = pd.Series({"000001.SZ": 500000000.0, "SMALL_CAP": 5000000.0})

    adj_w, rep = guard.constrain_target_weights(portfolio_equity, curr_w, target_w, adv_series)

    # SMALL_CAP 调仓额被截断为 50 万元 (即 5% 仓位)
    assert abs(adj_w["SMALL_CAP"] - 0.05) < 1e-4
    assert rep.is_capacity_breached is True
    assert "SMALL_CAP" in rep.clipped_symbols
    assert abs(rep.clipped_symbols["SMALL_CAP"] - 1500000.0) < 1.0  # 削减了 150 万
    assert rep.market_impact_bps["SMALL_CAP"] > 0.0
