"""
ashare-research-risk 快速入门演示 (Quickstart Demo)
演示：
1. 涨跌停与停牌可交易性校验
2. ADV 流动性容量约束与 Almgren-Chriss 冲击成本
3. 单票 (<=40%) 与行业 (<=30%) 集中度合规投影
4. 事中动态风控：波动率目标化与净值阶梯回撤熔断
5. Barra 风格因子风险暴露与极端历史情景压力测试
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from risk.engine import RiskEngine


def main():
    print("=" * 80)
    print("      ashare-research-risk: 投研端全流程风控与合规审查套件演示")
    print("=" * 80)

    symbols = ["000001.SZ", "600519.SH", "601088.SH", "688981.SH"]

    # 1. 模拟当前持仓权重与优化器生成的激进目标权重
    curr_weights = pd.Series({"000001.SZ": 0.10, "600519.SH": 0.20, "601088.SH": 0.10, "688981.SH": 0.00})
    # 激进权重: 600519 激增至 60% (严重超标), 601088 抛空
    raw_weights = pd.Series({"000001.SZ": 0.20, "600519.SH": 0.60, "601088.SH": 0.00, "688981.SH": 0.20})

    # 2. 模拟当日行情：600519 一字涨停 (无法增持), 601088 跌停 (无法抛售)
    bars_today = pd.DataFrame([
        {"symbol": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 1000000, "pre_close": 10.0},
        {"symbol": "600519.SH", "open": 1650.0, "high": 1650.0, "low": 1650.0, "close": 1650.0, "volume": 50000, "pre_close": 1500.0},
        {"symbol": "601088.SH", "open": 34.2, "high": 34.2, "low": 34.2, "close": 34.2, "volume": 20000, "pre_close": 38.0},
        {"symbol": "688981.SH", "open": 50.0, "high": 50.5, "low": 49.5, "close": 50.0, "volume": 1500000, "pre_close": 50.0},
    ])

    adv_20 = pd.Series({
        "000001.SZ": 50000000.0,
        "600519.SH": 200000000.0,
        "601088.SH": 80000000.0,
        "688981.SH": 60000000.0,
    })

    industry_map = {
        "000001.SZ": "银行",
        "600519.SH": "食品饮料",
        "601088.SH": "煤炭",
        "688981.SH": "电子",
    }

    # 3. 实例化投研风控引擎
    engine = RiskEngine(
        max_stock_weight=0.40,
        max_industry_weight=0.40,
        max_adv_participation=0.10,
        min_cash_buffer_ratio=0.03,
    )

    # 4. 执行全流程风控处理
    res = engine.process_target_weights(
        raw_target_weights=raw_weights,
        current_weights=curr_weights,
        portfolio_equity_cny=5000000.0,
        bars_slice=bars_today,
        adv_20_cny=adv_20,
        industry_mapping=industry_map,
    )

    print("\n[1] 权重调整与合规投影结果:")
    comparison_df = pd.DataFrame({
        "Current": curr_weights,
        "Raw Target": raw_weights,
        "Risk-Controlled": res.final_weights,
    }).round(4)
    print(comparison_df.to_string())
    print(f"现金预留比例: {res.cash_weight:.2%}")

    print("\n[2] 诊断审计报告:")
    print(res.diagnostic_summary)


if __name__ == "__main__":
    main()
