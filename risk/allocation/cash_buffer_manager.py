"""
企业级量化投研平台 - 现金缓冲垫与保证金流动性管理 (Cash Buffer Manager)
强制为实盘预留 2%~5% 的现金缓冲底仓：
1. 支付交易产生的法定印花税、过户费与最低 5 元佣金；
2. 应对日间持仓可能产生的分红除息估值变动与临时赎回冲击。
"""

from __future__ import annotations
from typing import Tuple
import pandas as pd


class CashBufferManager:
    """
    现金缓冲风控管理
    """

    def __init__(self, min_cash_buffer_ratio: float = 0.03):
        """
        :param min_cash_buffer_ratio: 最小预留现金比例 (默认 3%)
        """
        self.min_cash_buffer_ratio = max(min(min_cash_buffer_ratio, 0.50), 0.0)

    def apply_cash_buffer(self, target_weights: pd.Series) -> Tuple[pd.Series, float]:
        """
        压缩风险资产权重，为组合预留法定现金缓冲
        :param target_weights: 标的拟分配权重
        :return: (压缩后的标的权重, 实际预留现金比例)
        """
        w = target_weights.copy()
        raw_sum = float(w.sum())
        max_allowed_stock_weight = 1.0 - self.min_cash_buffer_ratio

        if raw_sum > max_allowed_stock_weight:
            scale = max_allowed_stock_weight / raw_sum
            w = w * scale
            cash_ratio = self.min_cash_buffer_ratio
        else:
            cash_ratio = 1.0 - raw_sum

        return w, float(cash_ratio)
