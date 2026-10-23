"""
企业级量化投研平台 - 可交易性风控与涨跌停/停牌过滤器 (Tradability Price Filter)
吸纳真实交易端物理约束：
1. 涨停板 (Limit Up): 若持仓意图为增持/买入，则禁止建仓，权重锁定为当前持仓权重；
2. 跌停板 (Limit Down): 若持仓意图为减持/卖出，则禁止抛售，权重锁定为当前持仓权重；
3. 停牌股 (Suspended): 交易量为 0，持仓严格冻结无法调仓。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradabilityConstraintReport:
    """可交易性调整明细"""
    blocked_symbols: Dict[str, str]  # symbol -> 封堵原因 ('LIMIT_UP_BUY_BLOCKED', 'LIMIT_DOWN_SELL_BLOCKED', 'SUSPENDED_BLOCKED')
    original_target_sum: float
    adjusted_target_sum: float


class TradabilityPriceFilter:
    """
    涨跌停与停牌可交易性过滤器
    """

    def __init__(
        self,
        st_limit_threshold: float = 0.048,
        main_limit_threshold: float = 0.098,
        gem_limit_threshold: float = 0.195,
    ):
        self.st_limit_threshold = st_limit_threshold
        self.main_limit_threshold = main_limit_threshold
        self.gem_limit_threshold = gem_limit_threshold

    def is_limit_up(self, row: pd.Series) -> bool:
        """判定当日是否为封板涨停 (一字涨停或触及涨停收盘)"""
        if "close" not in row or "open" not in row:
            return False
        high = float(row.get("high", row["close"]))
        close = float(row["close"])
        open_px = float(row["open"])

        # 估算涨幅
        pre_close = float(row.get("pre_close", open_px))
        if pre_close <= 1e-6:
            return False
        pct_change = (close - pre_close) / pre_close

        symbol = str(row.get("symbol", ""))
        threshold = self.gem_limit_threshold if symbol.startswith(("300", "688")) else self.main_limit_threshold
        return (close >= open_px) and (pct_change >= threshold) and (abs(high - close) <= 0.01)

    def is_limit_down(self, row: pd.Series) -> bool:
        """判定当日是否为封板跌停 (一字跌停或触及跌停收盘)"""
        if "close" not in row or "open" not in row:
            return False
        low = float(row.get("low", row["close"]))
        close = float(row["close"])
        open_px = float(row["open"])

        pre_close = float(row.get("pre_close", open_px))
        if pre_close <= 1e-6:
            return False
        pct_change = (close - pre_close) / pre_close

        symbol = str(row.get("symbol", ""))
        threshold = -self.gem_limit_threshold if symbol.startswith(("300", "688")) else -self.main_limit_threshold
        return (close <= open_px) and (pct_change <= threshold) and (abs(low - close) <= 0.01)

    def is_suspended(self, row: pd.Series) -> bool:
        """判定当日是否停牌 (成交量为 0 或 NaN)"""
        vol = float(row.get("volume", 0.0))
        return vol <= 0.0 or not np.isfinite(vol)

    def apply_filter(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        bars_slice: pd.DataFrame,
    ) -> Tuple[pd.Series, TradabilityConstraintReport]:
        """
        结合当日行情状态，校验并调整目标权重 TargetWeights
        :param current_weights: 当前实际持仓权重 (Series: symbol -> weight)
        :param target_weights: 优化器拟分配的目标权重 (Series: symbol -> weight)
        :param bars_slice: 当日所有相关标的的行情切片 (包含 symbol, open, high, low, close, volume, pre_close 等)
        :return: (调整后的合规目标权重, 封堵报告)
        """
        all_symbols = sorted(list(set(current_weights.index).union(set(target_weights.index))))
        curr_aligned = current_weights.reindex(all_symbols).fillna(0.0)
        tgt_aligned = target_weights.reindex(all_symbols).fillna(0.0)

        # 整理行情索引以供高效检索
        bars_indexed = bars_slice.set_index("symbol") if "symbol" in bars_slice.columns else bars_slice

        adjusted_weights = tgt_aligned.copy()
        blocked_reasons: Dict[str, str] = {}

        for sym in all_symbols:
            w_curr = float(curr_aligned.loc[sym])
            w_tgt = float(tgt_aligned.loc[sym])
            delta = w_tgt - w_curr

            if abs(delta) < 1e-6:
                continue

            if sym not in bars_indexed.index:
                # 行情缺失视为停牌
                adjusted_weights.loc[sym] = w_curr
                blocked_reasons[sym] = "DATA_MISSING_SUSPENDED_BLOCKED"
                continue

            row = bars_indexed.loc[sym]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            # 1. 停牌检查
            if self.is_suspended(row):
                adjusted_weights.loc[sym] = w_curr
                blocked_reasons[sym] = "SUSPENDED_BLOCKED"
                continue

            # 2. 涨停且意图买入
            if delta > 0 and self.is_limit_up(row):
                adjusted_weights.loc[sym] = w_curr
                blocked_reasons[sym] = "LIMIT_UP_BUY_BLOCKED"
                continue

            # 3. 跌停且意图卖出
            if delta < 0 and self.is_limit_down(row):
                adjusted_weights.loc[sym] = w_curr
                blocked_reasons[sym] = "LIMIT_DOWN_SELL_BLOCKED"
                continue

        report = TradabilityConstraintReport(
            blocked_symbols=blocked_reasons,
            original_target_sum=float(tgt_aligned.sum()),
            adjusted_target_sum=float(adjusted_weights.sum()),
        )

        return adjusted_weights, report
