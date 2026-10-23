"""
企业级量化投研平台 - 股票池合规黑名单与垃圾股过滤器 (Blacklist Filter)
实现：
1. ST / *ST 风险警示标的过滤；
2. 退市整理期、违规立案调查黑名单拦截；
3. 将黑名单标的目标权重归零，避免投研端选出高风险壳股产生虚假 Alpha。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


class BlacklistFilter:
    """
    黑名单与合规标的过滤器
    """

    def __init__(self, static_blacklist: Optional[Set[str]] = None):
        self.static_blacklist: Set[str] = set(static_blacklist or [])

    def add_to_blacklist(self, symbols: List[str]) -> None:
        self.static_blacklist.update(symbols)

    def remove_from_blacklist(self, symbols: List[str]) -> None:
        self.static_blacklist.difference_update(symbols)

    def apply_filter(
        self,
        target_weights: pd.Series,
        dynamic_blacklist: Optional[Set[str]] = None,
        st_symbols: Optional[Set[str]] = None,
    ) -> Tuple[pd.Series, List[str]]:
        """
        过滤黑名单股票并将目标权重置 0
        :param target_weights: Series(symbol -> weight)
        :param dynamic_blacklist: 动态外部黑名单 (如当日最新立案公告标的)
        :param st_symbols: 当日为 ST / *ST 的标的集合
        :return: (过滤后的目标权重, 被剔除的黑名单标的列表)
        """
        combined_blacklist = set(self.static_blacklist)
        if dynamic_blacklist:
            combined_blacklist.update(dynamic_blacklist)
        if st_symbols:
            combined_blacklist.update(st_symbols)

        filtered_weights = target_weights.copy()
        removed_symbols = []

        for sym in target_weights.index:
            if sym in combined_blacklist:
                if filtered_weights.loc[sym] > 0.0:
                    removed_symbols.append(sym)
                    filtered_weights.loc[sym] = 0.0

        return filtered_weights, removed_symbols
