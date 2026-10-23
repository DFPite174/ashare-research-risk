# ashare-research-risk

> A 股量化投研端全流程风控与合规审查引擎 (A-Share Research-Side Risk Management & Compliance Engine)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-pep8-brightgreen.svg)](https://pep8.org/)

---

## 🌟 核心定位与架构 (Research-Side Risk vs Trading-Side Risk)

量化系统的风控分为**“投研端风控”**与**“交易端风控”**：
- **交易端风控 (OMS/EMS)**：侧重微秒级报单流控、防乌龙指、自成交防敲门、资金穿仓拦截与物理断网熔断。
- **投研端风控 (本子系统)**：侧重**真实性建模、流动性容量约束与组合风格暴露管理**。在策略研究与回测阶段即内生约束，消除实盘与回测之间的落差（Implementation Shortfall），确保优化器生成的权重能被实盘交易端无阻力执行。

```
ashare-research-risk/
├── tradability/                   # 1. 交易端约束下沉：可交易性与容量风控
│   ├── limit_price_filter.py      # 一字涨停不可买入、一字跌停不可卖出、停牌理论估值与持仓锁定
│   ├── blacklist_filter.py        # ST/*ST、退市整理期、违规立案黑名单剔除
│   └── adv_capacity_guard.py      # ADV 约束: 单日调仓量 <= 10% ADV，Almgren-Chriss 平方根冲击成本估算
├── allocation/                    # 2. 组合级配置合规投影器
│   ├── concentration_limits.py    # 集中度硬约束: 单票持仓上限 (<=40%)、申万单一行业暴露上限 (<=30%)
│   └── cash_buffer_manager.py     # 现金缓冲垫: 强制保留 2%~5% 现金头寸用于扣划法定手续费与除权除息
├── dynamic/                       # 3. 事中动态风控：波动率目标与阶梯降仓
│   ├── vol_targeting.py           # 波动率目标化机制 (Volatility Targeting): 市场高波时自动等比例降仓
│   └── drawdown_circuit_breaker.py# 组合净值多级阶梯熔断: 回撤 8% 减仓 50%，回撤 12% 减仓 80%，回撤 15% 强制清仓冷冻
├── attribution/                   # 4. 事后风险剖析：Barra 归因与黑天鹅压力测试
│   ├── barra_risk_model.py        # Barra 风格因子风险暴露 (规模 Size, 价值 Value, 动量 Mom, 波动率 Vol, 流动性 Liq)
│   └── historical_stress_test.py  # 历史极端情景回放 (2015 股灾、2020 疫情流动性危机、2024 微盘踩踏)
└── engine.py                      # 全流程风控执行中枢 (RiskEngine)
```

---

## 📐 核心算法与数学原理

### 1. ADV 流动性容量与 Almgren-Chriss 冲击成本
为防止策略构建出“纸面收益高但实盘无法承接”的微盘小票仓位，限制单日调仓额：
$$\Delta \text{Amount}_i \le 10\% \times \text{ADV}_{20, i}$$
并依据平方根市场冲击法则估算交易滑点与冲击：
$$\text{Impact}_i (\text{bps}) = \eta \cdot \sqrt{\frac{|\Delta \text{Volume}_i|}{\text{ADV}_i}} \times 10,000$$

### 2. 波动率目标机制 (Volatility Targeting)
维持投资组合的年化风险预算平滑，当滚动 20 日实现年化波动率 $\sigma_t$ 攀升时，动态调整多头仓位倍率：
$$s_t = \min\left(1.0, \frac{\sigma^*}{\sigma_t}\right)$$

### 3. 多级净值回撤熔断状态机
$$DD_t = \frac{\max_{\tau \le t} NAV_\tau - NAV_t}{\max_{\tau \le t} NAV_\tau}$$
- $DD_t < 8\%$: `NORMAL` 状态，允许 100% 仓位；
- $8\% \le DD_t < 12\%$: `WARNING_DERISK` 状态，强制限额 50% 仓位；
- $12\% \le DD_t < 15\%$: `CRITICAL_DEFENSE` 状态，强制限额 20% 仓位；
- $DD_t \ge 15\%$: `MELTDOWN_FROZEN` 状态，强制全平清仓，锁定现金避险。

---

## 🚀 快速上手 (Quickstart)

```bash
pip install -r requirements.txt
python examples/quickstart_risk.py
```

### Python 代码调用示例：

```python
import pandas as pd
from risk.engine import RiskEngine

# 1. 准备优化器产出的激进目标权重与当前持仓
raw_weights = pd.Series({"000001.SZ": 0.60, "600519.SH": 0.40})
curr_weights = pd.Series({"000001.SZ": 0.20, "600519.SH": 0.20})

# 2. 初始化投研风控引擎 (单票上限 40%，预留 3% 现金)
engine = RiskEngine(
    max_stock_weight=0.40,
    max_industry_weight=0.30,
    max_adv_participation=0.10,
    min_cash_buffer_ratio=0.03,
)

# 3. 执行风控清洗与合规投影
result = engine.process_target_weights(
    raw_target_weights=raw_weights,
    current_weights=curr_weights,
    portfolio_equity_cny=10000000.0,
)

print("合规目标权重:\n", result.final_weights)
print("预留现金比例:", result.cash_weight)
print(result.diagnostic_summary)
```

---

## 🧪 运行单元测试

```bash
pytest tests/ -v
```

---

## 📄 开源许可证

本项目采用 [MIT License](LICENSE) 授权。
