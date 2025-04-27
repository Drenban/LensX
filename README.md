# LensX

量化项目架构：C++与Python结合
1. 数据层（Python）

功能：获取、清洗和存储市场数据。
工具：
数据源：Tushare（A股）、Alpha Vantage（全球）、CCXT（加密货币）。
数据库：SQLite（轻量）、PostgreSQL（企业级）。
库：pandas（数据处理）、numpy（数值计算）。


输出：清洗后的历史数据（如OHLCV）、实时数据流。

2. 策略开发与回测层（Python）

功能：开发交易策略，基于历史数据回测。
工具：
回测框架：backtrader、zipline。
技术分析：TA-Lib、pandas-ta。
机器学习：scikit-learn、tensorflow（可选）。


输出：优化后的策略参数、交易信号。

3. 交易执行层（C++）

功能：实时处理市场数据，执行交易信号。
工具：
库：QuantLib（金融模型）、TA-Lib（技术指标）、Boost（网络/并发）。
API：Interactive Brokers、Binance API、自定义交易所接口。


实现：
实时数据解析（如WebSocket流）。
低延迟订单执行（限价单、市价单）。
风控逻辑（如仓位限制、止损）。



4. 互操作层（Python与C++）

功能：Python调用C++模块，或C++接收Python信号。
工具：
pybind11：Python调用C++函数。
Cython：加速Python代码或桥接C++。
消息队列：ZeroMQ、Redis（Python与C++通信）。


示例：
Python生成信号，传递给C++执行。
C++计算高性能指标（如VaR），返回给Python可视化。



5. 监控与可视化层（Python）

功能：实时监控交易状态，生成报告。
工具：
可视化：matplotlib、seaborn、plotly。
日志：logging模块、ELK栈（大规模系统）。


输出：交易性能图表、风险报告。

