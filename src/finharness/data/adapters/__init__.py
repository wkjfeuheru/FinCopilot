"""外部数据适配器。"""

from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.adapters.base import AdapterError, DataAdapter
from finharness.data.adapters.eastmoney_report_adapter import EastmoneyReportAdapter
from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter
from finharness.data.adapters.tavily_adapter import TavilyAdapter
from finharness.data.adapters.tushare_adapter import TushareAdapter

__all__ = [
    "AdapterError",
    "AkShareAdapter",
    "DataAdapter",
    "EastmoneyReportAdapter",
    "FuyaoMcpAdapter",
    "TavilyAdapter",
    "TushareAdapter",
]
