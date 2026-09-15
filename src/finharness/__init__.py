"""FinHarness 后端包。

导入本包会应用数据层所需的兼容性设置，
且须在任何适配器引入 pandas/akshare 之前完成。
"""

from finharness.utils.compat import apply_data_runtime_compat

apply_data_runtime_compat()
