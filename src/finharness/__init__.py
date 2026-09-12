"""FinHarness backend package.

Importing this package applies the compatibility settings the data layer needs
before any adapter pulls in pandas/akshare.
"""

from finharness._compat import apply_data_runtime_compat

apply_data_runtime_compat()
