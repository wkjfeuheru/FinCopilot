"""计算子进程树的系统级生命周期；Windows 子进程在绑定 Job 前等待输入。"""

from __future__ import annotations

import os
import signal


class ProcessTree:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.handle = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            for name, args, result in (
                ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
                ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
                ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
                ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
                ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            ):
                function = getattr(kernel, name)
                function.argtypes, function.restype = args, result
            self.kernel = kernel
            self.handle = kernel.CreateJobObjectW(None, None)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            process = kernel.OpenProcess(0x0100 | 0x0001, False, pid)
            try:
                if not process or not kernel.AssignProcessToJobObject(self.handle, process):
                    error = ctypes.WinError(ctypes.get_last_error())
                    kernel.CloseHandle(self.handle)
                    self.handle = None
                    raise error
            finally:
                if process:
                    kernel.CloseHandle(process)

    def terminate(self) -> None:
        if os.name == "nt":
            if self.handle:
                self.kernel.TerminateJobObject(self.handle, 1)
                self.kernel.CloseHandle(self.handle)
                self.handle = None
        else:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
