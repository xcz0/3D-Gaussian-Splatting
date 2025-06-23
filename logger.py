#!/usr/bin/env python3
"""
日志记录模块
"""

import time


class Logger:
    """简单的日志记录器"""

    def __init__(self, name: str):
        self.name = name

    def _log(self, level: str, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level:<7} - {self.name}: {message}")

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

    def error(self, message: str):
        self._log("ERROR", message)

    def success(self, message: str):
        self._log("SUCCESS", message)
