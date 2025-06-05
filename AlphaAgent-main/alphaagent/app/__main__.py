#!/usr/bin/env python3
"""
AlphaAgent App模块主入口点
"""

import sys
import os

# 确保输出立即显示
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 设置环境变量强制输出
os.environ['PYTHONUNBUFFERED'] = '1'

from alphaagent.app.cli import app

if __name__ == "__main__":
    app() 