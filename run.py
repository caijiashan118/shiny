#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI城建系统启动脚本
"""

import os
import sys
import subprocess

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import flask
        import pandas
        import openpyxl
        import plotly
        import dash
        import sqlalchemy
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("AI城建系统启动中...")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 启动应用
    print("正在启动AI城建系统...")
    print("系统将在 http://localhost:5000 启动")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n系统已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()