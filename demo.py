#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI城建系统演示脚本
展示系统的主要功能和数据
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime

def show_database_summary():
    """显示数据库概览"""
    print("=" * 60)
    print("AI城建系统数据库概览")
    print("=" * 60)
    
    conn = sqlite3.connect('city_construction.db')
    
    # 项目基本信息统计
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM project_basic_info")
    project_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT project_status, COUNT(*) FROM project_basic_info GROUP BY project_status")
    status_stats = cursor.fetchall()
    
    cursor.execute("SELECT SUM(budget_amount) FROM project_basic_info")
    total_budget = cursor.fetchone()[0]
    
    print(f"📊 项目基本信息统计:")
    print(f"   总项目数: {project_count}")
    print(f"   总概算金额: {total_budget/10000:.2f} 万元")
    print(f"   项目状态分布:")
    for status, count in status_stats:
        print(f"     - {status}: {count} 个")
    
    print()
    
    # 资金安排统计
    cursor.execute("SELECT COUNT(*) FROM fund_arrangement")
    fund_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT fund_nature, COUNT(*), SUM(fund_approval_amount) FROM fund_arrangement GROUP BY fund_nature")
    fund_stats = cursor.fetchall()
    
    cursor.execute("SELECT SUM(fund_approval_amount) FROM fund_arrangement")
    total_fund = cursor.fetchone()[0]
    
    print(f"💰 资金安排统计:")
    print(f"   总资金安排记录: {fund_count}")
    print(f"   总资金批复金额: {total_fund/10000:.2f} 万元")
    print(f"   资金性质分布:")
    for nature, count, amount in fund_stats:
        print(f"     - {nature}: {count} 条记录, {amount/10000:.2f} 万元")
    
    print()
    
    # 债券资金统计
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN fund_nature = '一般债券' THEN fund_approval_amount ELSE 0 END) as general_bond,
            SUM(CASE WHEN fund_nature = '专项债券' THEN fund_approval_amount ELSE 0 END) as special_bond
        FROM fund_arrangement
    """)
    bond_stats = cursor.fetchone()
    total_bond = bond_stats[0] + bond_stats[1]
    
    print(f"🏛️ 债券资金统计:")
    print(f"   一般债券: {bond_stats[0]/10000:.2f} 万元")
    print(f"   专项债券: {bond_stats[1]/10000:.2f} 万元")
    print(f"   债券资金合计: {total_bond/10000:.2f} 万元")
    
    conn.close()

def show_sample_data():
    """显示样本数据"""
    print("\n" + "=" * 60)
    print("样本数据展示")
    print("=" * 60)
    
    conn = sqlite3.connect('city_construction.db')
    
    # 显示前5个项目
    print("📋 项目基本信息样本 (前5条):")
    df_projects = pd.read_sql_query("SELECT * FROM project_basic_info LIMIT 5", conn)
    for _, row in df_projects.iterrows():
        print(f"   项目名称: {row['project_name']}")
        print(f"   项目编码: {row['project_code']}")
        print(f"   建设单位: {row['construction_unit']}")
        print(f"   概算金额: {row['budget_amount']/10000:.2f} 万元")
        print(f"   项目状态: {row['project_status']}")
        print()
    
    # 显示前5条资金安排
    print("💵 资金安排样本 (前5条):")
    df_funds = pd.read_sql_query("SELECT * FROM fund_arrangement LIMIT 5", conn)
    for _, row in df_funds.iterrows():
        print(f"   项目编码: {row['project_code']}")
        print(f"   资金性质: {row['fund_nature']}")
        print(f"   批复金额: {row['fund_approval_amount']/10000:.2f} 万元")
        print(f"   经办人: {row['handler']}")
        print()
    
    conn.close()

def show_pivot_analysis():
    """显示数据透视分析"""
    print("\n" + "=" * 60)
    print("数据透视分析")
    print("=" * 60)
    
    conn = sqlite3.connect('city_construction.db')
    
    # 获取资金安排数据
    df = pd.read_sql_query("""
        SELECT project_code, project_name, fund_nature, fund_approval_amount
        FROM fund_arrangement
    """, conn)
    
    # 创建数据透视表
    pivot_df = df.pivot_table(
        index=['project_code', 'project_name'],
        columns='fund_nature',
        values='fund_approval_amount',
        aggfunc='sum',
        fill_value=0
    )
    
    # 计算债券资金合计
    pivot_df['债券资金合计'] = pivot_df.get('一般债券', 0) + pivot_df.get('专项债券', 0)
    
    # 重置索引
    pivot_df = pivot_df.reset_index()
    
    print("📊 数据透视表样本 (前5条):")
    for _, row in pivot_df.head().iterrows():
        print(f"   项目编码: {row['project_code']}")
        print(f"   项目名称: {row['project_name']}")
        print(f"   一般债券: {row.get('一般债券', 0)/10000:.2f} 万元")
        print(f"   专项债券: {row.get('专项债券', 0)/10000:.2f} 万元")
        print(f"   债券资金合计: {row['债券资金合计']/10000:.2f} 万元")
        print()
    
    # 统计债券资金使用情况
    total_general_bond = pivot_df['一般债券'].sum()
    total_special_bond = pivot_df['专项债券'].sum()
    total_bond = pivot_df['债券资金合计'].sum()
    
    print("🏛️ 债券资金使用统计:")
    print(f"   使用一般债券的项目数: {(pivot_df['一般债券'] > 0).sum()}")
    print(f"   使用专项债券的项目数: {(pivot_df['专项债券'] > 0).sum()}")
    print(f"   使用债券资金的项目数: {(pivot_df['债券资金合计'] > 0).sum()}")
    print(f"   一般债券总额: {total_general_bond/10000:.2f} 万元")
    print(f"   专项债券总额: {total_special_bond/10000:.2f} 万元")
    print(f"   债券资金总额: {total_bond/10000:.2f} 万元")
    
    conn.close()

def show_system_info():
    """显示系统信息"""
    print("\n" + "=" * 60)
    print("系统信息")
    print("=" * 60)
    
    print("🌐 Web界面访问地址: http://localhost:5000")
    print("📁 数据库文件: city_construction.db")
    print("📊 系统功能:")
    print("   - 基础数据管理 (项目基本信息、资金安排)")
    print("   - Excel导入导出")
    print("   - 数据透视表分析")
    print("   - 数据可视化图表")
    print("   - 智慧报表生成")
    print()
    print("🔧 技术架构:")
    print("   - 后端: Flask + SQLite")
    print("   - 前端: Bootstrap 5 + Chart.js")
    print("   - 数据处理: Pandas + OpenPyXL")
    print()
    print("📋 数据字段说明:")
    print("   项目状态: 在建、续建、完工、暂停")
    print("   资金性质: 一般债券、专项债券、中央综合财力、土地出让金、")
    print("           中央预算内投资、中央补助、省级补助")

def main():
    """主函数"""
    print("AI城建系统演示")
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        show_database_summary()
        show_sample_data()
        show_pivot_analysis()
        show_system_info()
        
        print("\n" + "=" * 60)
        print("✅ 系统演示完成！")
        print("请访问 http://localhost:5000 查看完整的Web界面")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")

if __name__ == '__main__':
    main()