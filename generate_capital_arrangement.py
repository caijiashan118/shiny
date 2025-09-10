#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资金安排表生成器
生成包含1000条模拟数据的Excel表格
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

# 设置随机种子以确保可重复性
random.seed(42)

# 定义字段名称
COLUMNS = [
    '项目名称',
    '项目编码',
    '建设单位',
    '项目主管部门',
    '概算批复金额',
    '资金批复金额',
    '批复日期',
    '资金性质',
    '经办人',
    '经办处室'
]

# 资金性质选项
FUND_TYPES = [
    '一般债券',
    '专项债券',
    '中央综合财力',
    '土地出让金',
    '中央预算内投资',
    '中央补助',
    '省级补助'
]

# 模拟数据生成函数
def generate_project_names():
    """生成项目名称"""
    prefixes = ['城市', '乡村', '工业园区', '高新区', '经济开发区', '产业园', '科技园']
    projects = ['基础设施建设', '道路改造', '管网升级', '环境整治', '公共服务设施建设', 
                '智慧城市建设', '产业升级改造', '生态修复工程', '民生工程', '教育设施建设',
                '医疗设施建设', '文化设施建设', '体育设施建设', '养老设施建设', '保障房建设']
    return [f"{random.choice(prefixes)}{random.choice(projects)}项目" for _ in range(1000)]

def generate_project_codes():
    """生成项目编码"""
    return [f"PROJ{datetime.now().year}{i:04d}" for i in range(1, 1001)]

def generate_construction_units():
    """生成建设单位"""
    units = ['市建设投资集团', '市城市建设发展有限公司', '市基础设施投资有限公司',
             '市交通投资集团', '市水务集团', '市环保投资公司', '市教育投资公司',
             '市卫生投资公司', '市文化投资公司', '市体育投资公司', '区建设投资公司',
             '县建设投资公司', '开发区投资公司', '高新区投资公司', '经济区投资公司']
    return [random.choice(units) for _ in range(1000)]

def generate_departments():
    """生成项目主管部门"""
    departments = ['发展和改革委员会', '住房和城乡建设局', '交通运输局', '水务局',
                   '生态环境局', '教育局', '卫生健康委员会', '文化和旅游局', '体育局',
                   '民政局', '自然资源和规划局', '科技局', '工业和信息化局', '农业农村局']
    return [random.choice(departments) for _ in range(1000)]

def generate_amounts(n=1000):
    """生成概算批复金额和资金批复金额（万元）"""
    budgets = []
    approvals = []
    for _ in range(n):
        # 概算批复金额在500万到50000万之间
        budget = round(random.uniform(500, 50000), 2)
        # 资金批复金额通常小于或等于概算批复金额
        approval = round(budget * random.uniform(0.6, 1.0), 2)
        budgets.append(budget)
        approvals.append(approval)
    return budgets, approvals

def generate_approval_dates():
    """生成批复日期"""
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 9, 10)
    dates = []
    for _ in range(1000):
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + timedelta(days=random_number_of_days)
        dates.append(random_date.strftime('%Y-%m-%d'))
    return dates

def generate_fund_types():
    """生成资金性质"""
    return [random.choice(FUND_TYPES) for _ in range(1000)]

def generate_handlers():
    """生成经办人"""
    surnames = ['张', '王', '李', '赵', '刘', '陈', '杨', '黄', '周', '吴', '徐', '孙', '马', '朱', '胡']
    names = ['伟', '芳', '秀英', '敏', '静', '丽', '强', '磊', '洋', '艳', '勇', '军', '杰', '娟', '涛']
    return [f"{random.choice(surnames)}{random.choice(names)}" for _ in range(1000)]

def generate_offices():
    """生成经办处室"""
    offices = ['综合处', '规划处', '投资处', '财务处', '项目管理处', '政策法规处',
               '科技创新处', '区域发展处', '社会发展处', '基础设施处', '产业发展处',
               '农村发展处', '能源发展处', '资源环境处', '经济运行处']
    return [random.choice(offices) for _ in range(1000)]

def create_excel_file():
    """创建Excel文件并添加数据"""
    # 生成数据
    project_names = generate_project_names()
    project_codes = generate_project_codes()
    construction_units = generate_construction_units()
    departments = generate_departments()
    budgets, approvals = generate_amounts()
    approval_dates = generate_approval_dates()
    fund_types = generate_fund_types()
    handlers = generate_handlers()
    offices = generate_offices()
    
    # 创建DataFrame
    data = {
        '项目名称': project_names,
        '项目编码': project_codes,
        '建设单位': construction_units,
        '项目主管部门': departments,
        '概算批复金额': budgets,
        '资金批复金额': approvals,
        '批复日期': approval_dates,
        '资金性质': fund_types,
        '经办人': handlers,
        '经办处室': offices
    }
    
    df = pd.DataFrame(data)
    
    # 创建Excel文件
    filename = '/workspace/资金安排表.xlsx'
    
    # 使用openpyxl引擎写入Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='资金安排表', index=False)
        
        # 获取worksheet对象
        worksheet = writer.sheets['资金安排表']
        
        # 设置标题行样式
        header_font = Font(bold=True, size=12)
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        header_alignment = Alignment(horizontal='center', vertical='center')
        
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.font = Font(bold=True, size=12, color='FFFFFF')
        
        # 设置列宽
        column_widths = {
            'A': 30,  # 项目名称
            'B': 15,  # 项目编码
            'C': 25,  # 建设单位
            'D': 20,  # 项目主管部门
            'E': 15,  # 概算批复金额
            'F': 15,  # 资金批复金额
            'G': 12,  # 批复日期
            'H': 15,  # 资金性质
            'I': 10,  # 经办人
            'J': 15   # 经办处室
        }
        
        for column, width in column_widths.items():
            worksheet.column_dimensions[column].width = width
        
        # 设置数据格式
        for row in worksheet.iter_rows(min_row=2, max_row=1001):
            # 金额列格式化为货币格式
            row[4].number_format = '#,##0.00'  # 概算批复金额
            row[5].number_format = '#,##0.00'  # 资金批复金额
            
            # 设置所有单元格的对齐方式
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # 添加数据验证（下拉列表）到资金性质列
        dv = DataValidation(
            type="list",
            formula1=f'"{",".join(FUND_TYPES)}"',
            allow_blank=False,
            showDropDown=True
        )
        dv.error = '请选择有效的资金性质'
        dv.errorTitle = '输入错误'
        dv.prompt = '请从下拉列表中选择'
        dv.promptTitle = '资金性质'
        
        # 应用数据验证到H列（资金性质）
        dv.add(f'H2:H1001')
        worksheet.add_data_validation(dv)
        
        # 添加边框
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for row in worksheet.iter_rows(min_row=1, max_row=1001, min_col=1, max_col=10):
            for cell in row:
                cell.border = thin_border
        
        # 冻结首行
        worksheet.freeze_panes = 'A2'
    
    print(f"Excel文件已成功创建：{filename}")
    print(f"共生成 {len(df)} 条数据记录")
    print("\n数据概览：")
    print(df.head(10))
    print("\n资金性质分布：")
    print(df['资金性质'].value_counts())

if __name__ == "__main__":
    create_excel_file()