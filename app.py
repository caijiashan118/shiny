#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI城建系统主应用
包含基础数据、数据可视化、智慧报表生成三个模块
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
import random
import json
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-city-construction-system'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 数据库初始化
def init_db():
    """初始化数据库"""
    conn = sqlite3.connect('city_construction.db')
    cursor = conn.cursor()
    
    # 创建项目基本信息表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_basic_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            project_code TEXT UNIQUE NOT NULL,
            construction_unit TEXT NOT NULL,
            project_department TEXT NOT NULL,
            approval_time DATE,
            budget_amount REAL,
            planned_start_time DATE,
            planned_completion_time DATE,
            project_status TEXT CHECK(project_status IN ('在建', '续建', '完工', '暂停')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建资金安排表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fund_arrangement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            project_code TEXT NOT NULL,
            construction_unit TEXT NOT NULL,
            project_department TEXT NOT NULL,
            budget_amount REAL,
            fund_approval_amount REAL,
            approval_date DATE,
            fund_nature TEXT CHECK(fund_nature IN ('一般债券', '专项债券', '中央综合财力', '土地出让金', '中央预算内投资', '中央补助', '省级补助')),
            handler TEXT,
            handling_department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_code) REFERENCES project_basic_info (project_code)
        )
    ''')
    
    conn.commit()
    conn.close()

# 生成模拟数据
def generate_mock_data():
    """生成模拟数据"""
    conn = sqlite3.connect('city_construction.db')
    cursor = conn.cursor()
    
    # 检查是否已有数据
    cursor.execute("SELECT COUNT(*) FROM project_basic_info")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    # 生成200个项目基本信息
    project_names = [
        "城市道路改造工程", "污水处理厂建设", "公园绿化提升", "桥梁维修加固",
        "地下管网改造", "学校建设", "医院扩建", "保障房建设", "商业综合体",
        "交通枢纽建设", "文化中心建设", "体育场馆建设", "科技园区建设",
        "住宅小区建设", "工业园区建设", "物流园区建设", "旅游设施建设",
        "环保设施建设", "水利工程建设", "电力设施建设"
    ]
    
    construction_units = [
        "市建设局", "市交通局", "市教育局", "市卫健委", "市文旅局",
        "市发改委", "市财政局", "市环保局", "市水利局", "市电力局"
    ]
    
    departments = [
        "建设处", "规划处", "工程处", "财务处", "监管处", "审批处"
    ]
    
    statuses = ['在建', '续建', '完工', '暂停']
    
    projects = []
    for i in range(200):
        project_name = f"{random.choice(project_names)}{i+1:03d}"
        project_code = f"PRJ{datetime.now().year}{i+1:04d}"
        construction_unit = random.choice(construction_units)
        project_department = random.choice(departments)
        approval_time = datetime.now() - timedelta(days=random.randint(30, 1000))
        budget_amount = random.randint(500, 20000) * 10000  # 500-20000万元
        planned_start_time = approval_time + timedelta(days=random.randint(30, 180))
        planned_completion_time = planned_start_time + timedelta(days=random.randint(180, 1095))
        project_status = random.choice(statuses)
        
        projects.append((
            project_name, project_code, construction_unit, project_department,
            approval_time.strftime('%Y-%m-%d'), budget_amount,
            planned_start_time.strftime('%Y-%m-%d'),
            planned_completion_time.strftime('%Y-%m-%d'), project_status
        ))
    
    cursor.executemany('''
        INSERT INTO project_basic_info 
        (project_name, project_code, construction_unit, project_department,
         approval_time, budget_amount, planned_start_time, planned_completion_time, project_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', projects)
    
    # 生成1000条资金安排记录
    fund_natures = ['一般债券', '专项债券', '中央综合财力', '土地出让金', 
                   '中央预算内投资', '中央补助', '省级补助']
    
    handlers = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"]
    handling_departments = ["财务处", "建设处", "规划处", "工程处", "监管处"]
    
    fund_arrangements = []
    for i in range(1000):
        # 随机选择一个项目
        project_idx = random.randint(0, 199)
        project = projects[project_idx]
        
        project_name = project[0]
        project_code = project[1]
        construction_unit = project[2]
        project_department = project[3]
        budget_amount = project[5]
        
        # 资金批复金额不超过概算批复金额
        fund_approval_amount = random.randint(100, int(budget_amount/10000)) * 10000
        approval_date = datetime.now() - timedelta(days=random.randint(1, 500))
        fund_nature = random.choice(fund_natures)
        handler = random.choice(handlers)
        handling_department = random.choice(handling_departments)
        
        fund_arrangements.append((
            project_name, project_code, construction_unit, project_department,
            budget_amount, fund_approval_amount, approval_date.strftime('%Y-%m-%d'),
            fund_nature, handler, handling_department
        ))
    
    cursor.executemany('''
        INSERT INTO fund_arrangement 
        (project_name, project_code, construction_unit, project_department,
         budget_amount, fund_approval_amount, approval_date, fund_nature, handler, handling_department)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', fund_arrangements)
    
    conn.commit()
    conn.close()

# 路由定义
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/projects')
def get_projects():
    """获取项目基本信息列表"""
    conn = sqlite3.connect('city_construction.db')
    df = pd.read_sql_query("SELECT * FROM project_basic_info ORDER BY created_at DESC", conn)
    conn.close()
    return jsonify(df.to_dict('records'))

@app.route('/api/funds')
def get_funds():
    """获取资金安排列表"""
    conn = sqlite3.connect('city_construction.db')
    df = pd.read_sql_query("SELECT * FROM fund_arrangement ORDER BY created_at DESC", conn)
    conn.close()
    return jsonify(df.to_dict('records'))

@app.route('/api/pivot')
def get_pivot_data():
    """获取数据透视表数据"""
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
    
    conn.close()
    return jsonify(pivot_df.to_dict('records'))

@app.route('/api/export/projects')
def export_projects():
    """导出项目基本信息到Excel"""
    conn = sqlite3.connect('city_construction.db')
    df = pd.read_sql_query("SELECT * FROM project_basic_info", conn)
    conn.close()
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='项目基本信息', index=False)
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'项目基本信息_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )

@app.route('/api/export/funds')
def export_funds():
    """导出资金安排到Excel"""
    conn = sqlite3.connect('city_construction.db')
    df = pd.read_sql_query("SELECT * FROM fund_arrangement", conn)
    conn.close()
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='资金安排', index=False)
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'资金安排_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )

@app.route('/api/export/pivot')
def export_pivot():
    """导出数据透视表到Excel"""
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
    
    conn.close()
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pivot_df.to_excel(writer, sheet_name='资金安排透视表', index=False)
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'资金安排透视表_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )

@app.route('/api/upload/projects', methods=['POST'])
def upload_projects():
    """上传项目基本信息Excel文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and file.filename.endswith('.xlsx'):
        try:
            df = pd.read_excel(file)
            
            # 验证必要的列
            required_columns = ['项目名称', '项目编码', '建设单位', '项目主管部门', 
                              '立项时间', '概算批复金额', '预计开工时间', '预计完工时间', '项目状态']
            
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': f'缺少必要的列: {required_columns}'}), 400
            
            # 插入数据库
            conn = sqlite3.connect('city_construction.db')
            df.to_sql('project_basic_info', conn, if_exists='append', index=False)
            conn.close()
            
            return jsonify({'message': '上传成功', 'count': len(df)})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': '文件格式不正确'}), 400

@app.route('/api/upload/funds', methods=['POST'])
def upload_funds():
    """上传资金安排Excel文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and file.filename.endswith('.xlsx'):
        try:
            df = pd.read_excel(file)
            
            # 验证必要的列
            required_columns = ['项目名称', '项目编码', '建设单位', '项目主管部门',
                              '概算批复金额', '资金批复金额', '批复日期', '资金性质', '经办人', '经办处室']
            
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': f'缺少必要的列: {required_columns}'}), 400
            
            # 插入数据库
            conn = sqlite3.connect('city_construction.db')
            df.to_sql('fund_arrangement', conn, if_exists='append', index=False)
            conn.close()
            
            return jsonify({'message': '上传成功', 'count': len(df)})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': '文件格式不正确'}), 400

if __name__ == '__main__':
    init_db()
    generate_mock_data()
    app.run(debug=True, host='0.0.0.0', port=5000)