# AI城建系统架构说明

## 系统整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AI城建系统架构                            │
├─────────────────────────────────────────────────────────────┤
│  前端层 (Frontend Layer)                                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   Web界面       │ │   数据可视化     │ │   智慧报表      │ │
│  │ Bootstrap 5     │ │ Chart.js        │ │ 自动生成        │ │
│  │ + Chart.js      │ │ 饼图、柱状图     │ │ 概览、分析      │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  应用层 (Application Layer)                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   Flask应用     │ │   API接口       │ │   业务逻辑      │ │
│  │   app.py        │ │ 数据获取、导入导出│ │ 数据处理、报表生成│ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  数据层 (Data Layer)                                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ SQLite数据库    │ │ 项目基本信息表   │ │ 资金安排表      │ │
│  │ city_construction│ │ 200条记录       │ │ 1000条记录      │ │
│  │ .db             │ │                 │ │                 │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│  ┌─────────────────┐                                       │
│  │ 数据透视表       │                                       │
│  │ 按项目编码和资金性质统计│                                 │
│  └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

## 核心功能模块

### 1. 基础数据模块
- **项目基本信息表**
  - 字段：项目名称、项目编码、建设单位、项目主管部门、立项时间、概算批复金额、预计开工时间、预计完工时间、项目状态
  - 约束：项目状态限制为"在建、续建、完工、暂停"
  - 功能：CRUD操作、Excel导入导出

- **资金安排表**
  - 字段：项目名称、项目编码、建设单位、项目主管部门、概算批复金额、资金批复金额、批复日期、资金性质、经办人、经办处室
  - 约束：资金性质限制为"一般债券、专项债券、中央综合财力、土地出让金、中央预算内投资、中央补助、省级补助"
  - 功能：CRUD操作、Excel导入导出、与项目基本信息表通过项目编码关联

### 2. 数据可视化模块
- **项目状态分布图**：饼图展示项目状态分布
- **资金性质分布图**：柱状图展示各类资金性质使用情况
- **投资趋势图**：折线图展示项目投资时间趋势

### 3. 智慧报表生成模块
- **项目概览报表**：生成项目基本信息和资金安排概览
- **资金分析报表**：分析各类资金使用情况和分布
- **趋势分析报表**：分析项目投资和建设趋势

### 4. 数据透视表功能
- 按项目编码和资金性质生成统计报表
- 自动计算债券资金合计（一般债券 + 专项债券）
- 支持Excel导出

## 技术栈

### 后端技术
- **Flask 3.1.2**：Web框架
- **SQLite**：轻量级数据库
- **Pandas 2.3.2**：数据处理和分析
- **OpenPyXL 3.1.5**：Excel文件处理

### 前端技术
- **Bootstrap 5.1.3**：响应式UI框架
- **Chart.js**：图表库
- **Font Awesome 6.0.0**：图标库
- **原生JavaScript**：交互逻辑

## 数据流

```
用户操作 → Web界面 → Flask路由 → 业务逻辑 → 数据库操作 → 返回结果 → 前端展示
```

## API接口设计

### 数据获取接口
- `GET /api/projects` - 获取项目基本信息列表
- `GET /api/funds` - 获取资金安排列表
- `GET /api/pivot` - 获取数据透视表数据

### 数据导出接口
- `GET /api/export/projects` - 导出项目基本信息Excel
- `GET /api/export/funds` - 导出资金安排Excel
- `GET /api/export/pivot` - 导出数据透视表Excel

### 数据导入接口
- `POST /api/upload/projects` - 上传项目基本信息Excel
- `POST /api/upload/funds` - 上传资金安排Excel

## 数据库设计

### 项目基本信息表 (project_basic_info)
```sql
CREATE TABLE project_basic_info (
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
);
```

### 资金安排表 (fund_arrangement)
```sql
CREATE TABLE fund_arrangement (
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
);
```

## 部署说明

### 环境要求
- Python 3.13+
- 依赖包：Flask, Pandas, OpenPyXL

### 启动步骤
1. 安装依赖：`pip3 install --break-system-packages Flask pandas openpyxl`
2. 启动应用：`python3 app.py`
3. 访问系统：http://localhost:5000

### 文件结构
```
AI城建系统/
├── app.py                 # 主应用文件
├── run.py                 # 启动脚本
├── demo.py                # 演示脚本
├── requirements.txt       # 依赖包列表
├── README.md             # 说明文档
├── ARCHITECTURE.md       # 架构说明
├── city_construction.db  # SQLite数据库文件
├── templates/            # HTML模板目录
│   └── index.html       # 主页面模板
├── static/              # 静态文件目录
│   ├── css/
│   │   └── style.css    # 样式文件
│   └── js/
│       └── app.js       # JavaScript文件
└── uploads/             # 上传文件目录
```

## 系统特色

1. **智能数据管理**：支持Excel导入导出，数据操作便捷
2. **可视化分析**：多种图表展示，数据一目了然
3. **智慧报表**：自动生成各类分析报表，支持决策
4. **响应式设计**：支持各种设备访问，用户体验良好
5. **数据透视**：自动计算债券资金合计，统计分析便捷
6. **模拟数据**：系统启动时自动生成200个项目信息和1000条资金安排记录

## 扩展性

系统采用模块化设计，便于后续功能扩展：
- 可增加更多数据表
- 可扩展更多图表类型
- 可增加更多报表模板
- 可集成更多数据分析算法