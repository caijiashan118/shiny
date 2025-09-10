from __future__ import annotations

PROJECT_STATUS_ALLOWED = ["在建", "续建", "完工", "暂停"]

FUND_NATURE_ALLOWED = [
    "一般债券",
    "专项债券",
    "中央综合财力",
    "土地出让金",
    "中央预算内投资",
    "中央补助",
    "省级补助",
]

PROJECT_COLUMNS = [
    "项目名称",
    "项目编码",
    "建设单位",
    "项目主管部门",
    "立项时间",
    "概算批复金额",
    "（预计）开工时间",
    "（预计）完工时间",
    "项目状态",
]

FUNDING_COLUMNS = [
    "项目名称",
    "项目编码",
    "建设单位",
    "项目主管部门",
    "概算批复金额",
    "资金批复金额",
    "批复日期",
    "资金性质",
    "经办人",
    "经办处室",
]

KEY_PROJECT_COLUMNS = ["项目编码", "项目名称"]

DATA_DIR = "/workspace/data"
PROJECTS_CSV = f"{DATA_DIR}/projects.csv"
FUNDINGS_CSV = f"{DATA_DIR}/fundings.csv"
KEY_PROJECTS_CSV = f"{DATA_DIR}/key_projects.csv"
