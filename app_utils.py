import os
import random
from datetime import datetime, timedelta
from io import BytesIO
from typing import Tuple

import numpy as np
import pandas as pd

DATA_DIR = "/workspace/data"
PROJECTS_PATH = os.path.join(DATA_DIR, "projects.xlsx")
FUNDING_PATH = os.path.join(DATA_DIR, "funding.xlsx")

PROJECT_COLUMNS = [
	"项目名称",
	"项目编码",
	"建设单位",
	"项目主管部门",
	"立项时间",
	"概算批复金额(万元)",
	"(预计)开工时间",
	"(预计)完工时间",
	"项目状态",
]

FUNDING_COLUMNS = [
	"项目名称",
	"项目编码",
	"建设单位",
	"项目主管部门",
	"概算批复金额(万元)",
	"资金批复金额(万元)",
	"批复日期",
	"资金性质",
	"经办人",
	"经办处室",
]

ALLOWED_PROJECT_STATUS = ["在建", "续建", "完工", "暂停"]
ALLOWED_FUND_NATURE = [
	"一般债券",
	"专项债券",
	"中央综合财力",
	"土地出让金",
	"中央预算内投资",
	"中央补助",
	"省级补助",
]

RANDOM_SEED = 20240910


def initialize_app_state() -> None:
	os.makedirs(DATA_DIR, exist_ok=True)
	if not os.path.exists(PROJECTS_PATH) or not os.path.exists(FUNDING_PATH):
		projects_df, funding_df = generate_simulated_data()
		save_projects(projects_df)
		save_funding(funding_df)


def load_projects() -> pd.DataFrame:
	if os.path.exists(PROJECTS_PATH):
		df = pd.read_excel(PROJECTS_PATH)
		return df
	return pd.DataFrame(columns=PROJECT_COLUMNS)


def save_projects(df: pd.DataFrame) -> None:
	df = df.copy()
	df.to_excel(PROJECTS_PATH, index=False)


def load_funding() -> pd.DataFrame:
	if os.path.exists(FUNDING_PATH):
		df = pd.read_excel(FUNDING_PATH)
		return df
	return pd.DataFrame(columns=FUNDING_COLUMNS)


def save_funding(df: pd.DataFrame) -> None:
	df = df.copy()
	df.to_excel(FUNDING_PATH, index=False)


def validate_projects(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
	missing_cols = [c for c in PROJECT_COLUMNS if c not in df.columns]
	if missing_cols:
		return False, f"缺少列: {', '.join(missing_cols)}", df

	df = df[PROJECT_COLUMNS].copy()

	# Drop duplicates by 项目编码, keep first
	if df["项目编码"].duplicated().any():
		duplicated_codes = df.loc[df["项目编码"].duplicated(), "项目编码"].unique()
		return False, f"项目编码重复: {', '.join(map(str, duplicated_codes[:10]))}...", df

	# Validate categorical values
	invalid_status = df.loc[~df["项目状态"].isin(ALLOWED_PROJECT_STATUS), "项目状态"].unique()
	if len(invalid_status) > 0:
		return False, f"项目状态非法: {', '.join(map(str, invalid_status))}", df

	# Coerce dates
	for col in ["立项时间", "(预计)开工时间", "(预计)完工时间"]:
		df[col] = pd.to_datetime(df[col], errors="coerce")

	# Numeric
	df["概算批复金额(万元)"] = pd.to_numeric(df["概算批复金额(万元)"], errors="coerce")

	return True, "OK", df


def validate_funding(df: pd.DataFrame, project_codes: pd.Series | list) -> Tuple[bool, str, pd.DataFrame]:
	missing_cols = [c for c in FUNDING_COLUMNS if c not in df.columns]
	if missing_cols:
		return False, f"缺少列: {', '.join(missing_cols)}", df

	df = df[FUNDING_COLUMNS].copy()

	# Validate fund nature
	invalid_nature = df.loc[~df["资金性质"].isin(ALLOWED_FUND_NATURE), "资金性质"].unique()
	if len(invalid_nature) > 0:
		return False, f"资金性质非法: {', '.join(map(str, invalid_nature))}", df

	# Validate project linkage
	code_set = set(pd.Series(project_codes).dropna().astype(str))
	invalid_codes = df.loc[~df["项目编码"].astype(str).isin(code_set), "项目编码"].unique()
	if len(invalid_codes) > 0:
		return False, f"资金安排存在无效项目编码: {', '.join(map(str, invalid_codes[:10]))}...", df

	# Dates and numerics
	df["批复日期"] = pd.to_datetime(df["批复日期"], errors="coerce")
	df["资金批复金额(万元)"] = pd.to_numeric(df["资金批复金额(万元)"], errors="coerce")
	df["概算批复金额(万元)"] = pd.to_numeric(df["概算批复金额(万元)"], errors="coerce")

	return True, "OK", df


def generate_simulated_data(num_projects: int = 200, num_funding: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
	random.seed(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)

	departments = [
		"发展改革局", "住建局", "财政局", "自然资源局", "交通局", "教育局", "卫健委",
	]
	owners = [
		"市政建设公司", "城市发展集团", "新区管委会", "城投公司", "园区开发公司",
	]
	offices = ["投融资处", "预算处", "项目管理处", "基建处"]
	handlers = ["张伟", "王芳", "李强", "刘洋", "陈磊", "杨静", "赵磊", "孙敏"]

	start_year = datetime.now().year - 8

	project_rows = []
	for i in range(num_projects):
		code = f"P{datetime.now().year}{i:04d}"
		name = f"城市建设项目{i+1:03d}"
		owner = random.choice(owners)
		dept = random.choice(departments)
		approval_date = datetime(start_year + random.randint(0, 6), random.randint(1, 12), random.randint(1, 28))
		estimate_million = int(random.randint(500, 20000))
		start_date = approval_date + timedelta(days=random.randint(30, 365))
		end_date = start_date + timedelta(days=random.randint(180, 1200))
		status = random.choice(ALLOWED_PROJECT_STATUS)
		project_rows.append([
			name,
			code,
			owner,
			dept,
			approval_date.date(),
			estimate_million,
			start_date.date(),
			end_date.date(),
			status,
		])

	projects_df = pd.DataFrame(project_rows, columns=PROJECT_COLUMNS)

	# Funding records
	nature_weights = np.array([3, 2, 1, 1, 1, 1, 1], dtype=float)
	nature_weights = nature_weights / nature_weights.sum()

	funding_rows = []
	for _ in range(num_funding):
		proj = projects_df.sample(1).iloc[0]
		proj_code = proj["项目编码"]
		proj_name = proj["项目名称"]
		owner = proj["建设单位"]
		dept = proj["项目主管部门"]
		estimate = float(proj["概算批复金额(万元)"])
		nature = np.random.choice(ALLOWED_FUND_NATURE, p=nature_weights)
		# Amount between 10 and 8% of estimate, capped
		max_amt = max(10.0, estimate * 0.08)
		amount = float(np.random.uniform(10.0, max_amt))
		approval_date = proj["(预计)开工时间"] + timedelta(days=int(np.random.uniform(0, 365)))
		handler = random.choice(handlers)
		office = random.choice(offices)
		funding_rows.append([
			proj_name,
			proj_code,
			owner,
			dept,
			estimate,
			round(amount, 2),
			pd.to_datetime(approval_date).date(),
			nature,
			handler,
			office,
		])

	funding_df = pd.DataFrame(funding_rows, columns=FUNDING_COLUMNS)

	return projects_df, funding_df


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
	buffer = BytesIO()
	with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
		df.to_excel(writer, sheet_name=sheet_name, index=False)
	return buffer.getvalue()


def build_pivot_table(funding_df: pd.DataFrame) -> pd.DataFrame:
	if funding_df.empty:
		return pd.DataFrame()

	pivot = pd.pivot_table(
		funding_df,
		index=["项目编码", "项目名称"],
		columns="资金性质",
		values="资金批复金额(万元)",
		aggfunc="sum",
		fill_value=0.0,
	)
	pivot = pivot.reset_index()

	# Ensure all nature columns exist
	for col in ALLOWED_FUND_NATURE:
		if col not in pivot.columns:
			pivot[col] = 0.0

	pivot["一般债券合计(万元)"] = pivot.get("一般债券", 0.0)
	pivot["专项债券合计(万元)"] = pivot.get("专项债券", 0.0)
	pivot["债券资金合计(万元)"] = pivot["一般债券合计(万元)"] + pivot["专项债券合计(万元)"]

	# Sort by 债券资金合计 desc for readability
	pivot = pivot.sort_values(by=["债券资金合计(万元)", "项目编码"], ascending=[False, True])
	return pivot