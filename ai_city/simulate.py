from __future__ import annotations

import random
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker

from .constants import (
	FUNDING_COLUMNS,
	FUND_NATURE_ALLOWED,
	PROJECT_COLUMNS,
	PROJECT_STATUS_ALLOWED,
)

fake = Faker("zh_CN")


def _random_code(existing: set[str]) -> str:
	while True:
		code = f"XM{random.randint(100000, 999999)}"
		if code not in existing:
			return code


def simulate_projects(n: int = 200, seed: int = 42) -> pd.DataFrame:
	random.seed(seed)
	np.random.seed(seed)
	project_names = []
	project_codes = []
	construction_units = []
	owner_depts = []
	approve_dates = []
	budget_amounts = []
	start_dates = []
	end_dates = []
	statuses = []

	existing_codes: set[str] = set()
	for _ in range(n):
		name = f"{fake.city_name()}基础设施提升项目{random.choice(['A', 'B', 'C', 'D'])}期"
		code = _random_code(existing_codes)
		existing_codes.add(code)
		unit = f"{fake.company()}"
		dept = random.choice(["住建局", "发改委", "城管局", "区政府", "财政局"])
		approve_date = fake.date_between(start_date="-10y", end_date="-1y")
		budget = float(np.random.randint(500, 20001))
		start_date = approve_date + timedelta(days=random.randint(30, 365))
		end_date = start_date + timedelta(days=random.randint(180, 1200))
		status = random.choice(PROJECT_STATUS_ALLOWED)

		project_names.append(name)
		project_codes.append(code)
		construction_units.append(unit)
		owner_depts.append(dept)
		approve_dates.append(pd.to_datetime(approve_date).date())
		budget_amounts.append(budget)
		start_dates.append(pd.to_datetime(start_date).date())
		end_dates.append(pd.to_datetime(end_date).date())
		statuses.append(status)

	df = pd.DataFrame(
		{
			"项目名称": project_names,
			"项目编码": project_codes,
			"建设单位": construction_units,
			"项目主管部门": owner_depts,
			"立项时间": approve_dates,
			"概算批复金额": budget_amounts,
			"（预计）开工时间": start_dates,
			"（预计）完工时间": end_dates,
			"项目状态": statuses,
		}
	)
	return df.reindex(columns=PROJECT_COLUMNS)


def simulate_fundings(projects_df: pd.DataFrame, n: int = 1000, seed: int = 42) -> pd.DataFrame:
	random.seed(seed)
	np.random.seed(seed)
	if projects_df.empty:
		projects_df = simulate_projects(200, seed)

	rows = []
	project_index = projects_df.reset_index(drop=True)
	for _ in range(n):
		proj = project_index.sample(1).iloc[0]
		proj_name = proj["项目名称"]
		proj_code = proj["项目编码"]
		unit = proj["建设单位"]
		dept = proj["项目主管部门"]
		budget = float(proj["概算批复金额"]) if pd.notna(proj["概算批复金额"]) else float(np.random.randint(500, 20001))
		fund_nature = random.choice(FUND_NATURE_ALLOWED)
		approve_date = fake.date_between(start_date=proj["立项时间"], end_date="today")
		amount = float(max(10, np.random.normal(loc=min(800.0, budget / 5.0), scale=300.0)))
		amount = float(np.round(amount, 2))
		handler = fake.name()
		dept_room = random.choice(["综合处", "投融资处", "项目管理处", "预算处", "资金处"])

		rows.append(
			{
				"项目名称": proj_name,
				"项目编码": proj_code,
				"建设单位": unit,
				"项目主管部门": dept,
				"概算批复金额": budget,
				"资金批复金额": amount,
				"批复日期": pd.to_datetime(approve_date).date(),
				"资金性质": fund_nature,
				"经办人": handler,
				"经办处室": dept_room,
			}
		)

	df = pd.DataFrame(rows)
	return df.reindex(columns=FUNDING_COLUMNS)
